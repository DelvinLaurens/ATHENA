import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from src.brain.backtester import AthenaBacktester, BacktestConfig, parse_symbols
from src.brain.live_log_analyzer import (
    calculate_metrics as calculate_live_metrics,
    load_log,
    normalize_log,
)


@dataclass(frozen=True)
class Policy:
    name: str
    score_bands: tuple[tuple[float, float], ...]
    risk_levels: frozenset[str] | None = None

    @property
    def score_bands_label(self):
        return ','.join(f'{lower:g}:{upper:g}' for lower, upper in self.score_bands)

    @property
    def risk_label(self):
        if not self.risk_levels:
            return 'ALL'

        return ','.join(sorted(self.risk_levels))


DEFAULT_POLICIES = [
    Policy('current_v1_rc', ((68, 75), (80, 100)), frozenset({'LOW'})),
    Policy('low_tight_68_72_80_85', ((68, 72), (80, 85)), frozenset({'LOW'})),
    Policy('low_68_70', ((68, 70),), frozenset({'LOW'})),
    Policy('low_70_72', ((70, 72),), frozenset({'LOW'})),
    Policy('low_72_75', ((72, 75),), frozenset({'LOW'})),
    Policy('all_68_70_72_75', ((68, 70), (72, 75)), None),
    Policy('low_medium_68_70_72_75', ((68, 70), (72, 75)), frozenset({'LOW', 'MEDIUM'})),
    Policy('medium_68_70_72_75', ((68, 70), (72, 75)), frozenset({'MEDIUM'})),
    Policy('high_68_70_72_75', ((68, 70), (72, 75)), frozenset({'HIGH'})),
    Policy('all_80_85', ((80, 85),), None),
]


def parse_csv(value, cast=str):
    if value is None or value == '':
        return []

    return [cast(item.strip()) for item in value.split(',') if item.strip()]


def parse_score_bands(value):
    if not value:
        return tuple()

    bands = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue

        lower, upper = item.split(':', maxsplit=1)
        bands.append((float(lower), float(upper)))
    return tuple(bands)


def parse_policy(value):
    parts = [part.strip() for part in value.split('|')]
    if len(parts) != 3:
        raise ValueError('Policy format harus: name|score_bands|risk_levels, contoh: my_policy|68:70,72:75|LOW,MEDIUM')

    name, score_bands_value, risk_levels_value = parts
    score_bands = parse_score_bands(score_bands_value)
    if not name or not score_bands:
        raise ValueError('Policy custom harus punya name dan minimal satu score band.')

    risk_levels = None
    if risk_levels_value and risk_levels_value.upper() not in ['ALL', 'NONE', '*']:
        risk_levels = frozenset(item.upper() for item in parse_csv(risk_levels_value, str))

    return Policy(name, score_bands, risk_levels)


def select_policies(args):
    policies = list(DEFAULT_POLICIES)
    for custom_policy in args.policy or []:
        policies.append(parse_policy(custom_policy))

    only_policies = {item for item in parse_csv(args.only_policies, str)}
    if only_policies:
        policies = [policy for policy in policies if policy.name in only_policies]

    if not policies:
        raise ValueError('Tidak ada policy yang dipilih.')

    return policies


def policy_min_score(policies):
    return min(lower for policy in policies for lower, _ in policy.score_bands)


def score_mask(df, score_bands):
    if df.empty:
        return pd.Series(False, index=df.index)

    mask = pd.Series(False, index=df.index)
    for lower, upper in score_bands:
        mask = mask | ((df['ai_score'] >= lower) & (df['ai_score'] < upper))
    return mask


def risk_mask(df, risk_levels):
    if df.empty:
        return pd.Series(False, index=df.index)
    if not risk_levels:
        return pd.Series(True, index=df.index)

    return df['risk_level'].astype(str).str.upper().isin(risk_levels)


def filter_policy_rows(df, policy):
    if df.empty:
        return df.copy()

    return df[score_mask(df, policy.score_bands) & risk_mask(df, policy.risk_levels)].copy()


def profit_factor_to_float(value):
    if isinstance(value, str) and value.lower() == 'inf':
        return math.inf

    numeric_value = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    if pd.isna(numeric_value):
        return 0.0
    return float(numeric_value)


def capped_metric(value, cap=5.0):
    numeric_value = profit_factor_to_float(value)
    if math.isinf(numeric_value):
        return cap
    return max(0.0, min(numeric_value, cap))


def build_backtest_base(args, policies):
    min_score = args.min_backtest_score
    if min_score is None:
        min_score = policy_min_score(policies)

    config = BacktestConfig(
        data_folder=args.data_folder,
        initial_balance=args.initial_balance,
        position_size=args.position_size,
        ai_score_threshold=min_score,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        train_size=args.train_size,
        max_steps=args.max_steps,
        risk_filter=None,
        model_mode=args.model,
        confirmation_threshold=args.confirm_threshold,
        use_alt_market_proxy=args.use_alt_market_proxy,
        score_bands=None,
        include_symbols=parse_symbols(args.include_symbols),
        exclude_symbols=parse_symbols(args.exclude_symbols),
    )
    backtester = AthenaBacktester(config)
    if args.symbol:
        result = backtester.backtest_symbol(args.symbol)
    else:
        result = backtester.backtest_folder(max_symbols=args.max_symbols)

    return backtester, result['trades']


def compare_backtest_policies(args, policies):
    backtester, base_trades = build_backtest_base(args, policies)
    rows = []
    for policy in policies:
        policy_trades = filter_policy_rows(base_trades, policy)
        policy_trades = backtester.recalculate_equity(policy_trades)
        metrics = backtester.calculate_metrics(policy_trades, policy.name)
        rows.append({
            'policy': policy.name,
            'backtest_trades': metrics['trades'],
            'backtest_wins': metrics['wins'],
            'backtest_losses': metrics['losses'],
            'backtest_win_rate_pct': metrics['win_rate_pct'],
            'backtest_profit_factor': metrics['profit_factor'],
            'backtest_max_drawdown_pct': metrics['max_drawdown_pct'],
            'backtest_net_profit': metrics['net_profit'],
            'backtest_total_return_pct': metrics['total_return_pct'],
        })

    return pd.DataFrame(rows)


def build_live_base(args):
    raw_df = load_log(args.log_path)
    live_df = normalize_log(raw_df, args.position_size, args.fee_pct, args.slippage_pct)
    live_df = live_df[live_df['result_num'].notna()].copy()

    if args.days is not None:
        cutoff = datetime.now() - timedelta(days=args.days)
        live_df = live_df[live_df['timestamp'] >= cutoff]

    include_symbols = parse_symbols(args.include_symbols)
    exclude_symbols = parse_symbols(args.exclude_symbols)
    if include_symbols:
        live_df = live_df[live_df['symbol'].isin(include_symbols)]
    if exclude_symbols:
        live_df = live_df[~live_df['symbol'].isin(exclude_symbols)]

    if args.live_signal:
        signals = {signal.upper() for signal in parse_csv(args.live_signal, str)}
        live_df = live_df[live_df['signal'].isin(signals)]

    return live_df


def compare_live_policies(args, policies):
    live_df = build_live_base(args)
    rows = []
    for policy in policies:
        policy_rows = filter_policy_rows(live_df, policy)
        metrics = calculate_live_metrics(policy_rows)
        rows.append({
            'policy': policy.name,
            'live_samples': metrics['samples'],
            'live_wins': metrics['wins'],
            'live_losses': metrics['losses'],
            'live_win_rate_pct': metrics['win_rate_pct'],
            'live_return_samples': metrics['return_samples'],
            'live_net_win_rate_pct': metrics['net_win_rate_pct'],
            'live_profit_factor': metrics['profit_factor'],
            'live_net_pnl': metrics['net_pnl'],
            'live_avg_net_return_pct': metrics['avg_net_return_pct'],
        })

    return pd.DataFrame(rows)


def score_policy(row, args):
    backtest_pf = capped_metric(row.get('backtest_profit_factor', 0.0))
    live_pf = capped_metric(row.get('live_profit_factor', 0.0))
    backtest_trades = float(row.get('backtest_trades', 0.0) or 0.0)
    live_samples = float(row.get('live_samples', 0.0) or 0.0)
    max_drawdown = float(row.get('backtest_max_drawdown_pct', 0.0) or 0.0)

    trade_score = min(backtest_trades / args.target_backtest_trades, 1.0)
    live_sample_score = min(live_samples / args.target_live_samples, 1.0)
    return round(
        (backtest_pf / 5.0) * 35
        + (live_pf / 5.0) * 35
        + trade_score * 15
        + live_sample_score * 15
        - min(max_drawdown, 10) * 0.5,
        4,
    )


def has_value(row, key):
    return key in row.index and not pd.isna(row.get(key))


def grade_policy(row, args):
    has_backtest = has_value(row, 'backtest_profit_factor') and has_value(row, 'backtest_trades')
    has_live = has_value(row, 'live_profit_factor') and has_value(row, 'live_samples')

    backtest_pf = profit_factor_to_float(row.get('backtest_profit_factor', 0.0)) if has_backtest else None
    live_pf = profit_factor_to_float(row.get('live_profit_factor', 0.0)) if has_live else None
    backtest_trades = int(row.get('backtest_trades', 0) or 0) if has_backtest else 0
    live_samples = int(row.get('live_samples', 0) or 0) if has_live else 0

    backtest_pass = (
        has_backtest
        and
        backtest_trades >= args.target_backtest_trades
        and backtest_pf >= args.target_backtest_pf
    )
    live_pass = (
        has_live
        and
        live_samples >= args.target_live_samples
        and live_pf >= args.target_live_pf
    )

    if has_backtest and has_live and backtest_pass and live_pass:
        return 'PASS'
    if has_backtest and not has_live and backtest_pass:
        return 'PASS'
    if has_live and not has_backtest and live_pass:
        return 'PASS'
    if (has_backtest and backtest_pf < 1) or (has_live and live_pf < 1):
        return 'REJECT'
    if backtest_pass or live_pass:
        return 'WATCH'
    return 'RESEARCH'


def combine_reports(policies, backtest_df=None, live_df=None):
    policy_df = pd.DataFrame([{
        'policy': policy.name,
        'score_bands': policy.score_bands_label,
        'risk_levels': policy.risk_label,
    } for policy in policies])

    combined_df = policy_df
    if backtest_df is not None:
        combined_df = combined_df.merge(backtest_df, on='policy', how='left')
    if live_df is not None:
        combined_df = combined_df.merge(live_df, on='policy', how='left')
    return combined_df


def rank_policies(combined_df, args):
    if combined_df.empty:
        return combined_df

    ranked_df = combined_df.copy()
    for column in ['backtest_trades', 'live_samples']:
        if column not in ranked_df.columns:
            ranked_df[column] = 0

    ranked_df['policy_score'] = ranked_df.apply(lambda row: score_policy(row, args), axis=1)
    ranked_df['grade'] = ranked_df.apply(lambda row: grade_policy(row, args), axis=1)
    grade_rank = {'PASS': 0, 'WATCH': 1, 'RESEARCH': 2, 'REJECT': 3}
    ranked_df['_grade_rank'] = ranked_df['grade'].map(grade_rank).fillna(9)
    ranked_df = ranked_df.sort_values(
        ['_grade_rank', 'policy_score', 'backtest_trades', 'live_samples'],
        ascending=[True, False, False, False],
    )
    return ranked_df.drop(columns=['_grade_rank'])


def print_table(title, df):
    print()
    print(title)
    print('=' * len(title))
    if df.empty:
        print('No rows.')
        return

    print(df.to_string(index=False))


def write_outputs(ranked_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ranked_df.to_csv(os.path.join(output_dir, 'policy_ranking.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare ATHENA policies across historical backtest and live prediction log')
    parser.add_argument('--data-folder', default=os.path.join('data', 'raw', '4h'))
    parser.add_argument('--log-path', default=os.path.join('data', 'predictions_log.csv'))
    parser.add_argument('--symbol')
    parser.add_argument('--max-symbols', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--train-size', type=int, default=700)
    parser.add_argument('--model', default='xgb', choices=['xgb', 'rf', 'ensemble', 'xgb_rf_confirm'])
    parser.add_argument('--confirm-threshold', type=float, default=50.0)
    parser.add_argument('--use-alt-market-proxy', action='store_true')
    parser.add_argument('--min-backtest-score', type=float)
    parser.add_argument('--initial-balance', type=float, default=1000.0)
    parser.add_argument('--position-size', type=float, default=100.0)
    parser.add_argument('--fee-pct', type=float, default=0.001)
    parser.add_argument('--slippage-pct', type=float, default=0.0)
    parser.add_argument('--include-symbols')
    parser.add_argument('--exclude-symbols')
    parser.add_argument('--days', type=int, help='Filter live prediction log to the last N days')
    parser.add_argument('--live-signal', default='LONG')
    parser.add_argument('--policy', action='append', help='Custom policy: name|score_bands|risk_levels, e.g. my_policy|68:70,72:75|LOW,MEDIUM')
    parser.add_argument('--only-policies', help='Comma-separated policy names to run')
    parser.add_argument('--skip-backtest', action='store_true')
    parser.add_argument('--skip-live', action='store_true')
    parser.add_argument('--target-backtest-pf', type=float, default=1.3)
    parser.add_argument('--target-live-pf', type=float, default=1.2)
    parser.add_argument('--target-backtest-trades', type=int, default=50)
    parser.add_argument('--target-live-samples', type=int, default=30)
    parser.add_argument('--output-dir')
    return parser.parse_args()


def main():
    args = parse_args()
    policies = select_policies(args)

    backtest_df = None
    live_df = None
    if not args.skip_backtest:
        backtest_df = compare_backtest_policies(args, policies)
    if not args.skip_live:
        live_df = compare_live_policies(args, policies)

    ranked_df = rank_policies(combine_reports(policies, backtest_df, live_df), args)
    print_table('ATHENA Policy Comparator', ranked_df)

    if args.output_dir:
        write_outputs(ranked_df, args.output_dir)
        print()
        print(f'Saved policy ranking to: {os.path.join(args.output_dir, "policy_ranking.csv")}')


if __name__ == '__main__':
    main()
