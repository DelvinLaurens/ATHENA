import argparse
import math
import os
from datetime import datetime, timedelta

import pandas as pd


DEFAULT_SCORE_BINS = [0, 50, 60, 65, 68, 70, 72, 75, 80, 85, 90, 95, 100]
DEFAULT_COLUMNS = [
    'timestamp',
    'symbol',
    'signal',
    'price_at_pred',
    'entry_price',
    'exit_price',
    'ai_score',
    'risk_level',
    'vol_pct',
    'change_4h',
    'is_top_opportunity',
    'is_scalper_hotlist',
    'is_validated',
    'return_pct',
    'result',
]


def parse_csv(value, cast=str):
    if value is None or value == '':
        return []

    return [cast(item.strip()) for item in value.split(',') if item.strip()]


def parse_symbols(value):
    symbols = parse_csv(value, str)
    if not symbols:
        return None

    return {symbol.upper() for symbol in symbols}


def parse_score_bands(value):
    if not value:
        return None

    bands = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue

        lower, upper = item.split(':', maxsplit=1)
        bands.append((float(lower), float(upper)))
    return bands


def score_in_bands(series, score_bands):
    if not score_bands:
        return pd.Series(True, index=series.index)

    mask = pd.Series(False, index=series.index)
    for lower, upper in score_bands:
        mask = mask | ((series >= lower) & (series < upper))
    return mask


def parse_result_series(series):
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({
        'true': 1,
        'win': 1,
        'hit': 1,
        '1': 1,
        '1.0': 1,
        'false': 0,
        'loss': 0,
        'miss': 0,
        '0': 0,
        '0.0': 0,
    })
    numeric = pd.to_numeric(series, errors='coerce')
    return numeric.where(numeric.notna(), mapped)


def parse_bool_series(series):
    return series.astype(str).str.strip().str.lower().isin(['true', '1', '1.0', 'yes', 'y'])


def profit_factor(group):
    if group.empty:
        return 0.0

    gross_profit = group.loc[group['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(group.loc[group['pnl'] < 0, 'pnl'].sum())
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def format_profit_factor(value):
    if isinstance(value, str):
        return value
    if math.isinf(value):
        return 'inf'
    return round(float(value), 4)


def load_log(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)

    df = pd.read_csv(log_path)
    for column in DEFAULT_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[DEFAULT_COLUMNS].copy()


def normalize_log(df, position_size, fee_pct, slippage_pct):
    normalized_df = df.copy()
    normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'], errors='coerce')
    normalized_df['symbol'] = normalized_df['symbol'].astype(str).str.upper()
    normalized_df['signal'] = normalized_df['signal'].astype(str).str.upper()
    normalized_df['risk_level'] = normalized_df['risk_level'].fillna('UNKNOWN').astype(str).str.upper()
    normalized_df['ai_score'] = pd.to_numeric(normalized_df['ai_score'], errors='coerce')
    normalized_df['return_pct'] = pd.to_numeric(normalized_df['return_pct'], errors='coerce')
    normalized_df['result_num'] = parse_result_series(normalized_df['result'])
    normalized_df['is_top_opportunity_bool'] = parse_bool_series(normalized_df['is_top_opportunity'])
    normalized_df['is_scalper_hotlist_bool'] = parse_bool_series(normalized_df['is_scalper_hotlist'])

    round_trip_cost_pct = (fee_pct + slippage_pct) * 2 * 100
    normalized_df['net_return_pct'] = normalized_df['return_pct'] - round_trip_cost_pct
    normalized_df['pnl'] = position_size * (normalized_df['net_return_pct'] / 100)
    normalized_df['net_is_win'] = normalized_df['net_return_pct'] > 0
    return normalized_df


def apply_filters(df, args):
    filtered_df = df.copy()

    if args.days is not None:
        cutoff = datetime.now() - timedelta(days=args.days)
        filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff]

    include_symbols = parse_symbols(args.include_symbols)
    exclude_symbols = parse_symbols(args.exclude_symbols)
    if include_symbols:
        filtered_df = filtered_df[filtered_df['symbol'].isin(include_symbols)]
    if exclude_symbols:
        filtered_df = filtered_df[~filtered_df['symbol'].isin(exclude_symbols)]

    if args.risk_level:
        risk_levels = {item.upper() for item in parse_csv(args.risk_level, str)}
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_levels)]

    if args.signal:
        signals = {item.upper() for item in parse_csv(args.signal, str)}
        filtered_df = filtered_df[filtered_df['signal'].isin(signals)]

    if args.min_score is not None:
        filtered_df = filtered_df[filtered_df['ai_score'] >= args.min_score]
    if args.max_score is not None:
        filtered_df = filtered_df[filtered_df['ai_score'] < args.max_score]

    score_bands = parse_score_bands(args.trade_score_bands)
    if score_bands:
        filtered_df = filtered_df[score_in_bands(filtered_df['ai_score'], score_bands)]

    return filtered_df


def add_score_bins(df, bins):
    if df.empty:
        df = df.copy()
        df['score_bin'] = None
        return df

    score_df = df.copy()
    labels = [f'{bins[index]:g}-{bins[index + 1]:g}' for index in range(len(bins) - 1)]
    score_values = score_df['ai_score'].clip(lower=bins[0], upper=bins[-1] - 0.000001)
    score_df['score_bin'] = pd.cut(
        score_values,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return score_df


def calculate_metrics(group):
    result_group = group[group['result_num'].notna()]
    return_group = group[group['return_pct'].notna()]

    samples = int(len(result_group))
    wins = int((result_group['result_num'] > 0).sum())
    losses = samples - wins
    return_samples = int(len(return_group))

    pf = profit_factor(return_group)
    gross_profit = return_group.loc[return_group['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(return_group.loc[return_group['pnl'] < 0, 'pnl'].sum())
    net_wins = int(return_group['net_is_win'].sum())

    return {
        'samples': samples,
        'wins': wins,
        'losses': losses,
        'win_rate_pct': round((wins / samples) * 100, 2) if samples else 0.0,
        'return_samples': return_samples,
        'return_coverage_pct': round((return_samples / samples) * 100, 2) if samples else 0.0,
        'net_win_rate_pct': round((net_wins / return_samples) * 100, 2) if return_samples else 0.0,
        'avg_score': round(float(result_group['ai_score'].mean()), 2) if samples else 0.0,
        'avg_gross_return_pct': round(float(return_group['return_pct'].mean()), 4) if return_samples else 0.0,
        'avg_net_return_pct': round(float(return_group['net_return_pct'].mean()), 4) if return_samples else 0.0,
        'gross_profit': round(float(gross_profit), 4),
        'gross_loss': round(float(gross_loss), 4),
        'profit_factor': format_profit_factor(pf),
        'net_pnl': round(float(return_group['pnl'].sum()), 4) if return_samples else 0.0,
    }


def build_group_report(df, group_col, label_col):
    if df.empty:
        return pd.DataFrame()

    rows = []
    for label, group in df.groupby(group_col, observed=False, dropna=False):
        if group.empty:
            continue

        label_value = 'UNKNOWN' if pd.isna(label) else str(label)
        row = {label_col: label_value}
        row.update(calculate_metrics(group))
        rows.append(row)

    return pd.DataFrame(rows)


def build_special_report(df):
    rows = []
    for label, column in [
        ('TOP', 'is_top_opportunity_bool'),
        ('SCALPER', 'is_scalper_hotlist_bool'),
        ('REGULAR', None),
    ]:
        group = df[~df['is_top_opportunity_bool'] & ~df['is_scalper_hotlist_bool']] if column is None else df[df[column]]
        row = {'flag': label}
        row.update(calculate_metrics(group))
        rows.append(row)

    return pd.DataFrame(rows)


def build_blacklist_candidates(symbol_report, min_symbol_samples):
    if symbol_report.empty:
        return symbol_report

    candidates = symbol_report[symbol_report['samples'] >= min_symbol_samples].copy()
    if candidates.empty:
        return candidates

    pf_numeric = pd.to_numeric(candidates['profit_factor'].replace('inf', math.inf), errors='coerce').fillna(0)
    candidates['_pf_sort'] = pf_numeric
    candidates = candidates[
        (candidates['win_rate_pct'] < 45)
        | (candidates['_pf_sort'] < 1)
        | (candidates['net_pnl'] < 0)
    ]
    if candidates.empty:
        return candidates.drop(columns=['_pf_sort'], errors='ignore')

    return (
        candidates
        .sort_values(['net_pnl', 'win_rate_pct', '_pf_sort'], ascending=[True, True, True])
        .drop(columns=['_pf_sort'])
    )


def sort_by_profit_factor(df):
    if df.empty or 'profit_factor' not in df.columns:
        return df

    sorted_df = df.copy()
    sorted_df['_pf_sort'] = pd.to_numeric(
        sorted_df['profit_factor'].replace('inf', math.inf),
        errors='coerce',
    ).fillna(0)
    sorted_df = sorted_df.sort_values(
        ['_pf_sort', 'samples', 'net_pnl'],
        ascending=[False, False, False],
    )
    return sorted_df.drop(columns=['_pf_sort'])


def print_table(title, df, max_rows=None):
    print()
    print(title)
    print('=' * len(title))
    if df.empty:
        print('No rows.')
        return

    display_df = df.head(max_rows) if max_rows else df
    print(display_df.to_string(index=False))


def run_analysis(args):
    raw_df = load_log(args.log_path)
    normalized_df = normalize_log(raw_df, args.position_size, args.fee_pct, args.slippage_pct)
    filtered_df = apply_filters(normalized_df, args)
    validated_df = filtered_df[filtered_df['result_num'].notna()].copy()
    scored_df = add_score_bins(validated_df, parse_csv(args.score_bins, float) or DEFAULT_SCORE_BINS)

    score_report = build_group_report(scored_df, 'score_bin', 'score_bin')
    symbol_report = build_group_report(scored_df, 'symbol', 'symbol')
    risk_report = build_group_report(scored_df, 'risk_level', 'risk_level')
    signal_report = build_group_report(scored_df, 'signal', 'signal')
    special_report = build_special_report(scored_df)
    blacklist_candidates = build_blacklist_candidates(symbol_report, args.min_symbol_samples)

    if not symbol_report.empty:
        symbol_report = sort_by_profit_factor(symbol_report)

    return {
        'raw_rows': len(raw_df),
        'filtered_rows': len(filtered_df),
        'validated_rows': len(validated_df),
        'return_rows': int(validated_df['return_pct'].notna().sum()),
        'overall': pd.DataFrame([calculate_metrics(validated_df)]),
        'score_bands': score_report,
        'symbols': symbol_report,
        'risk_levels': risk_report,
        'signals': signal_report,
        'special_flags': special_report,
        'blacklist_candidates': blacklist_candidates,
    }


def write_outputs(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(os.path.join(output_dir, f'{key}.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ATHENA live prediction logs')
    parser.add_argument('--log-path', default=os.path.join('data', 'predictions_log.csv'))
    parser.add_argument('--days', type=int, help='Analyze only the last N days')
    parser.add_argument('--score-bins', default='0,50,60,65,68,70,72,75,80,85,90,95,100')
    parser.add_argument('--risk-level', help='Comma-separated risk filters, e.g. LOW,MEDIUM')
    parser.add_argument('--signal', help='Comma-separated signal filters, e.g. LONG,SHORT')
    parser.add_argument('--include-symbols')
    parser.add_argument('--exclude-symbols')
    parser.add_argument('--min-score', type=float)
    parser.add_argument('--max-score', type=float)
    parser.add_argument('--trade-score-bands', help='Filter rows by score ranges, e.g. 68:75,80:100')
    parser.add_argument('--position-size', type=float, default=100.0)
    parser.add_argument('--fee-pct', type=float, default=0.001)
    parser.add_argument('--slippage-pct', type=float, default=0.0)
    parser.add_argument('--min-symbol-samples', type=int, default=10)
    parser.add_argument('--top-symbols', type=int, default=20)
    parser.add_argument('--output-dir')
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_analysis(args)

    print('ATHENA Live Prediction Log Analyzer')
    print('====================================')
    print(f"Rows loaded: {result['raw_rows']}")
    print(f"Rows after filters: {result['filtered_rows']}")
    print(f"Validated rows: {result['validated_rows']}")
    print(f"Rows with return_pct: {result['return_rows']}")
    print(f"Round-trip cost: {round((args.fee_pct + args.slippage_pct) * 2 * 100, 4)}%")

    print_table('Overall', result['overall'])
    print_table('Score Bands', result['score_bands'])
    print_table('Risk Levels', result['risk_levels'])
    print_table('Signals', result['signals'])
    print_table('Top And Scalper Flags', result['special_flags'])
    print_table('Best Symbols', result['symbols'], max_rows=args.top_symbols)
    print_table('Blacklist Candidates', result['blacklist_candidates'], max_rows=args.top_symbols)

    if args.output_dir:
        write_outputs(result, args.output_dir)
        print()
        print(f"Saved CSV reports to: {args.output_dir}")


if __name__ == '__main__':
    main()
