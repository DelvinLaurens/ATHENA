import argparse
import math
import os

import pandas as pd

from src.brain.backtester import AthenaBacktester, BacktestConfig, parse_score_bands, parse_symbols


DEFAULT_EXCLUDE_SYMBOLS = (
    'HBAR/USDT,SOL/USDT,LINK/USDT,DOGE/USDT,PEPE/USDT,AVAX/USDT,'
    'ENA/USDT,FIL/USDT,TRUMP/USDT,UNI/USDT,WLD/USDT'
)
DEFAULT_SCORE_BINS = [0, 50, 60, 65, 68, 70, 72, 75, 80, 85, 90, 95, 100]


def parse_csv(value, cast=str):
    if value is None or value == '':
        return []

    return [cast(item.strip()) for item in value.split(',') if item.strip()]


def parse_bool_csv(value):
    return [
        item.lower() in ['1', 'true', 'yes', 'on']
        for item in parse_csv(value, str)
    ]


def profit_factor(trades_df):
    if trades_df.empty:
        return 0.0

    gross_profit = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum())
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def build_score_calibration(trades_df, bins=None):
    bins = bins or DEFAULT_SCORE_BINS
    if trades_df.empty:
        return pd.DataFrame(columns=[
            'score_bin',
            'samples',
            'wins',
            'losses',
            'win_rate_pct',
            'avg_return_pct',
            'net_pnl',
            'profit_factor',
        ])

    scored_df = trades_df.copy()
    scored_df['score_bin'] = pd.cut(
        scored_df['ai_score'],
        bins=bins,
        include_lowest=True,
        right=False,
    )

    rows = []
    for score_bin, group in scored_df.groupby('score_bin', observed=False):
        if group.empty:
            continue

        wins = int(group['is_win'].sum())
        samples = int(len(group))
        pf = profit_factor(group)
        rows.append({
            'score_bin': str(score_bin),
            'samples': samples,
            'wins': wins,
            'losses': samples - wins,
            'win_rate_pct': round((wins / samples) * 100, 2),
            'avg_return_pct': round(float(group['return_pct'].mean()), 4),
            'net_pnl': round(float(group['pnl'].sum()), 4),
            'profit_factor': round(float(pf), 4) if math.isfinite(pf) else 'inf',
        })

    return pd.DataFrame(rows)


def run_threshold_suite(backtester, thresholds, symbol=None, max_symbols=None):
    optimizer_df = backtester.optimize_thresholds(
        thresholds,
        symbol=symbol,
        max_symbols=max_symbols,
    )
    return optimizer_df


def run_experiments(args):
    thresholds = parse_csv(args.thresholds, float)
    models = parse_csv(args.models, str)
    risk_levels = parse_csv(args.risk_levels, str)
    max_steps_values = parse_csv(args.max_steps_values, int)
    confirm_thresholds = parse_csv(args.confirm_thresholds, float)
    alt_proxy_values = parse_bool_csv(args.use_alt_market_proxy_values)

    if not thresholds:
        raise ValueError('Minimal satu threshold harus diberikan.')
    if not models:
        raise ValueError('Minimal satu model harus diberikan.')
    if not risk_levels:
        risk_levels = [None]
    if not max_steps_values:
        max_steps_values = [None]
    if not confirm_thresholds:
        confirm_thresholds = [50.0]
    if not alt_proxy_values:
        alt_proxy_values = [False]

    include_symbols = parse_symbols(args.include_symbols)
    exclude_symbols = parse_symbols(args.exclude_symbols)
    score_bands = parse_score_bands(args.trade_score_bands)
    rows = []

    for max_steps in max_steps_values:
        for risk_level in risk_levels:
            normalized_risk = None if risk_level.upper() in ['NONE', 'ALL'] else risk_level.upper()
            for model_mode in models:
                model_mode = model_mode.lower()
                model_confirm_thresholds = (
                    confirm_thresholds if model_mode == 'xgb_rf_confirm' else [confirm_thresholds[0]]
                )
                for confirm_threshold in model_confirm_thresholds:
                    for use_alt_market_proxy in alt_proxy_values:
                        config = BacktestConfig(
                            data_folder=args.data_folder,
                            initial_balance=args.initial_balance,
                            position_size=args.position_size,
                            ai_score_threshold=min(thresholds),
                            fee_pct=args.fee_pct,
                            slippage_pct=args.slippage_pct,
                            train_size=args.train_size,
                            max_steps=max_steps,
                            risk_filter=normalized_risk,
                            model_mode=model_mode,
                            confirmation_threshold=confirm_threshold,
                            use_alt_market_proxy=use_alt_market_proxy,
                            score_bands=score_bands,
                            include_symbols=include_symbols,
                            exclude_symbols=exclude_symbols,
                        )
                        backtester = AthenaBacktester(config)
                        metrics_df = run_threshold_suite(
                            backtester,
                            thresholds,
                            symbol=args.symbol,
                            max_symbols=args.max_symbols,
                        )

                        for _, row in metrics_df.iterrows():
                            result_row = row.to_dict()
                            result_row.update({
                                'model': model_mode,
                                'risk_level': normalized_risk or 'ALL',
                                'max_steps': max_steps,
                                'confirm_threshold': confirm_threshold,
                                'use_alt_market_proxy': use_alt_market_proxy,
                                'score_bands': args.trade_score_bands or 'ALL',
                                'train_size': args.train_size,
                            })
                            rows.append(result_row)

    return pd.DataFrame(rows)


def run_calibration(args):
    include_symbols = parse_symbols(args.include_symbols)
    exclude_symbols = parse_symbols(args.exclude_symbols)
    score_bands = parse_score_bands(args.calibration_trade_score_bands)
    bins = parse_csv(args.score_bins, float) or DEFAULT_SCORE_BINS
    risk_level = None if args.calibration_risk_level.upper() in ['NONE', 'ALL'] else args.calibration_risk_level.upper()

    config = BacktestConfig(
        data_folder=args.data_folder,
        initial_balance=args.initial_balance,
        position_size=args.position_size,
        ai_score_threshold=args.calibration_min_score,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        train_size=args.train_size,
        max_steps=args.calibration_max_steps,
        risk_filter=risk_level,
        model_mode=args.calibration_model,
        confirmation_threshold=args.calibration_confirm_threshold,
        use_alt_market_proxy=args.calibration_use_alt_market_proxy,
        score_bands=score_bands,
        include_symbols=include_symbols,
        exclude_symbols=exclude_symbols,
    )
    backtester = AthenaBacktester(config)
    if args.symbol:
        result = backtester.backtest_symbol(args.symbol)
    else:
        result = backtester.backtest_folder(max_symbols=args.max_symbols)

    return build_score_calibration(result['trades'], bins=bins)


def print_table(title, df):
    print(title)
    print('=' * len(title))
    if df.empty:
        print('No rows.')
        return

    print(df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description='ATHENA experiment runner and score calibration lab')
    parser.add_argument('--data-folder', default=os.path.join('data', 'raw', '4h'))
    parser.add_argument('--symbol')
    parser.add_argument('--max-symbols', type=int, default=100)
    parser.add_argument('--include-symbols')
    parser.add_argument('--exclude-symbols', default=DEFAULT_EXCLUDE_SYMBOLS)
    parser.add_argument('--train-size', type=int, default=700)
    parser.add_argument('--initial-balance', type=float, default=1000.0)
    parser.add_argument('--position-size', type=float, default=100.0)
    parser.add_argument('--fee-pct', type=float, default=0.001)
    parser.add_argument('--slippage-pct', type=float, default=0.0)
    parser.add_argument('--thresholds', default='68,70,72,75,78')
    parser.add_argument('--models', default='xgb,ensemble,xgb_rf_confirm')
    parser.add_argument('--risk-levels', default='LOW')
    parser.add_argument('--max-steps-values', default='300')
    parser.add_argument('--confirm-thresholds', default='45,50,55')
    parser.add_argument('--use-alt-market-proxy-values', default='false')
    parser.add_argument('--trade-score-bands', help='Filter trades by score ranges, e.g. 68:75,80:100')
    parser.add_argument('--skip-experiments', action='store_true')
    parser.add_argument('--skip-calibration', action='store_true')
    parser.add_argument('--calibration-model', default='xgb', choices=['xgb', 'rf', 'ensemble', 'xgb_rf_confirm'])
    parser.add_argument('--calibration-risk-level', default='LOW')
    parser.add_argument('--calibration-max-steps', type=int, default=300)
    parser.add_argument('--calibration-min-score', type=float, default=0.0)
    parser.add_argument('--calibration-confirm-threshold', type=float, default=50.0)
    parser.add_argument('--calibration-use-alt-market-proxy', action='store_true')
    parser.add_argument('--calibration-trade-score-bands')
    parser.add_argument('--score-bins', default='0,50,60,65,68,70,72,75,80,85,90,95,100')
    parser.add_argument('--output-dir')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True) if args.output_dir else None

    if not args.skip_experiments:
        experiments_df = run_experiments(args)
        if not experiments_df.empty:
            experiments_df['_profit_factor_sort'] = pd.to_numeric(
                experiments_df['profit_factor'].replace('inf', math.inf),
                errors='coerce',
            ).fillna(0.0)
            experiments_df = experiments_df.sort_values(
                ['_profit_factor_sort', 'trades', 'max_drawdown_pct'],
                ascending=[False, False, True],
            ).drop(columns=['_profit_factor_sort'])
        print_table('ATHENA Experiment Ranking', experiments_df)
        if args.output_dir:
            experiments_df.to_csv(os.path.join(args.output_dir, 'experiments.csv'), index=False)

    if not args.skip_calibration:
        calibration_df = run_calibration(args)
        print()
        print_table('ATHENA Score Calibration', calibration_df)
        if args.output_dir:
            calibration_df.to_csv(os.path.join(args.output_dir, 'score_calibration.csv'), index=False)


if __name__ == '__main__':
    main()
