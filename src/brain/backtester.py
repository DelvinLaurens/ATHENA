import argparse
import math
import os
from dataclasses import dataclass

import pandas as pd

from src.brain.intelligence import AthenaBrain


@dataclass
class BacktestConfig:
    data_folder: str = os.path.join('data', 'raw', '4h')
    initial_balance: float = 1000.0
    position_size: float = 100.0
    ai_score_threshold: float = 75.0
    fee_pct: float = 0.001
    slippage_pct: float = 0.0
    train_size: int = 700
    max_steps: int | None = None
    risk_filter: str | None = None


class AthenaBacktester:
    EXCLUDED_FILES = {
        'BTC_USDT.csv',
        'BTC_DOM_PROXY.csv',
        'BTCDOMUSDT.csv',
    }

    def __init__(self, config=None):
        self.config = config or BacktestConfig()
        self.brain = AthenaBrain()

    @staticmethod
    def symbol_to_filename(symbol):
        return f"{symbol.replace('/', '_')}.csv"

    @staticmethod
    def filename_to_symbol(filename):
        return filename.replace('.csv', '').replace('_', '/')

    def _csv_path(self, symbol):
        return os.path.join(self.config.data_folder, self.symbol_to_filename(symbol))

    def _load_optional_csv(self, filename):
        path = os.path.join(self.config.data_folder, filename)
        if not os.path.exists(path):
            return None

        return pd.read_csv(path)

    def _prepare_symbol_data(self, symbol):
        csv_path = self._csv_path(symbol)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)
        dom_df = self._load_optional_csv('BTC_DOM_PROXY.csv')
        btc_df = self._load_optional_csv('BTC_USDT.csv')
        processed_df = self.brain.prepare_data(df, dom_df, btc_df)
        processed_df = processed_df.dropna(subset=['future_return_4h'])
        return processed_df.reset_index(drop=True)

    def _score_row(self, train_df, latest_df):
        y_train = train_df['Target']
        if y_train.nunique() < 2:
            return round(float(y_train.mean()) * 100, 2)

        x_train = train_df[self.brain.FEATURES]
        x_latest = latest_df[self.brain.FEATURES]
        xgb_probability = self.brain._predict_positive_probability(
            self.brain.xgb_model,
            x_train,
            y_train,
            x_latest,
        )
        rf_probability = self.brain._predict_positive_probability(
            self.brain.rf_model,
            x_train,
            y_train,
            x_latest,
        )
        probability = (
            xgb_probability * self.brain.model_weights['xgb']
            + rf_probability * self.brain.model_weights['rf']
        )
        return round(probability * 100, 2)

    @staticmethod
    def _risk_level(volatility_pct):
        if volatility_pct < 3.5:
            return "LOW"
        if volatility_pct < 7.0:
            return "MEDIUM"
        return "HIGH"

    def _assess_row_risk(self, row):
        close = float(row['close'])
        atr = float(row['ATR_14'])
        if close == 0 or pd.isna(close) or pd.isna(atr):
            return "HIGH", 0.0

        volatility_pct = (atr / close) * 100
        return self._risk_level(volatility_pct), round(volatility_pct, 2)

    def _round_trip_cost_pct(self):
        return (self.config.fee_pct + self.config.slippage_pct) * 2

    def backtest_symbol(self, symbol):
        processed_df = self._prepare_symbol_data(symbol)
        if len(processed_df) <= self.config.train_size:
            return {
                'symbol': symbol,
                'trades': pd.DataFrame(),
                'metrics': self._empty_metrics(symbol),
            }

        balance = self.config.initial_balance
        trades = []
        test_stop = len(processed_df)
        if self.config.max_steps is not None:
            test_stop = min(test_stop, self.config.train_size + self.config.max_steps)

        for row_index in range(self.config.train_size, test_stop):
            train_df = processed_df.iloc[:row_index]
            latest_df = processed_df.iloc[[row_index]]
            latest_row = latest_df.iloc[0]
            ai_score = self._score_row(train_df, latest_df)
            risk_level, vol_pct = self._assess_row_risk(latest_row)

            if ai_score < self.config.ai_score_threshold:
                continue
            if self.config.risk_filter and risk_level != self.config.risk_filter:
                continue

            future_return = float(latest_row['future_return_4h'])
            net_return = future_return - self._round_trip_cost_pct()
            pnl = self.config.position_size * net_return
            balance += pnl
            return_pct = net_return * 100
            is_win = net_return > 0

            trades.append({
                'timestamp': latest_row['timestamp'],
                'symbol': symbol,
                'ai_score': ai_score,
                'risk_level': risk_level,
                'vol_pct': vol_pct,
                'entry_price': float(latest_row['close']),
                'gross_return_pct': round(future_return * 100, 4),
                'return_pct': round(return_pct, 4),
                'cost_pct': round(self._round_trip_cost_pct() * 100, 4),
                'pnl': round(pnl, 4),
                'balance': round(balance, 4),
                'is_win': is_win,
                'target_hit': future_return > self.brain.TARGET_PROFIT_THRESHOLD,
            })

        trades_df = pd.DataFrame(trades)
        return {
            'symbol': symbol,
            'trades': trades_df,
            'metrics': self.calculate_metrics(trades_df, symbol),
        }

    def backtest_folder(self, symbols=None, max_symbols=None):
        selected_symbols = symbols or self.discover_symbols()
        if max_symbols is not None:
            selected_symbols = selected_symbols[:max_symbols]

        all_trades = []
        per_symbol_metrics = []
        for symbol in selected_symbols:
            result = self.backtest_symbol(symbol)
            trades_df = result['trades']
            per_symbol_metrics.append(result['metrics'])
            if not trades_df.empty:
                all_trades.append(trades_df)

        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        combined_trades = self.recalculate_equity(combined_trades)

        return {
            'trades': combined_trades,
            'metrics': self.calculate_metrics(combined_trades, 'ALL'),
            'per_symbol': pd.DataFrame(per_symbol_metrics),
        }

    def discover_symbols(self):
        if not os.path.exists(self.config.data_folder):
            return []

        symbols = []
        for filename in sorted(os.listdir(self.config.data_folder)):
            if not filename.endswith('.csv') or filename in self.EXCLUDED_FILES:
                continue
            symbols.append(self.filename_to_symbol(filename))
        return symbols

    def recalculate_equity(self, trades_df):
        if trades_df.empty:
            return trades_df

        trades_df = trades_df.sort_values('timestamp').reset_index(drop=True).copy()
        trades_df['balance'] = (
            self.config.initial_balance + trades_df['pnl'].cumsum()
        ).round(4)
        return trades_df

    def build_equity_curve(self, trades_df):
        if trades_df.empty:
            return pd.DataFrame([{
                'timestamp': None,
                'balance': self.config.initial_balance,
                'drawdown_pct': 0.0,
            }])

        equity_df = trades_df[['timestamp', 'balance']].copy()
        running_peak = equity_df['balance'].cummax()
        equity_df['drawdown_pct'] = ((equity_df['balance'] - running_peak) / running_peak * 100).round(4)
        return equity_df

    def plot_equity_curve(self, trades_df, output_path):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise RuntimeError("matplotlib belum terinstall. Jalankan: pip install matplotlib") from e

        equity_df = self.build_equity_curve(trades_df)
        x_values = pd.to_datetime(equity_df['timestamp'], errors='coerce')
        if x_values.isna().all():
            x_values = range(len(equity_df))

        plt.figure(figsize=(11, 5))
        plt.plot(x_values, equity_df['balance'], linewidth=2)
        plt.title('ATHENA Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Balance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()

    def optimize_thresholds(self, thresholds, symbol=None, max_symbols=None):
        thresholds = sorted(float(threshold) for threshold in thresholds)
        if not thresholds:
            return pd.DataFrame()

        original_threshold = self.config.ai_score_threshold
        self.config.ai_score_threshold = thresholds[0]
        try:
            if symbol:
                base_result = self.backtest_symbol(symbol)
            else:
                base_result = self.backtest_folder(max_symbols=max_symbols)
        finally:
            self.config.ai_score_threshold = original_threshold

        base_trades = base_result['trades']
        rows = []
        for threshold in thresholds:
            if base_trades.empty:
                filtered_trades = base_trades
            else:
                filtered_trades = base_trades[base_trades['ai_score'] >= threshold].copy()
                filtered_trades = self.recalculate_equity(filtered_trades)

            metrics = self.calculate_metrics(filtered_trades, f"threshold_{threshold:g}")
            metrics['threshold'] = threshold
            rows.append(metrics)

        columns = ['threshold'] + [column for column in rows[0] if column != 'threshold']
        return pd.DataFrame(rows)[columns]

    def calculate_metrics(self, trades_df, symbol):
        if trades_df.empty:
            return self._empty_metrics(symbol)

        wins = trades_df[trades_df['is_win']]
        losses = trades_df[~trades_df['is_win']]
        gross_profit = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum()
        gross_loss = abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum())
        profit_factor = math.inf if gross_loss == 0 and gross_profit > 0 else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss

        balances = pd.concat([
            pd.Series([self.config.initial_balance]),
            trades_df['balance'].reset_index(drop=True),
        ], ignore_index=True)
        running_peak = balances.cummax()
        drawdown_pct = ((balances - running_peak) / running_peak) * 100

        return {
            'symbol': symbol,
            'trades': int(len(trades_df)),
            'wins': int(len(wins)),
            'losses': int(len(losses)),
            'win_rate_pct': round((len(wins) / len(trades_df)) * 100, 2),
            'gross_profit': round(float(gross_profit), 4),
            'gross_loss': round(float(gross_loss), 4),
            'profit_factor': round(float(profit_factor), 4) if math.isfinite(profit_factor) else 'inf',
            'max_drawdown_pct': round(abs(float(drawdown_pct.min())), 4),
            'final_balance': round(float(trades_df['balance'].iloc[-1]), 4),
            'net_profit': round(float(trades_df['balance'].iloc[-1] - self.config.initial_balance), 4),
            'total_return_pct': round(
                ((float(trades_df['balance'].iloc[-1]) - self.config.initial_balance) / self.config.initial_balance) * 100,
                4,
            ),
        }

    def _empty_metrics(self, symbol):
        return {
            'symbol': symbol,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate_pct': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown_pct': 0.0,
            'final_balance': self.config.initial_balance,
            'net_profit': 0.0,
            'total_return_pct': 0.0,
        }


def print_report(result):
    metrics = result['metrics']
    print("ATHENA v0.9 Backtest Report")
    print("===========================")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    per_symbol = result.get('per_symbol')
    if per_symbol is not None and not per_symbol.empty:
        print("\nWin Rate per Coin")
        print(
            per_symbol
            .sort_values(['win_rate_pct', 'trades'], ascending=[False, False])
            .to_string(index=False)
        )


def print_optimizer_report(optimizer_df):
    print("ATHENA v0.9 Threshold Optimizer")
    print("===============================")
    if optimizer_df.empty:
        print("Tidak ada hasil optimizer.")
        return

    print(optimizer_df.to_string(index=False))


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description='ATHENA v0.9 walk-forward backtester')
    parser.add_argument('--symbol', help='Run one symbol, e.g. SOL/USDT')
    parser.add_argument('--data-folder', default=os.path.join('data', 'raw', '4h'))
    parser.add_argument('--train-size', type=int, default=700)
    parser.add_argument('--threshold', type=float, default=75.0)
    parser.add_argument('--fee-pct', type=float, default=0.001)
    parser.add_argument('--slippage-pct', type=float, default=0.0)
    parser.add_argument('--initial-balance', type=float, default=1000.0)
    parser.add_argument('--position-size', type=float, default=100.0)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--max-symbols', type=int)
    parser.add_argument('--risk-level', choices=['LOW', 'MEDIUM', 'HIGH'])
    parser.add_argument('--optimize-thresholds', help='Comma-separated thresholds, e.g. 70,75,80,85,90')
    parser.add_argument('--plot-equity', action='store_true')
    parser.add_argument('--output-dir')
    return parser.parse_args()


def main():
    args = parse_args()
    config = BacktestConfig(
        data_folder=args.data_folder,
        initial_balance=args.initial_balance,
        position_size=args.position_size,
        ai_score_threshold=args.threshold,
        fee_pct=args.fee_pct,
        slippage_pct=args.slippage_pct,
        train_size=args.train_size,
        max_steps=args.max_steps,
        risk_filter=args.risk_level,
    )
    backtester = AthenaBacktester(config)

    if args.optimize_thresholds:
        optimizer_df = backtester.optimize_thresholds(
            parse_thresholds(args.optimize_thresholds),
            symbol=args.symbol,
            max_symbols=args.max_symbols,
        )
        print_optimizer_report(optimizer_df)
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            optimizer_df.to_csv(os.path.join(args.output_dir, 'threshold_optimizer.csv'), index=False)
        return

    if args.symbol:
        result = backtester.backtest_symbol(args.symbol)
    else:
        result = backtester.backtest_folder(max_symbols=args.max_symbols)

    print_report(result)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        result['trades'].to_csv(os.path.join(args.output_dir, 'trades.csv'), index=False)
        backtester.build_equity_curve(result['trades']).to_csv(
            os.path.join(args.output_dir, 'equity_curve.csv'),
            index=False,
        )
        if 'per_symbol' in result:
            result['per_symbol'].to_csv(os.path.join(args.output_dir, 'per_symbol.csv'), index=False)
        if args.plot_equity:
            try:
                backtester.plot_equity_curve(result['trades'], os.path.join(args.output_dir, 'equity_curve.png'))
            except RuntimeError as e:
                print(e)


if __name__ == '__main__':
    main()
