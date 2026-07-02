import os
from datetime import datetime, timedelta

import pandas as pd


class AthenaValidator:
    COLUMNS = [
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

    def __init__(self, log_path='data/predictions_log.csv'):
        self.log_path = log_path
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=self.COLUMNS).to_csv(self.log_path, index=False)
        else:
            self._load_log().to_csv(self.log_path, index=False)

    def _load_log(self):
        df = pd.read_csv(self.log_path)
        for column in self.COLUMNS:
            if column not in df.columns:
                df[column] = None

        if 'entry_price' in df.columns:
            df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
        df['price_at_pred'] = pd.to_numeric(df['price_at_pred'], errors='coerce')
        df['entry_price'] = df['entry_price'].fillna(df['price_at_pred'])
        df['signal'] = df['signal'].fillna(df['ai_score'].apply(lambda score: 'LONG' if float(score) > 50 else 'SHORT'))
        df['is_validated'] = df['is_validated'].astype(str).str.lower().isin(['true', '1'])

        return df[self.COLUMNS]

    def log_prediction(
        self,
        symbol,
        price,
        score,
        risk_level=None,
        vol_pct=None,
        change_4h=None,
        is_top_opportunity=False,
        is_scalper_hotlist=False,
    ):
        signal = 'LONG' if score > 50 else 'SHORT'
        new_pred = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'signal': signal,
            'price_at_pred': price,
            'entry_price': price,
            'exit_price': None,
            'ai_score': score,
            'risk_level': risk_level,
            'vol_pct': vol_pct,
            'change_4h': change_4h,
            'is_top_opportunity': is_top_opportunity,
            'is_scalper_hotlist': is_scalper_hotlist,
            'is_validated': False,
            'return_pct': None,
            'result': None,
        }
        df = self._load_log()
        df = pd.concat([df, pd.DataFrame([new_pred])], ignore_index=True)
        df.to_csv(self.log_path, index=False)

    def validate_yesterday(self, current_prices, min_age_hours=20):
        df = self._load_log()
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        cutoff = datetime.now() - timedelta(hours=min_age_hours)
        mask = (~df['is_validated']) & (timestamps < cutoff)

        hits = 0
        total_validated = 0

        for idx, row in df[mask].iterrows():
            symbol = row['symbol']
            if symbol not in current_prices:
                continue

            entry_price = float(row['entry_price'])
            exit_price = float(current_prices[symbol])
            signal = row['signal'] if row['signal'] in ['LONG', 'SHORT'] else ('LONG' if float(row['ai_score']) > 50 else 'SHORT')

            if entry_price == 0:
                continue

            if signal == 'LONG':
                return_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                return_pct = ((entry_price - exit_price) / entry_price) * 100

            is_hit = return_pct > 0

            df.at[idx, 'signal'] = signal
            df.at[idx, 'exit_price'] = exit_price
            df.at[idx, 'return_pct'] = round(return_pct, 4)
            df.at[idx, 'is_validated'] = True
            df.at[idx, 'result'] = 1 if is_hit else 0
            total_validated += 1
            if is_hit:
                hits += 1

        df.to_csv(self.log_path, index=False)
        return hits, total_validated

    def get_win_rate(self, days=7):
        df = self._load_log()
        df = df[df['result'].notnull()]

        if df.empty:
            return 0.0

        recent_df = df[pd.to_datetime(df['timestamp'], errors='coerce') > datetime.now() - timedelta(days=days)]
        if recent_df.empty:
            return 0.0

        win_rate = (pd.to_numeric(recent_df['result'], errors='coerce').sum() / len(recent_df)) * 100
        return round(win_rate, 2)

    def get_performance_summary(self, days=7):
        df = self._load_log()
        df = df[df['result'].notnull()]
        if df.empty:
            return {
                'trades': 0,
                'avg_return_pct': 0.0,
                'long_trades': 0,
                'long_win_rate': 0.0,
                'short_trades': 0,
                'short_win_rate': 0.0,
                'top_trades': 0,
                'top_win_rate': 0.0,
                'scalper_trades': 0,
                'scalper_win_rate': 0.0,
            }

        recent_df = df[pd.to_datetime(df['timestamp'], errors='coerce') > datetime.now() - timedelta(days=days)].copy()
        if recent_df.empty:
            return {
                'trades': 0,
                'avg_return_pct': 0.0,
                'long_trades': 0,
                'long_win_rate': 0.0,
                'short_trades': 0,
                'short_win_rate': 0.0,
                'top_trades': 0,
                'top_win_rate': 0.0,
                'scalper_trades': 0,
                'scalper_win_rate': 0.0,
            }

        recent_df['return_pct'] = pd.to_numeric(recent_df['return_pct'], errors='coerce').fillna(0)
        recent_df['result'] = pd.to_numeric(recent_df['result'], errors='coerce').fillna(0)

        def group_win_rate(group):
            if group.empty:
                return 0.0
            return round((group['result'].sum() / len(group)) * 100, 2)

        long_df = recent_df[recent_df['signal'] == 'LONG']
        short_df = recent_df[recent_df['signal'] == 'SHORT']
        top_df = recent_df[recent_df['is_top_opportunity'].astype(str).str.lower().isin(['true', '1'])]
        scalper_df = recent_df[recent_df['is_scalper_hotlist'].astype(str).str.lower().isin(['true', '1'])]

        return {
            'trades': len(recent_df),
            'avg_return_pct': round(recent_df['return_pct'].mean(), 4),
            'long_trades': len(long_df),
            'long_win_rate': group_win_rate(long_df),
            'short_trades': len(short_df),
            'short_win_rate': group_win_rate(short_df),
            'top_trades': len(top_df),
            'top_win_rate': group_win_rate(top_df),
            'scalper_trades': len(scalper_df),
            'scalper_win_rate': group_win_rate(scalper_df),
        }
