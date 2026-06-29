import os

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class AthenaBrain:
    FEATURES = [
        'RSI',
        'SMA_20',
        'EMA_9',
        'VOL_Change',
        'dom_change',
        'MACD',
        'MACD_signal',
        'MACD_hist',
        'BB_Position',
        'ATR_Ratio',
        'BTC_Correlation_20h',
    ]
    TARGET_PROFIT_THRESHOLD = 0.005
    BTC_CORRELATION_PERIODS = 5

    def __init__(self):
        self.xgb_model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=3,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=1,
        )
        self.model_weights = {
            'xgb': 0.6,
            'rf': 0.4,
        }
        self.model = self.xgb_model

    @staticmethod
    def _first_indicator_column(indicator_df, prefix):
        if indicator_df is None or indicator_df.empty:
            return None

        for column in indicator_df.columns:
            if column.startswith(prefix):
                return indicator_df[column]
        return None

    @staticmethod
    def _coerce_market_data(df):
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

        for column in ['open', 'high', 'low', 'close', 'volume']:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')

        return df

    def _add_btc_correlation(self, df, btc_df):
        if btc_df is None or btc_df.empty:
            df['BTC_Correlation_20h'] = 0.0
            return df

        btc_df = self._coerce_market_data(btc_df)
        btc_context = btc_df[['timestamp', 'close']].rename(columns={'close': 'BTC_Close'})
        df = pd.merge(df, btc_context, on='timestamp', how='left')
        df['coin_return'] = df['close'].pct_change()
        df['btc_return'] = df['BTC_Close'].pct_change()
        df['BTC_Correlation_20h'] = (
            df['coin_return']
            .rolling(self.BTC_CORRELATION_PERIODS)
            .corr(df['btc_return'])
        )
        return df.drop(columns=['BTC_Close', 'coin_return', 'btc_return'])

    def prepare_data(self, df, dom_df=None, btc_df=None):
        df = self._coerce_market_data(df)

        df['RSI'] = ta.rsi(df['close'], length=14)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['VOL_Change'] = df['volume'].pct_change()

        macd = ta.macd(df['close'])
        df['MACD'] = self._first_indicator_column(macd, 'MACD_')
        df['MACD_signal'] = self._first_indicator_column(macd, 'MACDs_')
        df['MACD_hist'] = self._first_indicator_column(macd, 'MACDh_')

        bbands = ta.bbands(df['close'], length=20, std=2)
        bb_lower = self._first_indicator_column(bbands, 'BBL_')
        bb_upper = self._first_indicator_column(bbands, 'BBU_')
        if bb_lower is None or bb_upper is None:
            df['BB_Position'] = np.nan
        else:
            bb_range = bb_upper - bb_lower
            df['BB_Position'] = np.where(bb_range != 0, (df['close'] - bb_lower) / bb_range, 0.5)

        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        atr_mean_50 = df['ATR_14'].rolling(50).mean()
        df['ATR_Ratio'] = np.where(atr_mean_50 != 0, df['ATR_14'] / atr_mean_50, 1.0)

        if dom_df is not None:
            dom_df = dom_df.copy()
            dom_df['timestamp'] = pd.to_datetime(dom_df['timestamp'])
            dom_df['close'] = pd.to_numeric(dom_df['close'], errors='coerce')
            dom_df['dom_change'] = dom_df['close'].pct_change()
            df = pd.merge(df, dom_df[['timestamp', 'dom_change']], on='timestamp', how='left')
        else:
            df['dom_change'] = 0.0

        df = self._add_btc_correlation(df, btc_df)

        for column in self.FEATURES:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        df['future_return_4h'] = (df['close'].shift(-1) - df['close']) / df['close']
        df['Target'] = (df['future_return_4h'] > self.TARGET_PROFIT_THRESHOLD).astype(int)
        df = df.replace([np.inf, -np.inf], np.nan)
        df['dom_change'] = df['dom_change'].fillna(0.0)
        df['BTC_Correlation_20h'] = df['BTC_Correlation_20h'].fillna(0.0)

        return df.dropna(subset=self.FEATURES + ['Target'])

    def _predict_positive_probability(self, model, X_train, y_train, X_latest):
        model.fit(X_train, y_train)
        classes = list(model.classes_)
        if 1 not in classes:
            return 0.0

        class_index = classes.index(1)
        return float(model.predict_proba(X_latest)[0][class_index])

    def train_and_predict(self, csv_path, dom_path=None, btc_path=None):
        try:
            df = pd.read_csv(csv_path)
            dom_df = pd.read_csv(dom_path) if dom_path and os.path.exists(dom_path) else None
            if btc_path is None:
                btc_path = os.path.join(os.path.dirname(csv_path), 'BTC_USDT.csv')
            btc_df = pd.read_csv(btc_path) if btc_path and os.path.exists(btc_path) else None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if dom_df is not None:
                dom_df['timestamp'] = pd.to_datetime(dom_df['timestamp'])
            if btc_df is not None:
                btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])

            processed_df = self.prepare_data(df, dom_df, btc_df)

            if len(processed_df) < 80:
                return 50.0

            X = processed_df[self.FEATURES]
            y = processed_df['Target']

            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            X_latest = X.tail(1)

            if y_train.nunique() < 2:
                return round(float(y_train.mean()) * 100, 2)

            xgb_probability = self._predict_positive_probability(
                self.xgb_model,
                X_train,
                y_train,
                X_latest,
            )
            rf_probability = self._predict_positive_probability(
                self.rf_model,
                X_train,
                y_train,
                X_latest,
            )
            probability = (
                xgb_probability * self.model_weights['xgb']
                + rf_probability * self.model_weights['rf']
            )

            return round(probability * 100, 2)
        except Exception as e:
            print(f"Brain Error: {e}")
            return 50.0
