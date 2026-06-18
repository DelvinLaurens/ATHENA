import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
import numpy as np
import os

class AthenaBrain:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)

    def prepare_data(self, df, dom_df=None):
        # 1. Feature Teknikal Koin
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['VOL_Change'] = df['volume'].pct_change()
        
        # 2. Feature Market Context (BTC Dominance)
        if dom_df is not None:
            dom_df['dom_change'] = dom_df['close'].pct_change()
            df = pd.merge(df, dom_df[['timestamp', 'dom_change']], on='timestamp', how='left')
        else:
            df['dom_change'] = 0.0
        
        # Bersihkan data
        cols_to_fix = ['RSI', 'SMA_20', 'EMA_9', 'VOL_Change', 'dom_change']
        for col in cols_to_fix:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Target: Apakah harga besok naik?
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df.dropna()

    def train_and_predict(self, csv_path, dom_path=None):
        try:
            df = pd.read_csv(csv_path)
            dom_df = pd.read_csv(dom_path) if dom_path and os.path.exists(dom_path) else None
            
            # Pastikan timestamp dalam format datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if dom_df is not None:
                dom_df['timestamp'] = pd.to_datetime(dom_df['timestamp'])
            
            processed_df = self.prepare_data(df, dom_df)
            
            if len(processed_df) < 50:
                return 50.0
            
            # Feature yang dipelajari AI (Sekarang termasuk dom_change!)
            features = ['RSI', 'SMA_20', 'EMA_9', 'VOL_Change', 'dom_change']
            X = processed_df[features]
            y = processed_df['Target']
            
            # Split data
            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            X_latest = X.tail(1)
            
            self.model.fit(X_train, y_train)
            probability = self.model.predict_proba(X_latest)[0][1]
            
            return round(probability * 100, 2)
        except Exception as e:
            print(f"🧠 Brain Error: {e}")
            return 50.0
