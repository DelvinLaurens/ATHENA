import pandas as pd
import os
from datetime import datetime, timedelta

class AthenaValidator:
    def __init__(self, log_path='data/predictions_log.csv'):
        self.log_path = log_path
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Buat file log jika belum ada
        if not os.path.exists(self.log_path):
            df = pd.DataFrame(columns=['timestamp', 'symbol', 'price_at_pred', 'ai_score', 'is_validated', 'result'])
            df.to_csv(self.log_path, index=False)

    def log_prediction(self, symbol, price, score):
        """Mencatat prediksi baru"""
        new_pred = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'price_at_pred': price,
            'ai_score': score,
            'is_validated': False,
            'result': None
        }
        df = pd.read_csv(self.log_path)
        df = pd.concat([df, pd.DataFrame([new_pred])], ignore_index=True)
        df.to_csv(self.log_path, index=False)

    def validate_yesterday(self, current_prices):
        """
        Mengecek prediksi lama apakah 'Hit' atau 'Miss'
        current_prices: dict { 'BTC/USDT': 62000, ... }
        """
        df = pd.read_csv(self.log_path)
        df['is_validated'] = df['is_validated'].astype(str).str.lower().isin(['true', '1'])
        
        # Cari yang belum divalidasi dan sudah lebih dari 20 jam
        mask = (~df['is_validated']) & (pd.to_datetime(df['timestamp']) < datetime.now() - timedelta(hours=20))
        
        hits = 0
        total_validated = 0
        
        for idx, row in df[mask].iterrows():
            symbol = row['symbol']
            if symbol in current_prices:
                price_now = current_prices[symbol]
                price_then = row['price_at_pred']
                score = row['ai_score']
                
                # Logika Validasi:
                # Jika AI bilang > 50% (Naik) dan harga sekarang > harga dulu = HIT
                # Jika AI bilang < 50% (Turun) dan harga sekarang < harga dulu = HIT
                is_hit = False
                if score > 50 and price_now > price_then:
                    is_hit = True
                elif score <= 50 and price_now < price_then:
                    is_hit = True
                
                df.at[idx, 'is_validated'] = True
                df.at[idx, 'result'] = 1 if is_hit else 0
                total_validated += 1
                if is_hit: hits += 1
        
        df.to_csv(self.log_path, index=False)
        return hits, total_validated

    def get_win_rate(self, days=7):
        """Menghitung win rate dalam N hari terakhir"""
        df = pd.read_csv(self.log_path)
        df = df[df['result'].notnull()] # Hanya ambil yang sudah divalidasi
        
        if df.empty:
            return 0.0
            
        # Ambil data X hari terakhir
        recent_df = df[pd.to_datetime(df['timestamp']) > datetime.now() - timedelta(days=days)]
        
        if recent_df.empty:
            return 0.0
            
        win_rate = (recent_df['result'].sum() / len(recent_df)) * 100
        return round(win_rate, 2)
