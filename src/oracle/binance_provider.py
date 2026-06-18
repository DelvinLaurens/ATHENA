import ccxt
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class BinanceProvider:
    def __init__(self):
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
        })

    def get_top_volume_coins(self, limit=50):
        print(f"🔍 Mencari {limit} koin paling ramai di market...")
        try:
            tickers = self.exchange.fetch_tickers()
            blacklist = ['EUR/USDT', 'GBP/USDT', 'USDC/USDT', 'FDUSD/USDT', 'TUSD/USDT', 'PAXG/USDT', 'XAUT/USDT', 'AEUR/USDT']
            
            usdt_pairs = []
            for symbol, data in tickers.items():
                quote_volume = data.get('quoteVolume') or 0
                if '/USDT' in symbol and all(x not in symbol for x in ['UP/', 'DOWN/', 'BEAR/', 'BULL/']):
                    if symbol not in blacklist and symbol.isascii() and quote_volume > 0:
                        usdt_pairs.append({
                            'symbol': symbol,
                            'quoteVolume': quote_volume
                        })
            
            sorted_pairs = sorted(usdt_pairs, key=lambda x: x['quoteVolume'], reverse=True)
            return [p['symbol'] for p in sorted_pairs[:limit]]
        except Exception as e:
            print(f"❌ Gagal mengambil tickers: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

    # --- UPDATE LIMIT DISINI (default limit kita jadikan 1000) ---
    def fetch_ohlcv(self, symbol, timeframe='1d', limit=1000):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error saat mengambil data {symbol}: {e}")
            return None

    # --- UPDATE FUNGSI SAVE AGAR INCREMENTAL ---
    def save_to_csv(self, df, symbol, data_folder='data/raw'):
        if df is not None and not df.empty:
            os.makedirs(data_folder, exist_ok=True)
            
            safe_symbol = symbol.replace('/', '_')
            filename = os.path.join(data_folder, f"{safe_symbol}.csv")
            
            # Jika file sudah ada, kita gabungkan (append)
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                # Pastikan format timestamp konsisten untuk perbandingan
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                # Gabungkan data lama dan baru
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # Hapus duplikat berdasarkan timestamp (jaga data terbaru)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                
                # Urutkan berdasarkan waktu agar tidak berantakan
                combined_df = combined_df.sort_values(by='timestamp')
                
                combined_df.to_csv(filename, index=False)
            else:
                # Jika belum ada file, langsung simpan
                df.to_csv(filename, index=False)

    def prepare_data(self, df, btc_dom_df):
        # 1. Feature Teknikal Koin Individu
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['VOL_Change'] = df['volume'].pct_change()
        
        # 2. Feature Makro (PENTING!)
        # Kita masukkan data BTC Dominance ke dalam koin individu
        # Kita hitung perubahan harian BTC.D
        btc_dom_df['dom_change'] = btc_dom_df['close'].pct_change()
        
        # Gabungkan data berdasarkan timestamp
        df = df.merge(btc_dom_df[['timestamp', 'dom_change']], on='timestamp', how='left')
        
        # 3. Labeling
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        df = df.dropna()
        return df

    def train_and_predict(self, csv_path, btc_dom_path):
        df = pd.read_csv(csv_path)
        btc_dom_df = pd.read_csv(btc_dom_path)
        
        df = self.prepare_data(df, btc_dom_df)
        
        if len(df) < 50:
            return 50.0
            
        # AI sekarang belajar dari RSI koin DAN Dominasi BTC sekaligus!
        features = ['RSI', 'SMA_20', 'VOL_Change', 'dom_change']
        X = df[features]
        y = df['Target']
        
        self.model.fit(X.iloc[:-1], y.iloc[:-1])
        probability = self.model.predict_proba(X.tail(1))[0][1]
        
        return round(probability * 100, 2)
