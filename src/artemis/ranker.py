import pandas as pd
import pandas_ta as ta
import os

class Artemis:
    def __init__(self, data_folder='data/raw'):
        self.data_folder = data_folder

    def calculate_score(self, symbol):
        # Baca file CSV
        path = os.path.join(self.data_folder, f"{symbol.replace('/', '_')}.csv")
        df = pd.read_csv(path)
        
        # Hitung RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Hitung Perubahan Harga 24 Jam (%)
        df['change_24h'] = df['close'].pct_change() * 100
        
        # Ambil data terakhir
        last_rsi = df['rsi'].iloc[-1]
        last_change = df['change_24h'].iloc[-1]
        last_price = df['close'].iloc[-1]
        
        return {
            'symbol': symbol,
            'price': last_price,
            'rsi': last_rsi,
            'change_24h': last_change
        }

    def rank_coins(self, watch_list):
        results = []
        for symbol in watch_list:
            try:
                score = self.calculate_score(symbol)
                results.append(score)
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
        
        # Ubah ke DataFrame untuk memudahkan sorting
        rank_df = pd.DataFrame(results)
        
        # Ranking berdasarkan kenaikan harga tertinggi (Relative Strength)
        rank_df = rank_df.sort_values(by='change_24h', ascending=False)
        return rank_df

if __name__ == "__main__":
    watch_list = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT']
    
    artemis = Artemis()
    ranking = artemis.rank_coins(watch_list)
    
    print("\n🏛 ATHENA - ALTCOIN RELATIVE STRENGTH RANKING")
    print("==============================================")
    print(ranking.to_string(index=False))