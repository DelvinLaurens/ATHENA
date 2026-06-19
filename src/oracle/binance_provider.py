import os

import ccxt
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


class BinanceProvider:
    PUBLIC_BASE_URLS = [
        'https://api.binance.com/api/v3',
        'https://api1.binance.com/api/v3',
        'https://api2.binance.com/api/v3',
        'https://api3.binance.com/api/v3',
        'https://api4.binance.com/api/v3',
        'https://data-api.binance.vision/api/v3',
    ]

    def __init__(self):
        # ATHENA hanya butuh market data public. Jangan pasang API key di sini,
        # karena ccxt bisa mencoba endpoint private Binance saat load market.
        self.endpoint_index = 0
        self.exchange = self._create_exchange(self.PUBLIC_BASE_URLS[self.endpoint_index])

    def _create_exchange(self, public_base_url):
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'fetchCurrencies': False,
                'adjustForTimeDifference': True,
            },
        })
        exchange.urls['api']['public'] = public_base_url
        exchange.urls['api']['v1'] = public_base_url.replace('/api/v3', '/api/v1')
        return exchange

    def _use_endpoint(self, index):
        self.endpoint_index = index
        self.exchange = self._create_exchange(self.PUBLIC_BASE_URLS[index])
        print(f"Binance public endpoint: {self.PUBLIC_BASE_URLS[index]}")

    def get_top_volume_coins(self, limit=50):
        print(f"Mencari {limit} koin paling ramai di market...")
        for index, _ in enumerate(self.PUBLIC_BASE_URLS):
            self._use_endpoint(index)
            try:
                tickers = self.exchange.fetch_tickers()
                break
            except Exception as e:
                print(f"Gagal mengambil tickers dari endpoint ini: {e}")
        else:
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

        try:
            blacklist = {
                'EUR/USDT',
                'GBP/USDT',
                'USDC/USDT',
                'FDUSD/USDT',
                'TUSD/USDT',
                'PAXG/USDT',
                'XAUT/USDT',
                'AEUR/USDT',
            }

            usdt_pairs = []
            for symbol, data in tickers.items():
                quote_volume = data.get('quoteVolume') or 0
                is_leveraged = any(x in symbol for x in ['UP/', 'DOWN/', 'BEAR/', 'BULL/'])

                if '/USDT' not in symbol or is_leveraged:
                    continue
                if symbol in blacklist or not symbol.isascii() or quote_volume <= 0:
                    continue

                usdt_pairs.append({
                    'symbol': symbol,
                    'quoteVolume': quote_volume,
                })

            sorted_pairs = sorted(usdt_pairs, key=lambda x: x['quoteVolume'], reverse=True)
            return [pair['symbol'] for pair in sorted_pairs[:limit]]
        except Exception as e:
            print(f"Gagal memproses tickers: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

    def fetch_ohlcv(self, symbol, timeframe='1d', limit=1000):
        endpoint_order = list(range(self.endpoint_index, len(self.PUBLIC_BASE_URLS)))
        endpoint_order += list(range(0, self.endpoint_index))

        for index in endpoint_order:
            self._use_endpoint(index)
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if not ohlcv:
                    return None

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception as e:
                print(f"Error saat mengambil data {symbol} dari endpoint ini: {e}")

        return None

    def save_to_csv(self, df, symbol, data_folder='data/raw'):
        if df is None or df.empty:
            return

        os.makedirs(data_folder, exist_ok=True)

        safe_symbol = symbol.replace('/', '_')
        filename = os.path.join(data_folder, f"{safe_symbol}.csv")

        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])

            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            combined_df = combined_df.sort_values(by='timestamp')
            combined_df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False)
