import os

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()


class BinanceProvider:
    MIN_QUOTE_VOLUME = 5_000_000
    PUBLIC_BASE_URLS = [
        'https://api.binance.com/api/v3',
        'https://api1.binance.com/api/v3',
        'https://api2.binance.com/api/v3',
        'https://api3.binance.com/api/v3',
        'https://api4.binance.com/api/v3',
        'https://data-api.binance.vision/api/v3',
    ]
    BLACKLIST_PATTERNS = [
        'USD',
        'EUR',
        'GBP',
        'DAI',
        'USDC',
        'USDT',
        'BUSD',
        'FDUSD',
        'TUSD',
        'PAXG',
        'AEUR',
    ]
    BLACKLIST_SYMBOLS = {
        'EURUSDT',
        'GBPUSDT',
        'USDCUSDT',
        'FDUSDUSDT',
        'TUSDUSDT',
        'PAXGUSDT',
        'XAUTUSDT',
        'AEURUSDT',
    }
    LEVERAGED_TOKENS = ['UPUSDT', 'DOWNUSDT', 'BEARUSDT', 'BULLUSDT']

    def __init__(self):
        self.endpoint_index = 0
        self.session = requests.Session()

    def _request(self, path, params=None):
        endpoint_order = list(range(self.endpoint_index, len(self.PUBLIC_BASE_URLS)))
        endpoint_order += list(range(0, self.endpoint_index))
        last_error = None

        for index in endpoint_order:
            base_url = self.PUBLIC_BASE_URLS[index]
            url = f"{base_url}{path}"
            try:
                response = self.session.get(url, params=params, timeout=20)
                response.raise_for_status()
                self.endpoint_index = index
                return response.json()
            except requests.RequestException as e:
                last_error = e
                print(f"Endpoint gagal: {url} -> {e}")

        raise RuntimeError(f"Semua endpoint Binance public gagal. Error terakhir: {last_error}")

    @staticmethod
    def _to_binance_symbol(symbol):
        return symbol.replace('/', '')

    @staticmethod
    def _to_slash_symbol(symbol):
        if symbol.endswith('USDT'):
            return f"{symbol[:-4]}/USDT"
        return symbol

    @classmethod
    def _base_asset(cls, symbol):
        symbol = symbol.upper()
        if symbol.endswith('USDT'):
            return symbol[:-4]
        return symbol.split('/')[0]

    @classmethod
    def _is_blacklisted_market(cls, symbol):
        normalized_symbol = symbol.upper()
        base_asset = cls._base_asset(normalized_symbol)
        return (
            normalized_symbol in cls.BLACKLIST_SYMBOLS
            or any(pattern in base_asset for pattern in cls.BLACKLIST_PATTERNS)
        )

    def get_top_volume_coins(self, limit=50):
        print(
            f"Mencari {limit} koin paling ramai di market "
            f"dengan volume minimal ${self.MIN_QUOTE_VOLUME:,.0f}..."
        )
        try:
            tickers = self._request('/ticker/24hr')
        except Exception as e:
            print(f"Gagal mengambil tickers: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

        usdt_pairs = []
        for item in tickers:
            symbol = item.get('symbol', '')
            quote_volume = float(item.get('quoteVolume') or 0)
            is_leveraged = any(token in symbol for token in self.LEVERAGED_TOKENS)

            if not symbol.endswith('USDT') or is_leveraged:
                continue
            if (
                self._is_blacklisted_market(symbol)
                or not symbol.isascii()
                or quote_volume < self.MIN_QUOTE_VOLUME
            ):
                continue

            usdt_pairs.append({
                'symbol': self._to_slash_symbol(symbol),
                'quoteVolume': quote_volume,
            })

        sorted_pairs = sorted(usdt_pairs, key=lambda x: x['quoteVolume'], reverse=True)
        return [pair['symbol'] for pair in sorted_pairs[:limit]]

    def fetch_ohlcv(self, symbol, timeframe='1d', limit=1000):
        try:
            klines = self._request('/klines', params={
                'symbol': self._to_binance_symbol(symbol),
                'interval': timeframe,
                'limit': limit,
            })
        except Exception as e:
            print(f"Error saat mengambil data {symbol}: {e}")
            return None

        if not klines:
            return None

        rows = []
        for candle in klines:
            rows.append({
                'timestamp': pd.to_datetime(candle[0], unit='ms'),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),
            })

        return pd.DataFrame(rows)

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
