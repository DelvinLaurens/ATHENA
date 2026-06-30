import argparse
import os
import time

import pandas as pd

from main import BTC_SYMBOL, build_btc_dominance_proxy
from src.brain.market_proxy import build_alt_market_proxy
from src.oracle.binance_provider import BinanceProvider


def parse_symbols(value):
    if not value:
        return None

    return [item.strip().upper() for item in value.split(',') if item.strip()]


def row_count(csv_path):
    if not os.path.exists(csv_path):
        return 0

    return len(pd.read_csv(csv_path))


def sync_market_data(args):
    provider = BinanceProvider()
    data_folder = args.data_folder or os.path.join('data', 'raw', args.timeframe)

    symbols = parse_symbols(args.symbols)
    if symbols is None:
        symbols = provider.get_top_volume_coins(limit=args.max_symbols)

    if args.include_btc and BTC_SYMBOL not in symbols:
        symbols.append(BTC_SYMBOL)

    print(
        f"Syncing {len(symbols)} symbols | timeframe={args.timeframe} | "
        f"candles={args.limit} | folder={data_folder}"
    )

    synced_symbols = []
    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] {symbol}")
        df = provider.fetch_ohlcv(symbol, timeframe=args.timeframe, limit=args.limit)
        if df is None or df.empty:
            print(f"  skipped: no data")
            continue

        provider.save_to_csv(df, symbol, data_folder=data_folder)
        csv_path = os.path.join(data_folder, f"{symbol.replace('/', '_')}.csv")
        print(f"  saved rows: {row_count(csv_path)}")
        synced_symbols.append(symbol)
        time.sleep(args.sleep)

    if args.build_btc_dom_proxy:
        dom_path = build_btc_dominance_proxy(data_folder, synced_symbols)
        print(f"BTC dominance proxy: {dom_path or 'not available'}")

    if args.build_alt_proxy:
        alt_path = build_alt_market_proxy(data_folder, synced_symbols)
        print(f"Alt market proxy: {alt_path or 'not available'}")

    print(f"Done. Synced {len(synced_symbols)}/{len(symbols)} symbols.")


def parse_args():
    parser = argparse.ArgumentParser(description='Sync ATHENA historical market data')
    parser.add_argument('--timeframe', default='4h')
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--max-symbols', type=int, default=100)
    parser.add_argument('--symbols', help='Comma-separated symbols, e.g. ETH/USDT,BCH/USDT')
    parser.add_argument('--data-folder')
    parser.add_argument('--sleep', type=float, default=0.05)
    parser.add_argument('--include-btc', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--build-btc-dom-proxy', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--build-alt-proxy', action='store_true')
    return parser.parse_args()


def main():
    sync_market_data(parse_args())


if __name__ == '__main__':
    main()
