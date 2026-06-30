import os

import pandas as pd


ALT_MARKET_PROXY_FILE = 'ALT_MARKET_PROXY.csv'
ALT_MARKET_EXCLUDED_SYMBOLS = {
    'BTC/USDT',
    'ETH/USDT',
    'BTCDOMUSDT',
}


def symbol_to_filename(symbol):
    return f"{symbol.replace('/', '_')}.csv"


def _load_symbol_frame(data_folder, symbol):
    csv_path = os.path.join(data_folder, symbol_to_filename(symbol))
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    required_columns = {'timestamp', 'close', 'volume'}
    if df.empty or not required_columns.issubset(df.columns):
        return None

    df = df[['timestamp', 'close', 'volume']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'close', 'volume']).sort_values('timestamp')
    df = df[df['close'] > 0]
    if len(df) < 2:
        return None

    base_close = df['close'].iloc[0]
    if base_close <= 0:
        return None

    key = symbol.replace('/', '_')
    return pd.DataFrame({
        'timestamp': df['timestamp'],
        f'{key}_index': (df['close'] / base_close) * 100,
        f'{key}_return': df['close'].pct_change(),
        f'{key}_quote_volume': df['close'] * df['volume'],
    })


def build_alt_market_proxy(data_folder, symbols, output_filename=ALT_MARKET_PROXY_FILE):
    frames = []
    for symbol in symbols:
        normalized_symbol = symbol.replace('_', '/').upper()
        if normalized_symbol in ALT_MARKET_EXCLUDED_SYMBOLS:
            continue

        frame = _load_symbol_frame(data_folder, normalized_symbol)
        if frame is not None:
            frames.append(frame)

    if len(frames) < 2:
        return None

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on='timestamp', how='outer')

    merged = merged.sort_values('timestamp')
    index_columns = [column for column in merged.columns if column.endswith('_index')]
    return_columns = [column for column in merged.columns if column.endswith('_return')]
    volume_columns = [column for column in merged.columns if column.endswith('_quote_volume')]
    if not index_columns or not return_columns or not volume_columns:
        return None

    positive_returns = (merged[return_columns] > 0).sum(axis=1)
    valid_returns = merged[return_columns].notna().sum(axis=1)
    total_quote_volume = merged[volume_columns].fillna(0.0).sum(axis=1)

    breadth_denominator = valid_returns.where(valid_returns != 0)

    proxy_df = pd.DataFrame({
        'timestamp': merged['timestamp'],
        'close': merged[index_columns].mean(axis=1, skipna=True),
        'alt_market_breadth': positive_returns / breadth_denominator,
        'alt_volume_change': total_quote_volume.pct_change(),
    })
    proxy_df['alt_market_change'] = proxy_df['close'].pct_change()
    proxy_df = proxy_df.replace([float('inf'), float('-inf')], pd.NA)
    proxy_df['alt_market_breadth'] = proxy_df['alt_market_breadth'].fillna(0.5)
    proxy_df['alt_market_change'] = proxy_df['alt_market_change'].fillna(0.0)
    proxy_df['alt_volume_change'] = proxy_df['alt_volume_change'].fillna(0.0)
    proxy_df = proxy_df.dropna(subset=['timestamp', 'close'])

    if len(proxy_df) < 2:
        return None

    output_path = os.path.join(data_folder, output_filename)
    proxy_df.to_csv(output_path, index=False)
    return output_path
