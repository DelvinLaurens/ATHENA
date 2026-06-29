import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import requests

from src.aegis.risk_manager import Aegis
from src.apollo.regime_engine import Apollo
from src.artemis.ranker import Artemis
from src.brain.intelligence import AthenaBrain
from src.brain.validator import AthenaValidator
from src.hermes.notifier import Hermes
from src.oracle.binance_provider import BinanceProvider


BTC_SYMBOL = 'BTC/USDT'
DOMINANCE_SYMBOL = 'BTCDOMUSDT'
DOMINANCE_PROXY_FILE = 'BTC_DOM_PROXY.csv'
SCAN_TIMEFRAME = '4h'
VALIDATION_HOURS = 4
DATA_FOLDER = os.path.join('data', 'raw', SCAN_TIMEFRAME)
LOCAL_REPORT_PATH = os.path.join('data', 'latest_report.md')
WATCH_LIST_LIMIT = 100
DEFAULT_ANALYSIS_WORKERS = 6
MAX_ANALYSIS_WORKERS = 8
AI_SCORE_THRESHOLD = 68.0
SIGNAL_RISK_LEVELS = {"LOW"}
TOP_OPPORTUNITY_LIMIT = 10
SCALPER_LIMIT = 5
HIGH_PROBABILITY_MESSAGE = "No high-probability low-risk opportunities detected. Market is too uncertain."
RISK_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}


def direction_label(change_pct):
    return "UP" if change_pct >= 0 else "DOWN"


def ai_label(ai_score):
    if ai_score >= 70:
        return "STRONG"
    if ai_score >= 55:
        return "BULL"
    if ai_score >= 45:
        return "NEUTRAL"
    return "BEAR"


def risk_label(risk_level):
    return {
        "LOW": "SAFE",
        "MEDIUM": "WATCH",
        "HIGH": "DANGER",
    }.get(risk_level, "UNKNOWN")


def risk_rank(risk_level):
    return RISK_ORDER.get(risk_level, 99)


def is_signal_candidate(item):
    return (
        float(item['ai_score']) >= AI_SCORE_THRESHOLD
        and item.get('risk_level') in SIGNAL_RISK_LEVELS
    )


def select_top_pick(items):
    if not items:
        return None

    tradable_items = [item for item in items if item.get('risk_level') != "HIGH"]
    candidates = tradable_items or items
    return sorted(
        candidates,
        key=lambda item: (-float(item['ai_score']), risk_rank(item.get('risk_level'))),
    )[0]


def format_top_pick(item):
    if item is None:
        return HIGH_PROBABILITY_MESSAGE

    return (
        f"**{item['symbol']}**\n"
        f"AI Score: {item['ai_score']:.1f}%\n"
        f"Risk: {risk_label(item['risk_level'])}\n"
        f"Price: {item['price']:.6g}\n"
        f"4H Change: {item['change_24h']:+.2f}%\n"
        f"Volatility: {item['vol_pct']:.2f}%"
    )


def format_opportunity_rows(items):
    if not items:
        return HIGH_PROBABILITY_MESSAGE

    lines = []
    for index, item in enumerate(items, start=1):
        pct_str = f"{item['change_24h']:+.2f}%"
        lines.append(
            f"{index:>2}. {item['symbol']:<12}"
            f"{pct_str:>8}   "
            f"AI {item['ai_score']:>4.1f}%   "
            f"{risk_label(item['risk_level'])}"
        )
    return "```\n" + "\n".join(lines) + "\n```"


def format_scalper_rows(items):
    if not items:
        return HIGH_PROBABILITY_MESSAGE

    lines = []
    for index, item in enumerate(items, start=1):
        bias = "LONG" if item['ai_score'] >= 55 else "AVOID" if item['ai_score'] <= 45 else "NEUTRAL"
        lines.append(
            f"{index:>2}. {item['symbol']:<12}"
            f"Vol {item['vol_pct']:>5.2f}%   "
            f"{bias}"
        )
    return "```\n" + "\n".join(lines) + "\n```"


def payload_to_markdown(payload):
    sections = [payload["content"]]
    for embed in payload["embeds"]:
        sections.append(f"## {embed['title']}")
        for field in embed.get("fields", []):
            sections.append(f"**{field['name']}**\n{field['value']}")
        if embed.get("footer"):
            sections.append(embed["footer"]["text"])
    return "\n\n".join(sections) + "\n"


def save_local_report(payload):
    os.makedirs(os.path.dirname(LOCAL_REPORT_PATH), exist_ok=True)
    with open(LOCAL_REPORT_PATH, 'w', encoding='utf-8') as report_file:
        report_file.write(payload_to_markdown(payload))
    print(f"Report lokal disimpan ke {LOCAL_REPORT_PATH}")


def should_fail_on_notification_error():
    return os.getenv('GITHUB_ACTIONS') == 'true' or os.getenv('ATHENA_FAIL_ON_NOTIFY_ERROR') == '1'


def get_analysis_worker_count(symbol_count):
    if symbol_count <= 0:
        return 1

    try:
        configured_workers = int(os.getenv('ATHENA_ANALYSIS_WORKERS', DEFAULT_ANALYSIS_WORKERS))
    except ValueError:
        configured_workers = DEFAULT_ANALYSIS_WORKERS

    return max(1, min(MAX_ANALYSIS_WORKERS, configured_workers, symbol_count))


def analyze_symbol_worker(symbol, data_folder, dom_path, btc_path):
    csv_path = os.path.join(data_folder, f"{symbol.replace('/', '_')}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    artemis = Artemis(data_folder=data_folder)
    brain = AthenaBrain()
    aegis = Aegis()

    score_data = artemis.calculate_score(symbol)
    ai_score = brain.train_and_predict(csv_path, dom_path, btc_path)
    risk_level, vol_pct = aegis.assess_risk(csv_path)

    score_data.update({
        'ai_score': ai_score,
        'risk_level': risk_level,
        'vol_pct': vol_pct,
    })
    return score_data


def send_discord_report(webhook_url, payload):
    if not webhook_url:
        save_local_report(payload)
        message = "DISCORD_WEBHOOK_URL belum diset. Report hanya disimpan lokal."
        if should_fail_on_notification_error():
            raise RuntimeError(message)
        print(message)
        return

    try:
        response = requests.post(webhook_url, params={'wait': 'true'}, json=payload, timeout=30)
        if response.status_code in (200, 204):
            if response.status_code == 200:
                message_data = response.json()
                print(
                    "Report accepted by Discord "
                    f"(channel_id={message_data.get('channel_id')}, message_id={message_data.get('id')})."
                )
            else:
                print("Report sent to Discord.")
            return

        save_local_report(payload)
        message = f"Discord webhook gagal dengan status {response.status_code}."
        if should_fail_on_notification_error():
            raise RuntimeError(message)
        print(message)
    except requests.RequestException as e:
        save_local_report(payload)
        message = f"Discord tidak bisa dijangkau dari environment ini: {type(e).__name__}"
        if should_fail_on_notification_error():
            raise RuntimeError(message) from e
        print(message)


def build_btc_dominance_proxy(data_folder, symbols):
    btc_path = os.path.join(data_folder, 'BTC_USDT.csv')
    if not os.path.exists(btc_path):
        return None

    btc_df = pd.read_csv(btc_path)
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df['quote_volume'] = btc_df['close'] * btc_df['volume']
    merged = btc_df[['timestamp', 'quote_volume']].rename(columns={'quote_volume': 'BTC_USDT'})

    for symbol in symbols:
        if symbol in [BTC_SYMBOL, DOMINANCE_SYMBOL]:
            continue

        csv_path = os.path.join(data_folder, f"{symbol.replace('/', '_')}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[symbol.replace('/', '_')] = df['close'] * df['volume']
        merged = merged.merge(df[['timestamp', symbol.replace('/', '_')]], on='timestamp', how='left')

    volume_columns = [column for column in merged.columns if column != 'timestamp']
    if len(volume_columns) < 2:
        return None

    merged[volume_columns] = merged[volume_columns].fillna(0.0)
    total_quote_volume = merged[volume_columns].sum(axis=1)
    merged['close'] = (merged['BTC_USDT'] / total_quote_volume.replace(0, pd.NA)) * 100
    proxy_df = merged[['timestamp', 'close']].dropna()

    if len(proxy_df) < 2:
        return None

    proxy_path = os.path.join(data_folder, DOMINANCE_PROXY_FILE)
    proxy_df.to_csv(proxy_path, index=False)
    return proxy_path


def run_athena():
    print("ATHENA v0.8.5 - The Sieve Engine starting...")

    provider = BinanceProvider()
    hermes = Hermes()
    validator = AthenaValidator()

    watch_list = provider.get_top_volume_coins(limit=WATCH_LIST_LIMIT)
    macro_symbols = [BTC_SYMBOL]

    for symbol in macro_symbols:
        if symbol not in watch_list:
            watch_list.append(symbol)

    print(f"Syncing historical data ({SCAN_TIMEFRAME} timeframe)...")
    current_prices = {}
    for symbol in watch_list:
        data = provider.fetch_ohlcv(symbol, timeframe=SCAN_TIMEFRAME, limit=1000)
        provider.save_to_csv(data, symbol, data_folder=DATA_FOLDER)
        if data is not None and not data.empty:
            current_prices[symbol] = data['close'].iloc[-1]
        time.sleep(0.05)

    hits, total = validator.validate_yesterday(current_prices, min_age_hours=VALIDATION_HOURS)
    win_rate = validator.get_win_rate(days=7)
    performance = validator.get_performance_summary(days=7)
    print(f"Validation: {hits}/{total} hits. 7-day win rate: {win_rate}%")

    btc_path = os.path.join(DATA_FOLDER, 'BTC_USDT.csv')
    dom_path = build_btc_dominance_proxy(DATA_FOLDER, watch_list)

    if not os.path.exists(btc_path):
        raise RuntimeError("BTC data tidak tersedia. Report tidak bisa dibuat.")

    if dom_path is None:
        print("BTC dominance proxy tidak tersedia. ATHENA lanjut tanpa fitur dominance.")

    apollo = Apollo(btc_path, dom_path)
    market_status = apollo.analyze_trend()

    print("Analyzing market context and opportunities...")
    results = []
    analysis_symbols = [
        symbol for symbol in watch_list
        if symbol not in macro_symbols and symbol != DOMINANCE_SYMBOL
    ]
    worker_count = get_analysis_worker_count(len(analysis_symbols))
    print(f"Parallel analysis: {len(analysis_symbols)} symbols with {worker_count} workers.")

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_symbol = {
            executor.submit(analyze_symbol_worker, symbol, DATA_FOLDER, dom_path, btc_path): symbol
            for symbol in analysis_symbols
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

    if not results:
        raise RuntimeError("Tidak ada koin yang berhasil dianalisis. Report tidak dikirim.")

    high_confidence_results = [
        item for item in results
        if is_signal_candidate(item)
    ]
    top_opps = sorted(
        high_confidence_results,
        key=lambda x: x['ai_score'],
        reverse=True,
    )[:TOP_OPPORTUNITY_LIMIT]
    scalp_list = sorted(
        high_confidence_results,
        key=lambda x: x['vol_pct'],
        reverse=True,
    )[:SCALPER_LIMIT]
    top_pick = select_top_pick(top_opps)
    top_symbols = {item['symbol'] for item in top_opps}
    scalp_symbols = {item['symbol'] for item in scalp_list}

    for item in results:
        validator.log_prediction(
            item['symbol'],
            item['price'],
            item['ai_score'],
            risk_level=item['risk_level'],
            vol_pct=item['vol_pct'],
            change_4h=item['change_24h'],
            is_top_opportunity=item['symbol'] in top_symbols,
            is_scalper_hotlist=item['symbol'] in scalp_symbols,
        )

    market_regime, _, dom_part = market_status.partition(" | ")
    btc_dom_status = dom_part.replace("BTC.D is ", "") if dom_part else "UNAVAILABLE"
    report_title = hermes.title_with_signal_warning(f"Market Regime: {market_regime}", performance)

    payload = {
        "username": "ATHENA Intelligence",
        "content": "**ATHENA 4H Intelligence Report**",
        "embeds": [
            {
                "title": report_title,
                "color": 0x2ecc71 if market_regime == "BULLISH" else 0xe74c3c,
                "fields": [
                    {
                        "name": "BTC Dominance",
                        "value": btc_dom_status.title(),
                        "inline": True,
                    },
                    {
                        "name": "7-Day Win Rate",
                        "value": (
                            f"{win_rate}%\n"
                            f"Avg return: {performance['avg_return_pct']:+.2f}%\n"
                            f"Trades: {performance['trades']}"
                        ),
                        "inline": True,
                    },
                    {
                        "name": "Signal Performance",
                        "value": (
                            f"LONG: {performance['long_win_rate']}%\n"
                            f"SHORT: {performance['short_win_rate']}%\n"
                            f"TOP: {performance['top_win_rate']}%\n"
                            f"SCALPER: {performance['scalper_win_rate']}%"
                        ),
                        "inline": True,
                    },
                    {
                        "name": "Top Pick",
                        "value": format_top_pick(top_pick),
                        "inline": True,
                    },
                    {
                        "name": "Top AI Opportunities",
                        "value": format_opportunity_rows(top_opps),
                        "inline": False,
                    },
                    {
                        "name": "Scalper Hotlist",
                        "value": format_scalper_rows(scalp_list),
                        "inline": False,
                    },
                ],
                "footer": {"text": "ATHENA Engine v0.8.5 The Sieve | AI >= 68 + LOW Risk Filter"},
            },
        ],
    }

    send_discord_report(hermes.webhook_url, payload)


if __name__ == "__main__":
    run_athena()
