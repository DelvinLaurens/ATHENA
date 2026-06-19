import os
import time

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


def format_opportunity_rows(items):
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
    print("ATHENA v0.6 - 4H Performance Tracking Engine starting...")

    provider = BinanceProvider()
    brain = AthenaBrain()
    artemis = Artemis(data_folder=DATA_FOLDER)
    aegis = Aegis()
    hermes = Hermes()
    validator = AthenaValidator()

    watch_list = provider.get_top_volume_coins(limit=50)
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
    for symbol in watch_list:
        if symbol in macro_symbols or symbol == DOMINANCE_SYMBOL:
            continue

        try:
            csv_path = os.path.join(DATA_FOLDER, f"{symbol.replace('/', '_')}.csv")
            score_data = artemis.calculate_score(symbol)
            ai_score = brain.train_and_predict(csv_path, dom_path)
            validator.log_prediction(symbol, score_data['price'], ai_score)
            risk_level, vol_pct = aegis.assess_risk(csv_path)

            score_data.update({
                'ai_score': ai_score,
                'risk_level': risk_level,
                'vol_pct': vol_pct,
            })
            results.append(score_data)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if not results:
        raise RuntimeError("Tidak ada koin yang berhasil dianalisis. Report tidak dikirim.")

    top_opps = sorted(results, key=lambda x: x['ai_score'], reverse=True)[:10]
    scalp_list = sorted(results, key=lambda x: x['vol_pct'], reverse=True)[:5]
    best_setup = next(
        (item for item in top_opps if item['ai_score'] >= 70 and item['risk_level'] != "HIGH"),
        top_opps[0],
    )

    market_regime, _, dom_part = market_status.partition(" | ")
    btc_dom_status = dom_part.replace("BTC.D is ", "") if dom_part else "UNAVAILABLE"

    payload = {
        "username": "ATHENA Intelligence",
        "content": "**ATHENA 4H Intelligence Report**",
        "embeds": [
            {
                "title": f"Market Regime: {market_regime}",
                "color": 0x2ecc71 if market_regime == "BULLISH" else 0xe74c3c,
                "fields": [
                    {
                        "name": "BTC Dominance",
                        "value": btc_dom_status.title(),
                        "inline": True,
                    },
                    {
                        "name": "7-Day Win Rate",
                        "value": f"{win_rate}%",
                        "inline": True,
                    },
                    {
                        "name": "Best Setup",
                        "value": (
                            f"{best_setup['symbol']} - {best_setup['ai_score']:.1f}% AI, "
                            f"{risk_label(best_setup['risk_level'])} risk"
                        ),
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
                "footer": {"text": "ATHENA Engine v0.6 | 4H Data-Driven Intelligence"},
            },
        ],
    }

    send_discord_report(hermes.webhook_url, payload)


if __name__ == "__main__":
    run_athena()
