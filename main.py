import os
import time

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
SCAN_TIMEFRAME = '4h'
VALIDATION_HOURS = 4
DATA_FOLDER = os.path.join('data', 'raw', SCAN_TIMEFRAME)


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
    dom_path = os.path.join(DATA_FOLDER, 'BTCDOMUSDT.csv')

    if not os.path.exists(btc_path):
        raise RuntimeError("BTC data tidak tersedia. Report Discord tidak bisa dibuat.")

    if not os.path.exists(dom_path):
        print("BTC dominance data tidak tersedia. ATHENA lanjut tanpa fitur dominance.")
        dom_path = None

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

    opp_rows = ""
    for item in top_opps:
        price_emoji = "UP" if item['change_24h'] > 0 else "DOWN"
        ai_emoji = "BULL" if item['ai_score'] > 50 else "BEAR"
        risk_emoji = "SAFE" if item['risk_level'] == "LOW" else "WATCH" if item['risk_level'] == "MEDIUM" else "HIGH"
        opp_rows += (
            f"{price_emoji} **{item['symbol']}**\n"
            f"- AI: {ai_emoji} {item['ai_score']:.1f}% | "
            f"Risk: {risk_emoji} {item['risk_level']} | "
            f"4h: {item['change_24h']:.2f}%\n\n"
        )

    scalp_rows = ""
    for item in scalp_list:
        ai_direction = "BULL" if item['ai_score'] > 50 else "BEAR"
        scalp_rows += (
            f"**{item['symbol']}** | Vol: **{item['vol_pct']}%** | "
            f"AI: {ai_direction} {item['ai_score']:.1f}%\n"
        )

    payload = {
        "username": "ATHENA Intelligence",
        "content": (
            f"**ATHENA 4H INTELLIGENCE REPORT**\n"
            f"Market: **{market_status}**\n"
            f"AI 7-Day Win Rate: **{win_rate}%**"
        ),
        "embeds": [
            {
                "title": "TOP 10 AI OPPORTUNITIES",
                "description": opp_rows,
                "color": 0x00ff00,
            },
            {
                "title": "SCALPER'S HOTLIST (Highest Volatility)",
                "description": scalp_rows,
                "color": 0xff0000,
            },
        ],
        "footer": {"text": "ATHENA Engine v0.6 | 4H Data-Driven Intelligence"},
    }

    if not hermes.webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL belum diset. Report tidak dikirim.")

    try:
        response = requests.post(hermes.webhook_url, json=payload, timeout=30)
        if response.status_code == 204:
            print(f"Report sent. Market: {market_status} | Win rate: {win_rate}%")
        else:
            raise RuntimeError(f"Discord webhook gagal: {response.status_code} {response.text}")
    except requests.RequestException as e:
        raise RuntimeError(f"Discord webhook request gagal: {e}") from e


if __name__ == "__main__":
    run_athena()
