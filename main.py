from src.oracle.binance_provider import BinanceProvider
from src.apollo.regime_engine import Apollo
from src.artemis.ranker import Artemis
from src.hermes.notifier import Hermes
from src.brain.intelligence import AthenaBrain
from src.aegis.risk_manager import Aegis
from src.brain.validator import AthenaValidator
import requests
import time
import os

BTC_SYMBOL = 'BTC/USDT'
DOMINANCE_SYMBOL = 'BTCDOMUSDT'
SCAN_TIMEFRAME = '4h'
VALIDATION_HOURS = 4
DATA_FOLDER = os.path.join('data', 'raw', SCAN_TIMEFRAME)

def run_athena():
    print("🏛 ATHENA v0.6 - Performance Tracking Engine Starting...")
    
    provider = BinanceProvider()
    brain = AthenaBrain()
    artemis = Artemis(data_folder=DATA_FOLDER)
    aegis = Aegis()
    hermes = Hermes()
    validator = AthenaValidator()

    # 1. Ambil List Koin & Tambahkan Macro Symbol
    watch_list = provider.get_top_volume_coins(limit=50)
    macro_symbols = [BTC_SYMBOL]
    
    for s in macro_symbols:
        if s not in watch_list:
            watch_list.append(s)

    # 2. Sync Data
    print(f"Syncing Historical Data ({SCAN_TIMEFRAME} Timeframe)...")
    current_prices = {}
    for s in watch_list:
        data = provider.fetch_ohlcv(s, timeframe=SCAN_TIMEFRAME, limit=1000)
        provider.save_to_csv(data, s, data_folder=DATA_FOLDER)
        # Simpan harga terakhir untuk validasi
        if data is not None:
            current_prices[s] = data['close'].iloc[-1]
        time.sleep(0.05)
        
    # 3. Validasi Hasil Kemarin & Hitung Win Rate
    hits, total = validator.validate_yesterday(current_prices, min_age_hours=VALIDATION_HOURS)
    win_rate = validator.get_win_rate(days=7)
    print(f"📊 Validation: {hits}/{total} Hits. 7-Day Win Rate: {win_rate}%")

    # 4. Market Context (Apollo)
    btc_path = os.path.join(DATA_FOLDER, 'BTC_USDT.csv')
    dom_path = os.path.join(DATA_FOLDER, 'BTCDOMUSDT.csv')

    if not os.path.exists(btc_path):
        print("BTC data tidak tersedia. Kemungkinan fetch Binance gagal di runner GitHub.")
        print("ATHENA berhenti tanpa mengirim report agar workflow tidak crash.")
        return

    if not os.path.exists(dom_path):
        print("BTC dominance data tidak tersedia. ATHENA lanjut tanpa fitur dominance.")
        dom_path = None

    apollo = Apollo(btc_path, dom_path)
    market_status = apollo.analyze_trend()
    
    # 5. Processing AI & Risk
    print("🧠 Analyzing Market Context & Opportunities...")
    results = []
    for symbol in watch_list:
        if symbol in macro_symbols or symbol == DOMINANCE_SYMBOL:
            continue
        
        try:
            csv_path = os.path.join(DATA_FOLDER, f"{symbol.replace('/', '_')}.csv")
            score_data = artemis.calculate_score(symbol)
            
            # AI Prediction
            ai_score = brain.train_and_predict(csv_path, dom_path)
            
            # Log Prediksi Hari Ini untuk divalidasi besok
            validator.log_prediction(symbol, score_data['price'], ai_score)
            
            # Risk & Volatility
            risk_level, vol_pct = aegis.assess_risk(csv_path)
            
            score_data.update({
                'ai_score': ai_score, 
                'risk_level': risk_level, 
                'vol_pct': vol_pct
            })
            results.append(score_data)
        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e}")
            continue

    if not results:
        print("Tidak ada koin yang berhasil dianalisis. Report tidak dikirim.")
        return

    # 6. Ranking
    top_opps = sorted(results, key=lambda x: x['ai_score'], reverse=True)[:10]
    scalp_list = sorted(results, key=lambda x: x['vol_pct'], reverse=True)[:5]
    
    # 7. Bangun Laporan Discord
    opp_rows = ""
    for item in top_opps:
        price_emoji = "🔼" if item['change_24h'] > 0 else "🔻"
        ai_emoji = "🧠" if item['ai_score'] > 50 else "📉"
        risk_emoji = "🛡️" if item['risk_level'] == "LOW" else "⚠️" if item['risk_level'] == "MEDIUM" else "🔥"
        opp_rows += (f"{price_emoji} **{item['symbol']}**\n"
                     f"└ AI: {ai_emoji} {item['ai_score']:.1f}% | Risk: {risk_emoji} {item['risk_level']} | 24h: {item['change_24h']:.2f}%\n\n")

    scalp_rows = ""
    for item in scalp_list:
        ai_direction = "🚀" if item['ai_score'] > 50 else "🧊" 
        scalp_rows += f"🔥 **{item['symbol']}** | Vol: **{item['vol_pct']}%** | AI: {ai_direction} {item['ai_score']:.1f}%\n"

    # Kirim ke Discord
    payload = {
        "username": "ATHENA Intelligence",
        "content": f"🏛 **ATHENA 4H INTELLIGENCE REPORT**\n"
                   f"Market: **{market_status}**\n"
                   f"📈 **AI 7-Day Win Rate: {win_rate}%**",
        "embeds": [
            {
                "title": "🎯 TOP 10 AI OPPORTUNITIES",
                "description": opp_rows,
                "color": 0x00ff00
            },
            {
                "title": "🔥 SCALPER'S HOTLIST (Highest Volatility)",
                "description": scalp_rows,
                "color": 0xff0000
            }
        ],
        "footer": {"text": "ATHENA Engine v0.6 | 4H Data-Driven Intelligence"}
    }
    
    if not hermes.webhook_url:
        print("DISCORD_WEBHOOK_URL belum diset. Report dibuat, tapi tidak dikirim.")
        return

    try:
        response = requests.post(hermes.webhook_url, json=payload, timeout=30)
        if response.status_code == 204:
            print(f"✅ Report Sent. Market: {market_status} | Win Rate: {win_rate}%")
        else:
            print(f"❌ Failed to send Discord report: {response.text}")
    except requests.RequestException as e:
        print(f"❌ Failed to send Discord report: {e}")

if __name__ == "__main__":
    run_athena()
