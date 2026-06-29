import requests
import os
from dotenv import load_dotenv

load_dotenv()

class Hermes:
    LONG_CAUTION_THRESHOLD = 45.0

    def __init__(self):
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')

    @classmethod
    def title_with_signal_warning(cls, title, performance):
        try:
            long_win_rate = float(performance.get('long_win_rate', 0.0))
        except (TypeError, ValueError):
            long_win_rate = 0.0

        if long_win_rate < cls.LONG_CAUTION_THRESHOLD:
            return f"[⚠️ CAUTION ON LONGS] {title}"
        return title

    def send_report(self, market_status, ranking_df):
        if not self.webhook_url:
            print("❌ Webhook URL tidak ditemukan di .env")
            return

        # Ambil Top 3 untuk dihighlight
        top_3 = ranking_df.head(3)
        
        # Buat format pesan
        rows = ""
        for _, row in ranking_df.iterrows():
            emoji = "🔼" if row['change_24h'] > 0 else "🔻"
            rows += f"{emoji} **{row['symbol']}** | Price: {row['price']:.4f} | RSI: {row['rsi']:.1f} | Change: {row['change_24h']:.2f}%\n"

        payload = {
            "content": "🏛 **ATHENA DAILY INTELLIGENCE REPORT**",
            "embeds": [
                {
                    "title": f"Market Regime: {market_status}",
                    "description": f"**Top Opportunities Today:**\n\n{rows}",
                    "color": 0x00ff00 if "BULLISH" in market_status else 0xff0000,
                    "footer": {"text": "Powered by ATHENA Engine v0.1"}
                }
            ]
        }

        response = requests.post(self.webhook_url, json=payload)
        if response.status_code == 204:
            print("✅ Laporan berhasil dikirim ke Discord!")
        else:
            print(f"❌ Gagal mengirim pesan: {response.text}")

if __name__ == "__main__":
    # Test sederhana
    import pandas as pd
    test_df = pd.DataFrame([{'symbol': 'BTC/USDT', 'price': 60000, 'rsi': 45, 'change_24h': 2.5}])
    hermes = Hermes()
    hermes.send_report("TEST REGIME", test_df)
