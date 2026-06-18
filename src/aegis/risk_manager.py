import pandas as pd
import pandas_ta as ta

class Aegis:
    def __init__(self):
        pass

    def assess_risk(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            if df.empty or len(df) < 15:
                return "HIGH", 0.0

            # Hitung ATR (Average True Range)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            if pd.isna(df['atr'].iloc[-1]) or pd.isna(df['close'].iloc[-1]):
                return "HIGH", 0.0

            last_price = df['close'].iloc[-1]
            last_atr = df['atr'].iloc[-1]

            if last_price == 0:
                return "HIGH", 0.0

            # Volatility Score (Persentase ATR terhadap Harga)
            # Semakin tinggi %, semakin berisiko koin tersebut
            volatility_pct = (last_atr / last_price) * 100

            if volatility_pct < 3.5:
                risk_level = "LOW"
            elif volatility_pct < 7.0:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            return risk_level, round(volatility_pct, 2)
        except Exception:
            return "HIGH", 0.0

if __name__ == "__main__":
    # Test Aegis
    aegis = Aegis()
    level, score = aegis.assess_risk('data/raw/BTC_USDT.csv')
    print(f"BTC Risk Level: {level} ({score}%)")
