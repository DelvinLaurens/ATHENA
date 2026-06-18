import pandas as pd
import pandas_ta as ta

class Apollo:
    def __init__(self, btc_path, dom_path):
        self.btc_df = pd.read_csv(btc_path)
        self.dom_df = pd.read_csv(dom_path)

    def analyze_trend(self):
        # Hitung SMA untuk BTC
        self.btc_df['sma_200'] = ta.sma(self.btc_df['close'], length=200)
        btc_price = self.btc_df['close'].iloc[-1]
        btc_sma = self.btc_df['sma_200'].iloc[-1]
        
        # Hitung tren Dominance (BTC.D)
        dom_now = self.dom_df['close'].iloc[-1]
        dom_before = self.dom_df['close'].iloc[-2]
        dom_trend = "RISING 📈" if dom_now > dom_before else "FALLING 📉"
        
        # Tentukan Regime
        if btc_price > btc_sma:
            market = "BULLISH"
        else:
            market = "BEARISH"
            
        return f"{market} | BTC.D is {dom_trend}"