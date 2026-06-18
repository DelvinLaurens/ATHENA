import os

import pandas as pd
import pandas_ta as ta


class Apollo:
    def __init__(self, btc_path, dom_path=None):
        self.btc_df = pd.read_csv(btc_path)
        self.dom_df = pd.read_csv(dom_path) if dom_path and os.path.exists(dom_path) else None

    def analyze_trend(self):
        # Market regime utama tetap memakai BTC vs SMA 200.
        self.btc_df['sma_200'] = ta.sma(self.btc_df['close'], length=200)
        btc_price = self.btc_df['close'].iloc[-1]
        btc_sma = self.btc_df['sma_200'].iloc[-1]

        if self.dom_df is not None and len(self.dom_df) >= 2:
            dom_now = self.dom_df['close'].iloc[-1]
            dom_before = self.dom_df['close'].iloc[-2]
            dom_trend = "RISING" if dom_now > dom_before else "FALLING"
        else:
            dom_trend = "UNAVAILABLE"

        if pd.notna(btc_sma) and btc_price > btc_sma:
            market = "BULLISH"
        else:
            market = "BEARISH"

        return f"{market} | BTC.D is {dom_trend}"
