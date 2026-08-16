import joblib
import pandas as pd
import numpy as np
import logging
import os
import requests
import json
import websocket
import ssl
from datetime import datetime
import pytz
import time as time_module

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmartXAUUSDBot:
    def __init__(self, model_path: str = "model_xauusd.pkl"):
        self.model_xauusd = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')

    def _safe_load(self, path):
        if path and os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                logging.warning(f"⚠️ Gagal load model: {e}")
        return None

    def fetch_deriv_candles(self, symbol="frxXAUUSD", count=200, granularity=60):
        """Ambil data candle Deriv WebSocket"""
        app_id = "1089"
        ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        ws = None
        try:
            ws = websocket.create_connection(ws_url, timeout=8, sslopt={"cert_reqs": ssl.CERT_NONE})
            req = {
                "ticks_history": symbol,
                "count": count,
                "end": "latest",
                "granularity": granularity,
                "style": "candles"
            }
            ws.send(json.dumps(req))
            res = json.loads(ws.recv())
            ws.close()
            if "candles" in res and len(res["candles"]) > 0:
                df = pd.DataFrame(res["candles"])
                df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
                df[['Close','High','Low']] = df[['Close','High','Low']].astype(float)
                return df
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
        finally:
            if ws:
                try: ws.close()
                except: pass
        return pd.DataFrame()

    def calc_indicators(self, df: pd.DataFrame):
        """Hitung indikator teknikal"""
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values

        # RSI (14)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().iloc[-1]
        avg_loss = pd.Series(loss).rolling(14).mean().iloc[-1]
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))

        # ATR (14)
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        atr = pd.Series(tr).rolling(14).mean().iloc[-1]

        # MACD
        ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
        macd = ema12 - ema26

        # Bollinger Bands (20)
        ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
        std20 = pd.Series(close).rolling(20).std().iloc[-1]
        upper_bb = ma20 + 2*std20
        lower_bb = ma20 - 2*std20

        # Stochastic (14)
        lowest_low = pd.Series(low).rolling(14).min().iloc[-1]
        highest_high = pd.Series(high).rolling(14).max().iloc[-1]
        stochastic = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low + 1e-6)

        return {
            "price": close[-1],
            "rsi": rsi,
            "atr": atr,
            "macd": macd,
            "upper_bb": upper_bb,
            "lower_bb": lower_bb,
            "stochastic": stochastic
        }

    def evaluate_signal(self):
        """Evaluasi sinyal multi-timeframe"""
        df1m = self.fetch_deriv_candles(count=200, granularity=60)
        df15m = self.fetch_deriv_candles(count=200, granularity=900)

        if df1m.empty or df15m.empty:
            return {"valid": False, "reason": "Data kosong"}

        ind1m = self.calc_indicators(df1m)
        ind15m = self.calc_indicators(df15m)

        # Rule sederhana: konfirmasi multi-timeframe
        signal = None
        if ind1m["rsi"] < 30 and ind15m["macd"] > 0 and ind1m["price"] < ind1m["lower_bb"]:
            signal = "BUY"
        elif ind1m["rsi"] > 70 and ind15m["macd"] < 0 and ind1m["price"] > ind1m["upper_bb"]:
            signal = "SELL"

        valid = signal is not None and (ind1m["stochastic"] < 20 or ind1m["stochastic"] > 80)

        return {
            "valid": valid,
            "signal": signal,
            "price": ind1m["price"],
            "rsi": ind1m["rsi"],
            "atr": ind1m["atr"],
            "macd": ind1m["macd"],
            "stochastic": ind1m["stochastic"],
            "upper_bb": ind1m["upper_bb"],
            "lower_bb": ind1m["lower_bb"]
        }

if __name__ == "__main__":
    bot = SmartXAUUSDBot()
    res = bot.evaluate_signal()
    print(res)
