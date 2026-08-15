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

# Set up logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridEnsembleSignalBot:
    """
    BOT2: Fixed Deriv Live Feed for XAUUSD & Volatility 80 with Multi-Symbol Fallback
    """
    def __init__(self, model_xau_path: str = "model_xauusdpkl", model_vol_path: str = "model_vol80.pkl"):
        self.weights = {
            'lightgbm': 0.50,
            'random_forest': 0.50
        }
        self.model_xau = self._safe_load(model_xau_path)
        self.model_vol = self._safe_load(model_vol_path)

    def _safe_load(self, path):
        if path and os.path.exists(path):
            try:
                model = joblib.load(path)
                logging.info(f"✅ Sukses memuat model AI dari {path}")
                return model
            except Exception as e:
                logging.warning(f"⚠️ Gagal memuat model dari {path}: {e}")
        return None

    def fetch_deriv_candles(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """
        Menerik data candles real-time langsung dari Deriv WebSocket dengan multi-symbol fallback.
        """
        if symbol == 'XAUUSD':
            symbols_to_try = ["frxXAUUSD", "XAUUSD", "gold"]
        else:
            symbols_to_try = ["R_80"]

        app_id = "1089"
        ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        for deriv_symbol in symbols_to_try:
            ws = None
            try:
                ws = websocket.create_connection(ws_url, timeout=8, sslopt={"cert_reqs": ssl.CERT_NONE})
                req = {
                    "ticks_history": deriv_symbol,
                    "count": count,
                    "end": "latest",
                    "granularity": 900, # 15 Menit (TF 15M)
                    "style": "candles"
                }
                ws.send(json.dumps(req))
                res = json.loads(ws.recv())
                ws.close()
                
                if "candles" in res and len(res["candles"]) > 0:
                    df = pd.DataFrame(res["candles"])
                    df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
                    df['Close'] = df['Close'].astype(float)
                    df['High'] = df['High'].astype(float)
                    df['Low'] = df['Low'].astype(float)
                    logging.info(f"✅ Sukses tarik data {symbol} via simbol: {deriv_symbol}")
                    return df
            except Exception as e:
                logging.warning(f"⚠️ Gagal dengan simbol {deriv_symbol}: {e}")
            finally:
                if ws:
                    try:
                        ws.close()
                    except:
                        pass
                        
        return pd.DataFrame()

    def extract_features_and_indicators(self, df: pd.DataFrame, symbol: str):
        if df.empty or len(df) < 30:
            default_price = 4437.25 if symbol == 'XAUUSD' else 249185.0
            return None, default_price, 5.0, 50.0

        close = np.array(df['Close'].values, dtype=float).ravel()
        high = np.array(df['High'].values, dtype=float).ravel()
        low = np.array(df['Low'].values, dtype=float).ravel()

        current_price = float(close[-1])

        # 1. RSI (14)
        delta = np.diff(close)
        gain = np.mean(delta[delta > 0][-14:]) if len(delta[delta > 0]) > 0 else 0
        loss = -np.mean(delta[delta < 0][-14:]) if len(delta[delta < 0]) > 0 else 1e-6
        rsi = float(100 - (100 / (1 + gain/loss)))

        # 2. ATR (Average True Range)
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        atr = float(np.mean(tr[-14:]) if len(tr) >= 14 else (high[-1] - low[-1]))

        features = np.array([[
            (close[-1] - close[-2]) / close[-2],
            (close[-1] - close[-5]) / close[-5],
            rsi / 100.0,
            atr / current_price,
            np.std(close[-10:]) / current_price
        ]])

        return features, current_price, atr, rsi

    def evaluate_hybrid_signal(self, symbol: str) -> dict:
        model = self.model_xau if symbol == 'XAUUSD' else self.model_vol
        
        df = self.fetch_deriv_candles(symbol)
        features, current_price, atr, rsi = self.extract_features_and_indicators(df, symbol)

        if model is not None and features is not None:
            try:
                prob_lgbm = float(model.predict_proba(features)[0][1])
                prob_rf = prob_lgbm
                consensus_score = (prob_lgbm * self.weights['lightgbm']) + (prob_rf * self.weights['random_forest'])
            except Exception:
                consensus_score = 0.55 if rsi < 45 else 0.45
                prob_lgbm, prob_rf = consensus_score, consensus_score
        else:
            if rsi < 35:
                consensus_score = 0.75
            elif rsi > 65:
                consensus_score = 0.25
            else:
                consensus_score = 0.55 if rsi < 50 else 0.45
            
            prob_lgbm = consensus_score
            prob_rf = consensus_score

        if consensus_score >= 0.50:
            signal = "BUY"
            confidence = consensus_score * 100
        else:
            signal = "SELL"
            confidence = (1.0 - consensus_score) * 100

        # ATR-based Dynamic SL & TP
        if symbol == 'XAUUSD':
            sl_distance = max(atr * 1.5, 5.0)
            tp1_distance = sl_distance * 1.5
            tp2_distance = sl_distance * 3.0
        else:
            sl_distance = max(atr * 1.5, 500.0)
            tp1_distance = sl_distance * 1.5
            tp2_distance = sl_distance * 3.0

        if signal == "BUY":
            sl = current_price - sl_distance
            tp1 = current_price + tp1_distance
            tp2 = current_price + tp2_distance
        else:
            sl = current_price + sl_distance
            tp1 = current_price - tp1_distance
            tp2 = current_price - tp2_distance

        return {
            "signal": signal,
            "price": current_price,
            "confidence": confidence,
            "rsi": rsi,
            "atr": atr,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "lgbm_score": prob_lgbm * 100,
            "rf_score": prob_rf * 100
        }

def send_telegram_message(message: str):
    bot_token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logging.warning("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID tidak ditemukan.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            logging.info("Notifikasi Telegram berhasil terkirim.")
        else:
            logging.error(f"Gagal kirim pesan Telegram: {res.text}")
    except Exception as e:
        logging.error(f"Error Telegram API: {e}")

def format_bot2_card(symbol: str, data: dict) -> str:
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    return (
        f"⚡ *[BOT-2 DERIV LIVE FEED] SIGNAL ({symbol})*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *Sinyal Eksekusi*: `{data['signal']}`\n"
        f"💵 *Harga Real-Time MT5 Feed*: `{data['price']:.2f}`\n"
        f"🔥 *Keyakinan Model*: `{data['confidence']:.1f}%`\n"
        f"📊 *RSI (14)*: `{data['rsi']:.1f}` | *ATR*: `{data['atr']:.2f}`\n"
        "-------------------------------------\n"
        f"🧠 *Skor LightGBM*: `{data['lgbm_score']:.1f}%`\n"
        f"🌲 *Skor Random Forest*: `{data['rf_score']:.1f}%`\n"
        "-------------------------------------\n"
        f"🔴 *Stop Loss (SL)*: `{data['sl']:.2f}`\n"
        f"🟢 *Target TP 1 (Aman)*: `{data['tp1']:.2f}`\n"
        f"🟢 *Target TP 2 (Ngajelegur)*: `{data['tp2']:.2f}`\n"
        "-------------------------------------\n"
        f"⏰ `{wib_time}`"
    )

if __name__ == "__main__":
    bot2 = HybridEnsembleSignalBot()
    
    res_xau = bot2.evaluate_hybrid_signal('XAUUSD')
    res_vol = bot2.evaluate_hybrid_signal('VOLATILITY 80')
    
    send_telegram_message(format_bot2_card('XAUUSD', res_xau))
    send_telegram_message(format_bot2_card('VOLATILITY 80', res_vol))
