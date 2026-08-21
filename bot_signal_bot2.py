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

class GeniusBot2Sniper:
    def __init__(self, model_path: str = "model_bot2.pkl"):
        self.model, self.scaler = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')
        self.state_file = "bot2_state.json"

    def _safe_load(self, path):
        if path and os.path.exists(path):
            try:
                loaded_data = joblib.load(path)
                if isinstance(loaded_data, tuple):
                    logging.info(f"🧠 Genius AI: Model & Scaler sukses dimuat ti {path}!")
                    return loaded_data[0], loaded_data[1]
                else:
                    logging.info(f"🧠 Genius AI: Model tunggal sukses dimuat ti {path}!")
                    return loaded_data, None
            except Exception as e:
                logging.warning(f"⚠️ Gagal muat model: {e}")
        return None, None

    def fetch_market_data(self, count: int = 250) -> pd.DataFrame:
        symbols = ["frxXAUUSD", "XAUUSD", "gold"]
        app_id = "1089"
        ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        for attempt in range(3):
            for s in symbols:
                ws = None
                try:
                    ws = websocket.create_connection(ws_url, timeout=10, sslopt={"cert_reqs": ssl.CERT_NONE})
                    req = {"ticks_history": s, "count": count, "end": "latest", "granularity": 300, "style": "candles"}
                    ws.send(json.dumps(req))
                    res = json.loads(ws.recv())
                    ws.close()
                    
                    if "candles" in res and len(res["candles"]) > 0:
                        df = pd.DataFrame(res["candles"])
                        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
                        for col in ['Close', 'Open', 'High', 'Low']:
                            df[col] = df[col].astype(float)
                        return df
                except:
                    time_module.sleep(2)
                finally:
                    if ws:
                        try: ws.close()
                        except: pass
            if attempt < 2: time_module.sleep(3)
        return pd.DataFrame()

    def load_last_signal(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f).get("signal", None)
            except: pass
        return None

    def save_last_signal(self, sig):
        try:
            with open(self.state_file, "w") as f:
                json.dump({"signal": sig}, f)
        except: pass

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series, slow=26, fast=12, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def evaluate_genius_strategy(self) -> dict:
        df = self.fetch_market_data(count=250)
        if df.empty or len(df) < 210:
            return {"valid": False, "conf": 0.0}

        df_closed = df.iloc[:-1]
        close = df_closed['Close']
        open_p = df_closed['Open']
        current_price = df['Close'].iloc[-1]

        ma200 = close.rolling(window=200).mean().iloc[-1]
        ma50 = close.rolling(window=50).mean().iloc[-1]
        
        high = df_closed['High']
        low = df_closed['Low']
        tr = np.maximum(high.values[1:] - low.values[1:], np.maximum(abs(high.values[1:] - close.values[:-1]), abs(low.values[1:] - close.values[:-1])))
        atr_series = pd.Series(tr).rolling(14).mean()
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 1.0
        avg_atr = float(atr_series.rolling(20).mean().iloc[-1]) if len(atr_series) >= 20 else atr
        
        rsi_s = self.calculate_rsi(close, 14)
        rsi = float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0
        
        _, _, macd_hist_series = self.calculate_macd(close)
        macd_hist = float(macd_hist_series.iloc[-1]) if not macd_hist_series.empty else 0.0
        
        body_size = abs(close.iloc[-1] - open_p.iloc[-1])
        
        # Setup saringan optimal (fleksibel tapi tetep aman)
        is_buy_setup = (current_price > ma200) or (rsi > 48)
        is_sell_setup = (current_price < ma200) or (rsi < 52)

        ai_approved = False
        confidence = 0.0
        final_signal = None
        
        if self.model is not None:
            try:
                features = np.array([[
                    float(atr), 
                    float(body_size), 
                    float(current_price - ma200), 
                    float(macd_hist),
                    float(rsi),
                    float(ma50 - ma200)
                ]])
                
                if self.scaler is not None:
                    features = self.scaler.transform(features)
                
                pred = self.model.predict(features)[0]
                probs = self.model.predict_proba(features)[0]
                confidence = float(np.max(probs))
                
                # Confidence diset stabil di 70% supaya sinyal gampang kaluar
                if confidence >= 0.70:
                    if pred == 1 and is_buy_setup:
                        ai_approved = True
                        final_signal = "BUY"
                    elif pred == 0 and is_sell_setup:
                        ai_approved = True
                        final_signal = "SELL"
            except Exception as e:
                logging.warning(f"AI Prediction Error: {e}")

        # --- RUMUS SL & TP DINAMIS (Nyesuaikeun Kaayaan Pasar Harita) ---
        # Jarak robah sacara otomatis gumantung kana volatilitas real (ATR) dibandingkeun rata-rata ATR
        volatility_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        dynamic_multiplier = np.clip(volatility_ratio, 0.7, 1.6)

        sl_distance = atr * (1.1 * dynamic_multiplier)
        tp_distance = atr * (2.8 * dynamic_multiplier)

        if ai_approved:
            if final_signal == "BUY":
                return {
                    "valid": True, "signal": "BUY", "price": current_price, "conf": confidence,
                    "sl": current_price - sl_distance, "tp": current_price + tp_distance
                }
            elif final_signal == "SELL":
                return {
                    "valid": True, "signal": "SELL", "price": current_price, "conf": confidence,
                    "sl": current_price + sl_distance, "tp": current_price - tp_distance
                }

        return {"valid": False, "conf": confidence, "price": current_price}

def send_telegram_alert(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except: pass

if __name__ == "__main__":
    bot = GeniusBot2Sniper('model_bot2.pkl')
    result = bot.evaluate_genius_strategy()
    
    if result.get("valid"):
        current_signal = result["signal"]
        last_signal = bot.load_last_signal()
        
        if current_signal != last_signal:
            if current_signal == "BUY":
                sig_display = "🟢 `BUY`"
            else:
                sig_display = "🔴 `SELL`"

            card = (
                f"🚨 **BOT 2 MASTERMIND UPDATED!** 🚨\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• **EKSEKUSI** : {sig_display}\n"
                f"• **HARGA REAL** : `{result['price']:.2f}`\n"
                f"• **AKURASI LIVE** : `{result['conf']*100:.2f}%`\n"
                f"• **OPEN POSISI** : `{result['price']:.2f}`\n"
                f"• **TP** : `{result['tp']:.2f}`\n"
                f"• **SL** : `{result['sl']:.2f}`\n"
                f"• **WAKTU** : `{datetime.now(bot.wib_tz).strftime('%Y-%m-%d %H:%M:%S')} WIB`"
            )
            send_telegram_alert(card)
            bot.save_last_signal(current_signal)
            logging.info(f"✅ Sinyal {current_signal} suksés dikirim!")
        else:
            logging.info("ℹ️ Sinyal masih sami, anti-spam aktif.")
    else:
        live_conf = result.get("conf", 0.0) * 100
        live_price = result.get("price", 0.0)
        logging.info(f"ℹ️ Bot Aktif | Harga: {live_price:.2f} | Akurasi AI: {live_conf:.2f}% (Menunggu setup optimal)")
