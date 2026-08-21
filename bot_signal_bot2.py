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
        self.model = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')
        self.state_file = "bot2_state.json"

    def _safe_load(self, path):
        if path and os.path.exists(path):
            try:
                m = joblib.load(path)
                logging.info(f"🧠 Genius AI: Model kasohor ti {path} sukses dimuat!")
                return m
            except Exception as e:
                logging.warning(f"⚠️ Gagal muat model: {e}")
        return None

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

    def evaluate_genius_strategy(self) -> dict:
        df = self.fetch_market_data(count=250)
        if df.empty or len(df) < 210:
            return {"valid": False}

        df_closed = df.iloc[:-1]
        close = df_closed['Close']
        high = df_closed['High']
        low = df_closed['Low']
        open_p = df_closed['Open']
        current_price = df['Close'].iloc[-1]

        ma200 = close.rolling(window=200).mean().iloc[-1]
        ma50 = close.rolling(window=50).mean().iloc[-1]
        
        tr = np.maximum(high.values[1:] - low.values[1:], np.maximum(abs(high.values[1:] - close.values[:-1]), abs(low.values[1:] - close.values[:-1])))
        atr = float(np.mean(tr[-14:]) if len(tr) >= 14 else (high.iloc[-1] - low.iloc[-1]))
        
        rsi_s = self.calculate_rsi(close, 14)
        rsi = float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0
        momentum = float(close.diff(3).iloc[-1])

        h5 = np.max(high.values[-6:-1])
        l5 = np.min(low.values[-6:-1])
        body_size = abs(close.iloc[-1] - open_p.iloc[-1])
        avg_body = np.mean(abs(close.iloc[-10:] - open_p.iloc[-10:]))

        is_buy_setup = (current_price > ma200) and (current_price > ma50) and (current_price > h5) and (atr >= 0.7) and (rsi > 50 and rsi < 72)
        is_sell_setup = (current_price < ma200) and (current_price < ma50) and (current_price < l5) and (atr >= 0.7) and (rsi < 50 and rsi > 28)

        ai_approved = False
        confidence = 0.0
        if self.model is not None:
            try:
                features = np.array([[float(atr), float(body_size), float(current_price - ma200), float(momentum), float(rsi)]])
                pred = self.model.predict(features)[0]
                probs = self.model.predict_proba(features)[0]
                confidence = float(np.max(probs))
                
                if is_buy_setup and pred == 1 and confidence >= 0.85:
                    ai_approved = True
                elif is_sell_setup and pred == 0 and confidence >= 0.85:
                    ai_approved = True
            except Exception as e:
                logging.warning(f"AI Prediction Error: {e}")

        final_buy = is_buy_setup and ai_approved and (body_size > (avg_body * 1.3)) and (close.iloc[-1] > open_p.iloc[-1])
        final_sell = is_sell_setup and ai_approved and (body_size > (avg_body * 1.3)) and (close.iloc[-1] < open_p.iloc[-1])

        if final_buy:
            return {
                "valid": True, "signal": "BUY", "price": current_price, "conf": confidence,
                "sl": current_price - (atr * 1.5), 
                "tp": current_price + (atr * 5.0)
            }
        elif final_sell:
            return {
                "valid": True, "signal": "SELL", "price": current_price, "conf": confidence,
                "sl": current_price + (atr * 1.5), 
                "tp": current_price - (atr * 5.0)
            }

        return {"valid": False}

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
            card = (
                f"🩴💥 *[BOT 2: DISABET KU SENDAL JEPIT]* 💥🩴\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔥 *EKSEKUSI*: `STRONG {current_signal}`\n"
                f"🧠 *Akurasi AI*: `{result['conf']*100:.1f}%`\n"
                f"💵 *Harga Masuk (OP)*: `{result['price']:.2f}`\n"
                f"🛑 *Stop Loss (SL)*: `{result['sl']:.2f}`\n"
                f"🎯 *Take Profit (TP)*: `{result['tp']:.2f}` (Banting Jauh Boss!)\n"
                "-------------------------------------\n"
                f"⏰ *WAKTU*: `{datetime.now(bot.wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')}`"
            )
            send_telegram_alert(card)
            bot.save_last_signal(current_signal)
            logging.info(f"✅ Sinyal Genius {current_signal} suksés dikirim ka Telegram!")
        else:
            logging.info(f"ℹ️ Sinyal masih {current_signal}, teu dikirim (Anti-spam aktif).")
    else:
        logging.info("ℹ️ Pasar tacan nyugemakeun / AI nolak sinyal palsu.")
