import websocket
import json
import pandas as pd
import numpy as np
import joblib
import os
import ssl
import logging
import requests
from datetime import datetime
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram_alert(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except: pass

def fetch_expert_data(count: int = 5000) -> pd.DataFrame:
    symbols = ["frxXAUUSD", "XAUUSD", "gold"]
    app_id = "1089"
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    for s in symbols:
        try:
            ws = websocket.create_connection(ws_url, timeout=15, sslopt={"cert_reqs": ssl.CERT_NONE})
            req = {
                "ticks_history": s,
                "count": count,
                "end": "latest",
                "granularity": 300,
                "style": "candles"
            }
            ws.send(json.dumps(req))
            res = json.loads(ws.recv())
            ws.close()
            
            if "candles" in res and len(res["candles"]) > 0:
                df = pd.DataFrame(res["candles"])
                df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
                for col in ['Close', 'Open', 'High', 'Low']:
                    df[col] = df[col].astype(float)
                logging.info(f"✅ Bot 2: Berhasil tarik {len(df)} data history tina {s}.")
                return df
        except Exception as e:
            logging.warning(f"⚠️ Bot 2 Gagal tarik data {s}: {e}")
    return pd.DataFrame()

def get_current_real_price() -> float:
    symbols = ["frxXAUUSD", "XAUUSD", "gold"]
    app_id = "1089"
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    for s in symbols:
        ws = None
        try:
            ws = websocket.create_connection(ws_url, timeout=10, sslopt={"cert_reqs": ssl.CERT_NONE})
            req = {"ticks_history": s, "count": 1, "end": "latest", "granularity": 60, "style": "candles"}
            ws.send(json.dumps(req))
            res = json.loads(ws.recv())
            ws.close()
            if "candles" in res and len(res["candles"]) > 0:
                return float(res["candles"][-1]["close"])
        except:
            if ws:
                try: ws.close()
                except: pass
    return 0.0

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_bot2_model():
    logging.info("🧠 Bot 2: Memulai Pelatihan AI Presisi Tinggi...")
    df = fetch_expert_data(count=5000)
    
    if df.empty or len(df) < 400:
        logging.error("❌ Bot 2: Data teu cukup!")
        return False

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']

    ma200 = close.rolling(window=200).mean()
    ma50 = close.rolling(window=50).mean()
    
    tr = np.maximum(high.values[1:] - low.values[1:], 
                    np.maximum(abs(high.values[1:] - close.values[:-1]), 
                               abs(low.values[1:] - close.values[:-1])))
    atr_series = pd.Series(tr).rolling(window=14).mean()
    atr = pd.Series(atr_series, index=df.index).bfill().fillna(1.0)
    
    rsi = calculate_rsi(close, 14).fillna(50)
    momentum = close.diff(3).fillna(0)
    body_size = abs(close - open_p)

    df_feat = pd.DataFrame({
        'Close': close,
        'MA200': ma200,
        'MA50': ma50,
        'ATR': atr,
        'RSI': rsi,
        'Momentum': momentum,
        'BodySize': body_size
    }).dropna()

    X, y = [], []
    for i in range(200, len(df_feat) - 3):
        row = df_feat.iloc[i]
        
        # Fitur diperkaya agar AI punya pola tren yang jelas
        features = [
            float(row['ATR']), 
            float(row['BodySize']), 
            float(row['Close'] - row['MA200']), 
            float(row['MA50'] - row['MA200']),
            float(row['RSI']),
            float(row['Momentum'])
        ]
        
        cur_c = row['Close']
        current_atr = row['ATR']
        future_move = df_feat['Close'].iloc[i+3] - cur_c  # Cek pergerakan 3 candle ke depan

        # Labeling ketat khusus tren kuat (menghindari noise sideways)
        if future_move > (current_atr * 1.5):
            X.append(features); y.append(1)
        elif future_move < -(current_atr * 1.5):
            X.append(features); y.append(0)

    if len(X) < 30:
        logging.error("❌ Sampel latihan teuing saeutik, dibatalkeun.")
        return False

    X = np.array(X)
    y = np.array(y)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menggunakan estimator yang lebih stabil untuk klasifikasi arah tren
    clf1 = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced')
    clf2 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

    mastermind_model = VotingClassifier(
        estimators=[('rf', clf1), ('gb', clf2)],
        voting='soft'
    )

    logging.info("🚀 Bot 2: Fitting Model AI Presisi...")
    mastermind_model.fit(X_train, y_train)

    score = mastermind_model.score(X_test, y_test)
    logging.info(f"✨ Bot 2 Sukses Dilatih! Akurasi test: {score * 100:.2f}%")

    model_filename = "model_bot2.pkl"
    joblib.dump(mastermind_model, model_filename)
    
    real_price = get_current_real_price()
    wib_tz = pytz.timezone('Asia/Jakarta')
    current_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')

    notification_card = (
        f"🧠🔥 *[BOT 2: AI PRESISI TINGGI]* 🔥🧠\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"✨ *Status*: `Model Berhasil Diperbarui`\n"
        f"🎯 *Akurasi Model*: `{score * 100:.2f}%`\n"
        f"💵 *Harga Real XAUUSD*: `{real_price:.2f}`\n"
        "-------------------------------------\n"
        f"🩴 *Info*: `Sinyal disaring ketat, siap ngebul!`\n"
        f"⏰ *Waktu*: `{current_time}`"
    )
    send_telegram_alert(notification_card)
    return True

if __name__ == "__main__":
    train_bot2_model()
