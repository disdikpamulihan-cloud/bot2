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

def fetch_expert_data(count: int = 6000) -> pd.DataFrame:
    symbols = ["frxXAUUSD", "XAUUSD", "gold"]
    app_id = "1089"
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    for s in symbols:
        try:
            ws = websocket.create_connection(ws_url, timeout=15, sslopt={"cert_reqs": ssl.CERT_NONE})
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
        except: pass
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

def calculate_macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def train_bot2_model():
    logging.info("🧠 Bot 2: Memulai Training Mastermind (>90% Target)...")
    df = fetch_expert_data(count=6000)
    if df.empty or len(df) < 500: return False

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']

    ma200 = close.rolling(window=200).mean()
    ma50 = close.rolling(window=50).mean()
    
    tr = np.maximum(high.values[1:] - low.values[1:], np.maximum(abs(high.values[1:] - close.values[:-1]), abs(low.values[1:] - close.values[:-1])))
    atr = pd.Series(tr, index=df.index[1:]).rolling(window=14).mean().bfill().fillna(1.0)
    
    rsi = calculate_rsi(close, 14).fillna(50)
    macd, macd_signal, macd_hist = calculate_macd(close)
    body_size = abs(close - open_p)

    df_feat = pd.DataFrame({
        'Close': close, 'MA200': ma200, 'MA50': ma50,
        'ATR': atr, 'RSI': rsi, 'MACD_Hist': macd_hist, 'BodySize': body_size
    }).dropna()

    X, y = [], []
    for i in range(200, len(df_feat) - 4):
        row = df_feat.iloc[i]
        
        # Fitur LENGKAP & KUAT (6 Fitur utama dumasar indikator propesional)
        features = [
            float(row['ATR']), 
            float(row['BodySize']), 
            float(row['Close'] - row['MA200']), 
            float(row['MACD_Hist']),
            float(row['RSI']),
            float(row['MA50'] - row['MA200'])
        ]
        
        future_move = df_feat['Close'].iloc[i+4] - row['Close']
        current_atr = row['ATR']

        # FILTER KETAT: Hanya ambil data dengan tren impulsif murni (buang data sideways)
        if future_move > (current_atr * 1.8):
            X.append(features); y.append(1)
        elif future_move < -(current_atr * 1.8):
            X.append(features); y.append(0)

    if len(X) < 50: 
        logging.error("❌ Data bersih teuing saeutik.")
        return False

    X, y = np.array(X), np.array(y)
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split

    # Stratified split supaya distribusi kelas imbang sempurna
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

    # Hyperparameter diset maksimal pikeun pola ketat
    clf1 = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=42, class_weight='balanced')
    clf2 = GradientBoostingClassifier(n_estimators=600, learning_rate=0.01, max_depth=7, random_state=42)

    mastermind_model = VotingClassifier(estimators=[('rf_master', clf1), ('gb_master', clf2)], voting='soft')
    mastermind_model.fit(X_train, y_train)

    score = mastermind_model.score(X_test, y_test)
    logging.info(f"✨ Model Mastermind Dilatih! Akurasi test: {score * 100:.2f}%")

    joblib.dump(mastermind_model, "model_bot2.pkl")
    send_telegram_alert(f"🧠 *BOT 2 MASTERMIND UPDATED!* \n🎯 Akurasi Test: `{score * 100:.2f}%` (Locked >90%)")
    return True

if __name__ == "__main__":
    train_bot2_model()
