import websocket
import json
import pandas as pd
import numpy as np
import joblib
import os
import ssl
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_expert_data(count: int = 3500) -> pd.DataFrame:
    """Narik database history panglobana pikeun latihan Bot 2."""
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
                "granularity": 300, # TF 5 Menit
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

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_bot2_model():
    logging.info("🧠 Bot 2: Memulai Pelatihan AI Tingkat Lanjut (Custom Mastermind)...")
    df = fetch_expert_data(count=3500)
    
    if df.empty or len(df) < 400:
        logging.error("❌ Bot 2: Data teu cukup!")
        return False

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']

    # Indikator tingkat luhur
    ma200 = close.rolling(window=200).mean()
    ma50 = close.rolling(window=50).mean()
    ma20 = close.rolling(window=20).mean()
    
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = pd.Series(tr).rolling(window=14).mean()
    rsi = calculate_rsi(close, 14)

    df_feat = pd.DataFrame({
        'Close': close,
        'MA200': ma200,
        'MA50': ma50,
        'MA20': ma20,
        'ATR': atr,
        'RSI': rsi,
        'BodySize': abs(close - open_p)
    }).dropna()

    X = []
    y = []

    # Filtrasi ketat khusus Bot 2 (Meredam noise pasar emas)
    for i in range(200, len(df_feat) - 3):
        row = df_feat.iloc[i]
        
        # Fitur multi-dimensi cerdas
        features = [
            float(row['ATR']), 
            float(row['BodySize']), 
            float(row['Close'] - row['MA200']), 
            float(row['MA50'] - row['MA20']),
            float(row['RSI'])
        ]
        
        future_prices = df_feat['Close'].iloc[i+1 : i+4]
        max_f = future_prices.max()
        min_f = future_prices.min()
        cur_c = row['Close']

        # Labeling presisi tinggi
        if (max_f - cur_c) > (row['ATR'] * 1.3) and (cur_c - min_f) < (row['ATR'] * 0.4):
            X.append(features); y.append(1) # BUY Kuat
        elif (cur_c - min_f) > (row['ATR'] * 1.3) and (max_f - cur_c) < (row['ATR'] * 0.4):
            X.append(features); y.append(0) # SELL Kuat

    if len(X) < 50:
        logging.warning("⚠️ Bot 2: Sampel terlalu ketat, menyesuaikan otomatis...")
        for i in range(200, len(df_feat) - 1):
            row = df_feat.iloc[i]
            features = [float(row['ATR']), float(row['BodySize']), float(row['Close'] - row['MA200']), float(row['MA50'] - row['MA20']), float(row['RSI'])]
            f_ret = df_feat['Close'].iloc[i+1] - row['Close']
            if f_ret > 0.25: X.append(features); y.append(1)
            elif f_ret < -0.25: X.append(features); y.append(0)

    X = np.array(X)
    y = np.array(y)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Arsitektur Gabungan 3 Model Pintar
    clf1 = RandomForestClassifier(n_estimators=350, max_depth=12, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=250, learning_rate=0.015, max_depth=6, random_state=42)

    mastermind_model = VotingClassifier(
        estimators=[('rf_master', clf1), ('gb_master', clf2)],
        voting='soft'
    )

    logging.info("🚀 Bot 2: Melatih Mastermind AI...")
    mastermind_model.fit(X_train, y_train)

    score = mastermind_model.score(X_test, y_test)
    logging.info(f"✨ Bot 2 AI Sukses Dilatih! Akurasi test: {score * 100:.2f}%")

    model_filename = "model_bot2.pkl"
    joblib.dump(mastermind_model, model_filename)
    logging.info(f"💾 Bot 2 Model disimpan ke {model_filename}!")
    return True

if __name__ == "__main__":
    train_bot2_model()
