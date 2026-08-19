import websocket
import json
import pandas as pd
import numpy as np
import joblib
import os
import ssl
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_expert_data(count: int = 4000) -> pd.DataFrame:
    """Narik data sajarah panglobana pikeun latihan model Sniper Ultra-Jitu."""
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
                logging.info(f"✅ Bot 2 Ultra: Berhasil tarik {len(df)} data history tina {s}.")
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
    logging.info("🧠 Bot 2 Ultra: Memulai Pelatihan AI High-Probability & Momentum...")
    df = fetch_expert_data(count=4000)
    
    if df.empty or len(df) < 400:
        logging.error("❌ Bot 2 Ultra: Data teu cukup!")
        return False

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']

    # Indikator Komplit Kelas Institusi
    ma200 = close.rolling(window=200).mean()
    ma50 = close.rolling(window=50).mean()
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = pd.Series(tr).rolling(window=14).mean()
    rsi = calculate_rsi(close, 14)
    
    # Momentum Velocity (Ukuran kecepatan pergerakan harga)
    momentum = close.diff(3)

    df_feat = pd.DataFrame({
        'Close': close,
        'MA200': ma200,
        'MA50': ma50,
        'ATR': atr,
        'RSI': rsi,
        'Momentum': momentum,
        'BodySize': abs(close - open_p)
    }).dropna()

    X = []
    y = []

    # Filter Ekstrim: AI ngan diajar tina momentum anu langsung ngajelegur (Direct Explosion)
    for i in range(200, len(df_feat) - 2):
        row = df_feat.iloc[i]
        
        features = [
            float(row['ATR']), 
            float(row['BodySize']), 
            float(row['Close'] - row['MA200']), 
            float(row['Momentum']),
            float(row['RSI'])
        ]
        
        # Target: Harga 2 candle ka hareup kedah langsung lumpat minimal 2 kali ATR tanpa loba nyolok ka tukang
        next_c1 = df_feat['Close'].iloc[i+1]
        next_c2 = df_feat['Close'].iloc[i+2]
        cur_c = row['Close']
        current_atr = row['ATR']

        if (next_c1 - cur_c) > (current_atr * 1.0) and (next_c2 - cur_c) > (current_atr * 1.8):
            X.append(features); y.append(1) # BUY Ngajelegur
        elif (cur_c - next_c1) > (current_atr * 1.0) and (cur_c - next_c2) > (current_atr * 1.8):
            X.append(features); y.append(0) # SELL Ngajelegur

    if len(X) < 40:
        logging.warning("⚠️ Bot 2: Sampel terlalu ketat, melonggarkeun sakedik...")
        for i in range(200, len(df_feat) - 1):
            row = df_feat.iloc[i]
            features = [float(row['ATR']), float(row['BodySize']), float(row['Close'] - row['MA200']), float(row['Momentum']), float(row['RSI'])]
            f_ret = df_feat['Close'].iloc[i+1] - row['Close']
            if f_ret > 0.4: X.append(features); y.append(1)
            elif f_ret < -0.4: X.append(features); y.append(0)

    X = np.array(X)
    y = np.array(y)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Arsitektur AI Kuat: Random Forest + Gradient Boosting tingkat jero
    clf1 = RandomForestClassifier(n_estimators=400, max_depth=14, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=6, random_state=42)

    mastermind_model = VotingClassifier(
        estimators=[('rf_ultra', clf1), ('gb_ultra', clf2)],
        voting='soft'
    )

    logging.info("🚀 Bot 2 Ultra: Melatih Model AI Sniper...")
    mastermind_model.fit(X_train, y_train)

    score = mastermind_model.score(X_test, y_test)
    logging.info(f"✨ Bot 2 Ultra Sukses Dilatih! Akurasi test: {score * 100:.2f}%")

    model_filename = "model_bot2.pkl"
    joblib.dump(mastermind_model, model_filename)
    logging.info(f"💾 Model Bot 2 Ultra disimpen kana {model_filename}!")
    return True

if __name__ == "__main__":
    train_bot2_model()
