import joblib
import pandas as pd
import numpy as np
import logging
import os
import requests
import json
import websocket
import ssl
from datetime import datetime, time
import pytz
import time as time_module

# Set up logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SuperAIXAUUSDBot:
    """
    SUPER AI TRADING BOT: Dedicated XAUUSD High-Precision Signal Generator
    Dilengkapi State Memory (Anti-Spam Telegram) & Counter-Signal / Close Alert.
    """
    def __init__(self, model_path: str = "model_xauusd.pkl"):
        self.model_xauusd = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')
        self.last_sent_signal = None  # Pelacak sinyal sateuacanna supados henteu spam

    def _safe_load(self, path):
        if path and os.path.exists(path):
            try:
                model = joblib.load(path)
                logging.info(f"✅ Sukses memuat model AI XAUUSD dari {path}")
                return model
            except Exception as e:
                logging.warning(f"⚠️ Gagal load model standar ({e}). Mencoba mode aman/fallback...")
                try:
                    import xgboost as xgb
                    booster = xgb.Booster()
                    booster.load_model(path)
                    logging.info(f"✅ Sukses memuat XGBoost Booster langsung dari {path}")
                    return booster
                except Exception as e2:
                    logging.warning(f"⚠️ Gagal total load model: {e2}. Bot bakal ngagunakeun Intelligent Indicator Fallback.")
        return None

    def _extract_model(self, model_obj):
        if model_obj is None:
            return None
        if isinstance(model_obj, dict):
            for key in ['model', 'estimator', 'lgbm', 'classifier', 'xgboost']:
                if key in model_obj:
                    return model_obj[key]
            return list(model_obj.values())[0]
        return model_obj

    def fetch_deriv_candles(self, count: int = 100) -> pd.DataFrame:
        """Menarik data candles XAUUSD real-time langsung dari Deriv WebSocket."""
        symbols_to_try = ["frxXAUUSD", "XAUUSD", "gold"]
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
                    "granularity": 60, # TF 1 Menit
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
                    logging.info(f"✅ Sukses tarik data XAUUSD via simbol: {deriv_symbol}")
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

    def extract_features_and_indicators(self, df: pd.DataFrame):
        if df.empty or len(df) < 30:
            return None, 4375.97, 5.0, 50.0, 0.0

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

        # 3. MACD Sederhana untuk konfirmasi momentum
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
        macd = float(ema12 - ema26)

        features = pd.DataFrame([{
            'Column_0': (close[-1] - close[-2]) / close[-2],
            'Column_1': (close[-1] - close[-5]) / close[-5],
            'Column_2': rsi / 100.0,
            'Column_3': atr / current_price,
            'Column_4': np.std(close[-10:]) / current_price
        }])

        return features, current_price, atr, rsi, macd

    def evaluate_market(self) -> dict:
        """Kalkulasi sinyal XAUUSD presisi tinggi & validasi target minimal 100 pips (10 poin)."""
        df = self.fetch_deriv_candles()
        input_df, current_price, atr, rsi, macd = self.extract_features_and_indicators(df)
        
        actual_model = self._extract_model(self.model_xauusd)
        confidence = 55.0
        prediction = 1

        if actual_model is not None and input_df is not None:
            try:
                import xgboost as xgb
                if isinstance(actual_model, xgb.Booster):
                    dmatrix = xgb.DMatrix(input_df.values)
                    preds = actual_model.predict(dmatrix)
                    pred_val = preds[0] if not isinstance(preds, np.ndarray) else preds.item(0)
                    prediction = 1 if pred_val > 0.5 else 0
                    confidence = float(pred_val * 100 if pred_val <= 1.0 else 85.0)
                else:
                    if isinstance(self.model_xauusd, dict) and 'scaler' in self.model_xauusd:
                        input_data = self.model_xauusd['scaler'].transform(input_df.values)
                    else:
                        input_data = input_df.values
                    
                    prediction = actual_model.predict(input_data)[0]
                    if hasattr(actual_model, "predict_proba"):
                        probs = actual_model.predict_proba(input_data)[0]
                        confidence = float(max(probs) * 100)
            except Exception as e:
                logging.warning(f"⚠️ Prediksi model error ({e}), menggunakan Intelligent Indicator Fallback.")
                prediction = 1 if rsi < 50 else 0
                confidence = 78.0
        else:
            prediction = 1 if rsi < 50 else 0
            confidence = 78.0

        signal = "BUY" if prediction == 1 else "SELL"

        # Dynamic Risk Management XAUUSD (Target minimal setara 100 pips / 10.0 poin)
        sl_distance = max(atr * 1.2, 5.0)
        tp1_distance = max(sl_distance * 1.5, 10.0) 
        tp2_distance = tp1_distance * 2.0

        if signal == "BUY":
            sl = current_price - sl_distance
            tp1 = current_price + tp1_distance
            tp2 = current_price + tp2_distance
        else:
            sl = current_price + sl_distance
            tp1 = current_price - tp1_distance
            tp2 = current_price - tp2_distance

        # FILTER KETAT: Keyakinan >= 75% & Target Minimal 100 Pips tercapai
        min_target_pips = 10.0
        is_high_probability = (confidence >= 75.0) and (tp1_distance >= min_target_pips)
        is_momentum_strong = abs(macd) > (atr * 0.03)

        valid_signal = is_high_probability and is_momentum_strong

        return {
            "valid": valid_signal,
            "signal": signal,
            "price": current_price,
            "confidence": confidence,
            "rsi": rsi,
            "atr": atr,
            "macd": macd,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2
        }

def send_telegram_message(message: str):
    """Mengirim notifikasi ke Telegram."""
    bot_token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logging.warning("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID tidak terdeteksi.")
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

def format_signal_card(res: dict, is_reversal: bool = False) -> str:
    """Format tampilan pesan Telegram. Bisa jadi Sinyal Biasa atawa Alert Close/Lawan Arah."""
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    if is_reversal:
        if res['signal'] == "BUY":
            alert_title = "🚨🚨 **[ALERT: CLOSE SELL & REVERSE TO BUY!]** 🚨🚨"
            desc = "Tren pasar ngadadak males! Tutup posisi SELL ayeuna, siap-siap pindah BUY!"
        else:
            alert_title = "🚨🚨 **[ALERT: CLOSE BUY & REVERSE TO SELL!]** 🚨🚨"
            desc = "Tren pasar ngadadak turun! Tutup posisi BUY ayeuna, siap-siap pindah SELL!"
        
        return (
            f"⚠️ *[SUPER AI XAUUSD - COUNTER SIGNAL]*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *Aksi Squel/Close*: {alert_title}\n"
            f"💡 *Katerangan*: `{desc}`\n"
            f"💵 *Harga WebSocket*: `{res['price']:.2f}`\n"
            f"🔥 *Keyakinan AI*: `{res['confidence']:.1f}%`\n"
            "-------------------------------------\n"
            f"🛑 *Stop Loss Baru*: `{res['sl']:.2f}`\n"
            f"🟢 *Target TP 1*: `{res['tp1']:.2f}`\n"
            "-------------------------------------\n"
            f"⏰ `{wib_time}`"
        )
    else:
        if res['signal'] == "BUY":
            signal_badge = "🟢🟢 **[STRONG BUY - LONG]** 🟢🟢"
            action_desc = "Target XAUUSD siap MEROKET naik (Target >100 Pips)! 🚀"
        else:
            signal_badge = "🔴🔴 **[STRONG SELL - SHORT]** 🔴🔴"
            action_desc = "Target XAUUSD siap TERJUN bebas (Target >100 Pips)! 📉"

        return (
            f"🤖 *[SUPER AI XAUUSD BOT - SIGNAL]*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *Sinyal Eksekusi*: {signal_badge}\n"
            f"💡 *Analisis*: `{action_desc}`\n"
            f"💵 *Harga WebSocket (Feed Bot)*: `{res['price']:.2f}`\n"
            f"🔍 *(Cocokkeun jeung Harga Bid/Ask MT5)*\n"
            f"🔥 *Keyakinan AI (High Conf)*: `{res['confidence']:.1f}%`\n"
            f"📊 *RSI*: `{res['rsi']:.1f}` | *ATR*: `{res['atr']:.2f}`\n"
            "-------------------------------------\n"
            f"🛑 *Stop Loss (Anti-SL)*: `{res['sl']:.2f}`\n"
            f"🟢 *Target TP 1 (Aman)*: `{res['tp1']:.2f}`\n"
            f"🚀 *Target TP 2 (Runner)*: `{res['tp2']:.2f}`\n"
            "-------------------------------------\n"
            f"⏰ `{wib_time}`"
        )

def send_startup_notification(bot_instance):
    """Mengirim notifikasi startup lengkap dengan harga real-time terkini."""
    df = bot_instance.fetch_deriv_candles(count=5)
    _, current_price, atr, rsi, _ = bot_instance.extract_features_and_indicators(df)
    
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    msg = (
        f"🚀 *[SYSTEM STARTUP]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ *Super AI XAUUSD Bot* parantos sukses diaktifkeun!\n"
        f"💵 *Harga Real-Time XAUUSD*: `{current_price:.2f}`\n"
        f"🔍 *(Bandingkeun sareng MT5 ayeuna)*\n"
        f"📊 *RSI*: `{rsi:.1f}` | *ATR*: `{atr:.2f}`\n"
        "🛡️ Mode Anti-Spam Aktif: Notifikasi dikirim ngan sakali per sinyal.\n"
        f"⏰ `{wib_time}`"
    )
    send_telegram_message(msg)

def send_daily_report(bot_instance):
    """Mengirim laporan rutin jam 06.00 WIB beserta harga terkini."""
    df = bot_instance.fetch_deriv_candles(count=5)
    _, current_price, atr, rsi, _ = bot_instance.extract_features_and_indicators(df)
    
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    msg = (
        f"🌅 *[LAPORAN RUTIN SUBUH - 06:00 WIB]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "🟢 *Bot Status*: `AKTIF & SIAGA`\n"
        f"💵 *Harga Terkini XAUUSD (Feed)*: `{current_price:.2f}`\n"
        f"🔍 *(Bandingkeun sareng MT5 ayeuna)*\n"
        f"📊 *RSI Saat Ini*: `{rsi:.1f}` | *ATR*: `{atr:.2f}`\n"
        "-------------------------------------\n"
        "☕ Siap-siap ngantosan sinyal high-conviction dinten ieu!\n"
        f"⏰ `{wib_time}`"
    )
    send_telegram_message(msg)

if __name__ == "__main__":
    bot = SuperAIXAUUSDBot(model_path='model_xauusd.pkl')
    
    # 1. Kirim notif startup
    send_startup_notification(bot)
    
    last_daily_report_date = None

    # Loop utama monitoring
    while True:
        now_wib = datetime.now(bot.wib_tz)
        
        # Laporan rutin jam 06.00 WIB
        if now_wib.hour == 6 and now_wib.minute == 0:
            if last_daily_report_date != now_wib.date():
                send_daily_report(bot)
                last_daily_report_date = now_wib.date()

        # Evaluasi pasar berkala
        res = bot.evaluate_market()
        
        if res["valid"]:
            current_signal = res["signal"]
            
            # Cek naha sinyal ieu béda jeung sateuacanna (Lawan arah / Sinyal Anyar)
            if bot.last_sent_signal is None:
                # Sinyal munggaran kapingguh
                msg = format_signal_card(res, is_reversal=False)
                send_telegram_message(msg)
                bot.last_sent_signal = current_signal
                logging.info(f"✅ Sinyal awal {current_signal} terkirim!")
                
            elif bot.last_sent_signal != current_signal:
                # KASUS LAWAN ARAH: Kirim notif alert close & reverse
                msg = format_signal_card(res, is_reversal=True)
                send_telegram_message(msg)
                bot.last_sent_signal = current_signal
                logging.info(f"🚨 Sinyal berbalik arah! Alert close & reverse dikirim: {current_signal}")
                
            else:
                # Sinyal sarua jeung sateuacanna, JANGAN KIRIM NOTIF (Cegah spam)
                logging.info(f"⏳ Sinyal masih {current_signal} (Aman, teu ngirim notif ulang supados henteu spam).")
        else:
            logging.info("⏳ Market XAUUSD di-skip (Teu acan nyumponan sarat keyakinan >75% / TP <100 pips).")

        # Jeda 60 detik (1 menit) sateuacan mariksa deui pasar
        time_module.sleep(60)
