import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import datetime as dt
import time
import requests
import pyotp
import json
import os
import io
import threading
import uuid
import urllib.parse
import yfinance as yf
from SmartApi import SmartConnect
from streamlit.runtime.scriptrunner import add_script_run_ctx
from collections import deque
from streamlit_lightweight_charts import renderLightweightCharts

# Windows 11 Native Notifications (Works only when running locally via CMD)
try:
    from plyer import notification
    HAS_NOTIFY = True
except ImportError:
    HAS_NOTIFY = False

# ==========================================
# 0. DEVELOPER SIGNATURE
# ==========================================
def render_signature():
    st.sidebar.markdown(
        f'<div style="text-align: center; color: #38bdf8; font-size: 0.8rem; font-weight: bold; border-top: 1px solid #38bdf8; padding-top: 10px; margin-top: 10px;">'
        f'🚀 Pro Scalper Engine<br>Developed by: Vijayakumar</div>', 
        unsafe_allow_html=True
    )
    st.session_state['_dev_sig'] = "AUTH_OWNER_VIJAYAKUMAR"

# ==========================================
# 1. SECURITY & CONFIG
# ==========================================
CRED_FILE = "secure_creds.json"
TRADE_FILE = "persistent_trades.csv"

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0b0f19; background-image: linear-gradient(180deg, #0b0f19 0%, #111827 100%); }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #38bdf8 !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > input, input[type="number"], input[type="password"], input[type="text"] {
        color: #000000 !important; font-weight: 900 !important; background-color: #ffffff !important; -webkit-text-fill-color: #000000 !important; 
    }
    [data-testid="stTable"], [data-testid="stTable"] > div > table { background-color: #ffffff !important; width: 100% !important; }
    [data-testid="stTable"] th, [data-testid="stTable"] td { color: #000000 !important; font-weight: 800 !important; border: 1px solid #000000 !important; }
    .main .block-container { padding-bottom: 120px; }
</style>
""", unsafe_allow_html=True)

def load_creds():
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"client_id": "", "pwd": "", "totp_secret": "", "api_key": "", "tg_token": "", "tg_chat": "", "wa_phone": "", "wa_api": ""}

def save_creds(client_id, pwd, totp_secret, api_key, tg_token, tg_chat, wa_phone, wa_api):
    with open(CRED_FILE, 'w') as f: json.dump({"client_id": client_id, "pwd": pwd, "totp_secret": totp_secret, "api_key": api_key, "tg_token": tg_token, "tg_chat": tg_chat, "wa_phone": wa_phone, "wa_api": wa_api}, f)

def save_trade(trade_record):
    df_new = pd.DataFrame([trade_record])
    if not os.path.exists(TRADE_FILE): df_new.to_csv(TRADE_FILE, index=False)
    else: df_new.to_csv(TRADE_FILE, mode='a', header=False, index=False)

DEFAULT_LOTS = {"NIFTY": 75, "BANKNIFTY": 30, "SENSEX": 20, "CRUDEOIL": 100, "NATURALGAS": 1250, "GOLD": 100, "SILVER": 30}
YF_TICKERS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "CRUDEOIL": "CL=F", "GOLD": "GC=F", "SILVER": "SI=F"}

INDEX_TOKENS = {
    "NIFTY": ("NSE", "26000"),
    "BANKNIFTY": ("NSE", "26009"),
    "SENSEX": ("BSE", "99919000")
}

STRAT_LIST = ["Momentum Breakout + S&R", "Institutional FVG + SMC", "Combined Convergence"]

if 'sb_index_input' not in st.session_state: st.session_state.sb_index_input = list(DEFAULT_LOTS.keys())[0]
if 'sb_strat_input' not in st.session_state: st.session_state.sb_strat_input = STRAT_LIST[0]
if 'bot' not in st.session_state: st.session_state.bot = None
if 'prev_index' not in st.session_state: st.session_state.prev_index = "NIFTY"
if 'custom_stock' not in st.session_state: st.session_state.custom_stock = ""
if 'asset_options' not in st.session_state: st.session_state.asset_options = list(DEFAULT_LOTS.keys())

def get_market_status():
    now_ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    if now_ist.weekday() >= 5: return False, "Market Closed (Weekend)"
    if dt.time(9, 15) <= now_ist.time() <= dt.time(15, 30): return True, "Market Live 🟢"
    if dt.time(17, 00) <= now_ist.time() <= dt.time(23, 30): return True, "Commodity Live 🟠"
    return False, "Market Closed (After Hours)"

def check_btst_stbt(df):
    if df is None or len(df) < 5: return "NO DATA"
    df = df.copy()
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    last = df.iloc[-1]
    if last['close'] > last['ema9'] > last['ema21'] and (last['high'] - last['close']) < (last['close'] - last['open']): return "🔥 BTST Suggested"
    elif last['close'] < last['ema9'] < last['ema21'] and (last['close'] - last['low']) < (last['open'] - last['close']): return "🩸 STBT Suggested"
    return "⚖️ Neutral (No Hold)"

@st.cache_data(ttl=43200) 
def get_angel_scrip_master():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df = pd.DataFrame(requests.get(url, timeout=15).json())
        df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce') / 100 
        return df
    except Exception:
        return pd.DataFrame()

# ==========================================
# 2. ADVANCED TECHNICAL ANALYZER 
# ==========================================
class TechnicalAnalyzer:
    def get_atr(self, df, period=14):
        df = df.copy()
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        return df['tr'].rolling(period).mean()

    def calculate_psar(self, df, step=0.02, max_step=0.2):
        psar = df['close'].copy()
        bull = True
        af = step
        hp = df['high'].iloc[0]
        lp = df['low'].iloc[0]
        
        for i in range(2, len(df)):
            if bull:
                psar.iloc[i] = psar.iloc[i-1] + af * (hp - psar.iloc[i-1])
                if df['low'].iloc[i] < psar.iloc[i]:
                    bull = False
                    psar.iloc[i] = hp
                    lp = df['low'].iloc[i]
                    af = step
                else:
                    if df['high'].iloc[i] > hp:
                        hp = df['high'].iloc[i]
                        af = min(af + step, max_step)
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (lp - psar.iloc[i-1])
                if df['high'].iloc[i] > psar.iloc[i]:
                    bull = True
                    psar.iloc[i] = lp
                    hp = df['high'].iloc[i]
                    af = step
                else:
                    if df['low'].iloc[i] < lp:
                        lp = df['low'].iloc[i]
                        af = min(af + step, max_step)
        return psar

    def calculate_indicators(self, df, is_index=False):
        df = df.copy()
        df['vol_sma'] = df['volume'].rolling(20).mean()
        
        if is_index: df['vol_spike'] = True
        else: df['vol_spike'] = df['volume'] > (df['vol_sma'] * 1.5) 
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        rsi_min = df['rsi'].rolling(window=14).min()
        rsi_max = df['rsi'].rolling(window=14).max()
        stoch_rsi = (df['rsi'] - rsi_min) / (rsi_max - rsi_min).replace(0, 1e-10)
        df['stoch_k'] = stoch_rsi.rolling(window=3).mean() * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        atr = self.get_atr(df, 14)
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['is_sideways'] = (df['rsi'].between(45, 55)) & (abs(df['close'] - df['ema21']) < (atr * 0.5))
        return df

    def calculate_fib_zones(self, df, lookback=100):
        major_high = df['high'].rolling(lookback).max().iloc[-1]
        major_low = df['low'].rolling(lookback).min().iloc[-1]
        diff = major_high - major_low
        fib_0_5 = major_high - (diff * 0.5)
        fib_0_618 = major_high - (diff * 0.618)
        return major_high, major_low, min(fib_0_5, fib_0_618), max(fib_0_5, fib_0_618)

    def evaluate_momentum_and_structure(self, df, is_ce):
        last = df.iloc[-1]
        mh, ml, _, _ = self.calculate_fib_zones(df)
        
        near_top = (mh - last['close']) / last['close'] < 0.0015
        near_bottom = (last['close'] - ml) / last['close'] < 0.0015
        
        if is_ce:
            if near_top and not last['vol_spike']: return False, "Near Resistance"
            if last['stoch_k'] < last['stoch_d']: return False, "Momentum Down"
            if last['stoch_k'] > 80 and not last['vol_spike']: return False, "Overbought"
            return True, "Pass"
        else:
            if near_bottom and not last['vol_spike']: return False, "Near Support"
            if last['stoch_k'] > last['stoch_d']: return False, "Momentum Up"
            if last['stoch_k'] < 20 and not last['vol_spike']: return False, "Oversold"
            return True, "Pass"

    def apply_vwap_ema_strategy(self, df, index_name="NIFTY", short_ema=9, long_ema=21):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX"]
        df = self.calculate_indicators(df, is_index)
        df['psar'] = self.calculate_psar(df)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']
        df['ema_short'] = df['close'].ewm(span=short_ema, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=long_ema, adjust=False).mean()
        
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high}
        
        last = df.iloc[-1]
        signal, trend = "WAIT", "FLAT"
        
        if last['is_sideways']:
            trend = "SIDEWAYS 🟡"
            signal = "WAIT"
        else:
            benchmark = last['ema_long'] if is_index else last['vwap']
            if (last['ema_short'] > last['ema_long']) and (last['close'] > benchmark) and (last['psar'] < last['close']):
                trend = "BULLISH MOMENTUM 🟢"
                valid, _ = self.evaluate_momentum_and_structure(df, True)
                if valid: signal = "BUY_CE"
            elif (last['ema_short'] < last['ema_long']) and (last['close'] < benchmark) and (last['psar'] > last['close']):
                trend = "BEARISH MOMENTUM 🔴"
                valid, _ = self.evaluate_momentum_and_structure(df, False)
                if valid: signal = "BUY_PE"

        return trend, signal, last['vwap'], last['ema_short'], df, atr, fib_data

    def apply_fvg_strategy(self, df, index_name="NIFTY", momentum_mult=1.5):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX"]
        df = self.calculate_indicators(df, is_index)
        df['psar'] = self.calculate_psar(df)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean() 
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high}
        
        bullish_fvg = df['low'] > df['high'].shift(2)
        bearish_fvg = df['high'] < df['low'].shift(2)
        body_size = abs(df['close'].shift(1) - df['open'].shift(1))
        avg_body = abs(df['close'] - df['open']).rolling(10).mean()
        strong_disp = body_size > (avg_body * momentum_mult)
        
        last = df.iloc[-1]
        signal, trend = "WAIT", "CONSOLIDATING"
        
        if last['is_sideways']:
            trend = "SIDEWAYS 🟡"
            signal = "WAIT"
        else:
            if (bullish_fvg & strong_disp).iloc[-1] and (last['psar'] < last['close']): 
                trend = "FVG BULLISH 🟢"
                valid, _ = self.evaluate_momentum_and_structure(df, True)
                if valid: signal = "BUY_CE"
            elif (bearish_fvg & strong_disp).iloc[-1] and (last['psar'] > last['close']): 
                trend = "FVG BEARISH 🔴"
                valid, _ = self.evaluate_momentum_and_structure(df, False)
                if valid: signal = "BUY_PE"
                
        return trend, signal, last['vwap'], last['ema_short'], df, atr, fib_data

    def apply_combined_strategy(self, df, index_name="NIFTY"):
        t1, s1, v, e, d, a, f = self.apply_vwap_ema_strategy(df, index_name)
        t2, s2, _, _, _, _, _ = self.apply_fvg_strategy(df, index_name)
        
        signal, trend = "WAIT", "MIXED SIGNALS 🟡"
        if s1 == s2 and s1 != "WAIT":
            signal = s1
            trend = f"CONFLUENCE {t1.split(' ')[0]} 🚀"
        elif t1 != "FLAT" and "SIDEWAYS" not in t1:
            trend = t1
            
        return trend, signal, v, e, d, a, f

# ==========================================
# 3. CORE BOT ENGINE
# ==========================================
class SniperBot:
    def __init__(self, api_key="", client_id="", pwd="", totp_secret="", tg_token="", tg_chat="", wa_phone="", wa_api="", is_mock=False):
        self.api_key, self.client_id, self.pwd, self.totp_secret = api_key, client_id, pwd, totp_secret
        self.tg_token, self.tg_chat, self.wa_phone, self.wa_api = tg_token, tg_chat, wa_phone, wa_api
        self.api, self.token_map, self.is_mock = None, None, is_mock
        self.client_name = "Offline User"
        self.analyzer = TechnicalAnalyzer()
        
        self.state = {
            "is_running": False, "order_in_flight": False, "active_trade": None, "last_trade": None,
            "logs": deque(maxlen=50), "current_trend": "WAIT", "current_signal": "WAIT",
            "spot": 0.0, "vwap": 0.0, "ema": 0.0, "atr": 0.0, "fib_data": {}, "latest_data": None,
            "global_alerts": deque(maxlen=5), "ui_popups": deque(maxlen=10), "loop_count": 0
        }
        self.settings = {}

    def push_notify(self, title, message):
        # 1. ALWAYS push to the web interface queue first
        self.state["ui_popups"].append({"title": title, "message": message})
        
        # 2. Native Windows 11 Notification (Works when running locally)
        if HAS_NOTIFY:
            try: notification.notify(title=title, message=message, app_name="Pro Scalper", timeout=5)
            except: pass
        # 3. Telegram Alert
        if self.tg_token and self.tg_chat:
            try: requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", json={"chat_id": self.tg_chat, "text": f"*{title}*\n{message}", "parse_mode": "Markdown"}, timeout=3)
            except: pass
        # 4. WhatsApp Alert
        if self.wa_phone and self.wa_api:
            try: requests.get(f"https://api.callmebot.com/whatsapp.php?phone={self.wa_phone}&text={urllib.parse.quote(f'*{title}* %0A {message}')}&apikey={self.wa_api}", timeout=3)
            except: pass

    def log(self, msg):
        self.state["logs"].appendleft(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")

    def login(self):
        if self.is_mock: 
            self.client_name = "Paper Trading User"
            self.push_notify("🟢 Session Started", "Simulating offline login for Paper Trading.")
            return True
        try:
            obj = SmartConnect(api_key=self.api_key)
            res = obj.generateSession(self.client_id, self.pwd, pyotp.TOTP(self.totp_secret).now())
            if res['status']:
                self.api = obj
                self.client_name = res['data'].get('name', self.client_id)
                self.log("✅ Exchange Connected Successfully")
                self.push_notify("🟢 Exchange Connected", f"Live session started for user: {self.client_name}")
                return True
            self.log(f"❌ Login failed: {res.get('message', 'Check credentials or TOTP')}")
            return False
        except Exception as e: 
            self.log(f"❌ Login Exception: {e}")
            return False

    def get_master(self):
        if self.token_map is None or self.token_map.empty:
            self.token_map = get_angel_scrip_master()
            if not self.token_map.empty:
                self.log("✅ Loaded Scrip Master (Cached)")
        return self.token_map

    def get_token_info(self, index_name):
        if index_name in INDEX_TOKENS:
            return INDEX_TOKENS[index_name]
        
        df_map = self.get_master()
        if df_map is not None and not df_map.empty:
            today_date = pd.Timestamp(dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)).normalize()
            futs = df_map[(df_map['name'] == index_name) & (df_map['instrumenttype'].isin(['FUTCOM', 'FUTIDX', 'FUTSTK', 'EQ']))]
            if not futs.empty:
                eqs = futs[futs['instrumenttype'] == 'EQ']
                if not eqs.empty:
                    best_eq = eqs.iloc[0]
                    return best_eq['exch_seg'], best_eq['token']
                futs = futs[futs['expiry'] >= today_date]
                if not futs.empty:
                    best_fut = futs[futs['expiry'] == futs['expiry'].min()].iloc[0]
                    return best_fut['exch_seg'], best_fut['token']
        return "NSE", "12345"

    def get_live_price(self, exchange, symbol, token):
        if self.is_mock:
            if symbol in YF_TICKERS:
                try:
                    data = yf.Ticker(YF_TICKERS[symbol]).history(period="1d", interval="1m")
                    if not data.empty: return float(data['Close'].iloc[-1])
                except: pass
            return float(np.random.uniform(22000, 22100)) 
            
        if not self.api: return None
        try:
            res = self.api.ltpData(exchange, symbol, str(token))
            if res and res.get('status'): return float(res['data']['ltp'])
        except Exception: pass
        return None

    def get_historical_data(self, exchange, token, symbol="NIFTY", interval="5m"):
        if self.is_mock:
            return self._fallback_yfinance(symbol, interval)

        if not self.api: return None
        try:
            interval_map = {"1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE"}
            api_interval = interval_map.get(interval, "FIVE_MINUTE")
            
            now_ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
            fromdate = now_ist - dt.timedelta(days=10) 
            
            req_params = {
                "exchange": exchange, 
                "symboltoken": str(token), 
                "interval": api_interval, 
                "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"), 
                "todate": now_ist.strftime("%Y-%m-%d %H:%M")
            }
            res = self.api.getCandleData(req_params)
            
            if res and res.get('status') and res.get('data'):
                df = pd.DataFrame(res['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                if df.empty: 
                    return self._fallback_yfinance(symbol, interval)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.index = df['timestamp']
                return df
            else:
                return self._fallback_yfinance(symbol, interval)
        except Exception: 
            return self._fallback_yfinance(symbol, interval)
            
    def _fallback_yfinance(self, symbol, interval):
        yf_int = interval if interval in ["1m", "5m", "15m"] else "5m" 
        yf_ticker = YF_TICKERS.get(symbol)
        if yf_ticker:
            try:
                period = "5d" if interval == "1m" else "10d" 
                df = yf.Ticker(yf_ticker).history(period=period, interval=yf_int)
                if not df.empty:
                    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                    return df
            except: pass
        
        times = [dt.datetime.now() - dt.timedelta(minutes=5*i) for i in range(150)][::-1]
        base = 22000
        close_prices = base + np.random.normal(0, 10, 150).cumsum()
        df = pd.DataFrame({'timestamp': times, 'open': close_prices - np.random.uniform(0, 5, 150), 'high': close_prices + np.random.uniform(0, 10, 150), 'low': close_prices - np.random.uniform(0, 10, 150), 'close': close_prices, 'volume': np.random.randint(1000, 50000, 150)})
        df.index = df['timestamp']
        return df

    def place_real_order(self, symbol, token, qty, side="BUY", exchange="NFO"):
        if self.is_mock: return "MOCK_" + uuid.uuid4().hex[:6].upper()
        try:
            return self.api.placeOrder({"variety": "NORMAL", "tradingsymbol": symbol, "symboltoken": str(token), "transactiontype": side, "exchange": exchange, "ordertype": "MARKET", "producttype": "INTRADAY", "duration": "DAY", "quantity": str(qty)})
        except Exception as e: 
            self.log(f"Order failed: {e}")
            return None

    def get_strike(self, symbol, spot, signal, max_premium):
        if self.is_mock: 
            opt_type = "CE" if "BUY_CE" in signal else "PE"
            mock_expiry = (pd.Timestamp.today() + pd.Timedelta(days=2)).strftime('%d%b').upper()
            return f"{symbol}{mock_expiry}{int(spot)}{opt_type}", "12345", "NFO", 100.0 
            
        df = self.get_master()
        if df is None or df.empty: return None, None, None, 0.0
        
        is_ce = "BUY_CE" in signal
        opt_type = "CE" if is_ce else "PE"
        
        exch_list, valid_instruments = ["NFO", "MCX", "BFO"], ['OPTIDX', 'OPTSTK', 'OPTFUT', 'OPTCOM']
        today = pd.Timestamp(dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)).normalize()
        
        mask = (df['name'] == symbol) & (df['exch_seg'].isin(exch_list)) & (df['expiry'] >= today) & (df['symbol'].str.endswith(opt_type)) & (df['instrumenttype'].isin(valid_instruments))
        subset = df[mask].copy()
        if subset.empty: return None, None, None, 0.0
        
        closest_expiry = subset['expiry'].min()
        
        if self.settings.get('hero_zero'):
            if closest_expiry.date() != pd.Timestamp.today().date(): return None, None, None, 0.0
            subset = subset[subset['expiry'] == closest_expiry]
            candidates = subset[subset['strike'] >= spot].sort_values('strike', ascending=True) if is_ce else subset[subset['strike'] <= spot].sort_values('strike', ascending=False)
            actual_max_prem = self.settings.get('hz_premium', 15)
            for _, row in candidates.head(15).iterrows():
                ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
                if ltp and ltp <= actual_max_prem: return row['symbol'], row['token'], row['exch_seg'], ltp
            return None, None, None, 0.0
            
        subset = subset[subset['expiry'] == closest_expiry]
        subset['dist_to_spot'] = abs(subset['strike'] - spot)
        candidates = subset.sort_values('dist_to_spot', ascending=True).head(10)
        
        for _, row in candidates.iterrows():
            ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
            if ltp and ltp <= max_premium: 
                return row['symbol'], row['token'], row['exch_seg'], ltp
                
        return None, None, None, 0.0

    def check_fomo_alerts(self, current_index):
        watchlist = ["NIFTY", "BANKNIFTY", "SENSEX", "CRUDEOIL"]
        for sym in watchlist:
            if sym == current_index: continue
            
            exch, token = self.get_token_info(sym)
            df_bg = self.get_historical_data(exch, token, symbol=sym, interval="5m")
            if df_bg is not None and not df_bg.empty:
                trend, sig, _, _, _, _, _ = self.analyzer.apply_vwap_ema_strategy(df_bg, sym)
                if sig != "WAIT":
                    icon = "🚀" if "CE" in sig else "🩸"
                    msg = f"Strong {sig} setup detected on {sym}!"
                    if msg not in self.state["global_alerts"]:
                        self.state["global_alerts"].append(msg)
                        self.push_notify(f"{icon} FOMO ALERT", msg)

    def get_higher_timeframe(self, tf):
        if tf == "1m": return "5m"
        if tf == "3m": return "15m"
        if tf == "5m": return "15m"
        return "15m"

    def trading_loop(self):
        self.log("▶️ Engine thread started.")
        while self.state["is_running"]:
            try:
                is_open, mkt_msg = get_market_status()
                if not is_open:
                    time.sleep(10)
                    continue

                self.state["loop_count"] += 1
                s = self.settings
                index, timeframe = s['index'], s['timeframe']
                paper = s['paper_mode']
                strategy = s['strategy']
                mtf_confirm = s.get('mtf_confirm', False)

                if self.state["loop_count"] % 15 == 0:
                    self.check_fomo_alerts(index)

                cutoff_time = dt.time(15, 15) if index not in ["CRUDEOIL", "GOLD", "SILVER"] else dt.time(23, 15)
                if self.is_mock: cutoff_time = dt.time(23, 59) 
                
                spot, base_lot_size = None, 1
                df_candles = None
                
                if not self.is_mock:
                    exch, token = self.get_token_info(index)
                    spot = self.get_live_price(exch, index, token)
                    df_candles = self.get_historical_data(exch, token, symbol=index, interval=timeframe)
                    
                    if spot is None and df_candles is not None and not df_candles.empty:
                        spot = float(df_candles['close'].iloc[-1])
                            
                    base_lot_size = DEFAULT_LOTS.get(index, 25)
                else:
                    spot = self.get_live_price("NSE", index, "12345")
                    df_candles = self.get_historical_data("MOCK", "12345", symbol=index, interval=timeframe)
                    base_lot_size = DEFAULT_LOTS.get(index, 25)
                
                if spot and df_candles is not None and not df_candles.empty:
                    self.state["spot"] = spot
                    
                    if "Institutional FVG" in strategy:
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_fvg_strategy(df_candles, index)
                    elif "Combined" in strategy:
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_combined_strategy(df_candles, index)
                    else:
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_vwap_ema_strategy(df_candles, index)

                    if mtf_confirm and signal != "WAIT":
                        htf = self.get_higher_timeframe(timeframe)
                        htf_df = self.get_historical_data(exch, token, symbol=index, interval=htf)
                        if htf_df is not None and not htf_df.empty:
                            htf_ema_short = htf_df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                            htf_ema_long = htf_df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
                            
                            if signal == "BUY_CE" and htf_ema_short <= htf_ema_long:
                                signal = "WAIT"
                                trend += " (Blocked by MTF)"
                            elif signal == "BUY_PE" and htf_ema_short >= htf_ema_long:
                                signal = "WAIT"
                                trend += " (Blocked by MTF)"

                    self.state.update({"current_trend": trend, "current_signal": signal, "vwap": vwap, "ema": ema, "atr": current_atr, "fib_data": fib_data, "latest_data": df_chart})

                    current_time = dt.datetime.now().time()
                    if self.state["active_trade"] is None and signal in ["BUY_CE", "BUY_PE"] and not self.state["order_in_flight"]:
                        if current_time < cutoff_time:
                            self.state["order_in_flight"] = True 
                            try:
                                qty = s['lots'] * base_lot_size
                                max_prem = s['max_capital'] / qty if qty > 0 else 0
                                strike_sym, strike_token, strike_exch, entry_ltp = self.get_strike(index, spot, signal, max_prem)
                                
                                if strike_sym and entry_ltp:
                                    dynamic_sl = entry_ltp - (current_atr * 1.5) if (current_atr * 1.5) > s['sl_pts'] else entry_ltp - s['sl_pts']
                                    dynamic_tgt = entry_ltp + (current_atr * 3.0) if (current_atr * 3.0) > s['tgt_pts'] else entry_ltp + s['tgt_pts']

                                    trade_type = "REAL" if not paper and not self.is_mock else "PAPER"
                                    
                                    if trade_type == "REAL":
                                        order_id = self.place_real_order(strike_sym, strike_token, qty, "BUY", strike_exch)
                                        if order_id: 
                                            self.log(f"🟢 REAL ENTRY: {strike_sym} @ ₹{entry_ltp}")
                                            self.push_notify("🟢 Trade Executed", f"Bought {qty}x {strike_sym}\nEntry: ₹{entry_ltp}\nSL: ₹{round(dynamic_sl, 2)}")
                                            self.state["active_trade"] = {"symbol": strike_sym, "token": strike_token, "exch": strike_exch, "type": "CE" if "CE" in strike_sym else "PE", "entry": entry_ltp, "highest_price": entry_ltp, "qty": qty, "sl": dynamic_sl, "tgt": dynamic_tgt}
                                    else:
                                        self.log(f"🟢 PAPER ENTRY: {strike_sym} @ ₹{entry_ltp}")
                                        self.push_notify("📝 Paper Trade", f"Entered {strike_sym}\nEntry: ₹{entry_ltp}\nSL: ₹{round(dynamic_sl, 2)}")
                                        self.state["active_trade"] = {"symbol": strike_sym, "token": strike_token, "exch": strike_exch, "type": "CE" if "CE" in strike_sym else "PE", "entry": entry_ltp, "highest_price": entry_ltp, "qty": qty, "sl": dynamic_sl, "tgt": dynamic_tgt}
                            finally:
                                self.state["order_in_flight"] = False 

                    elif self.state["active_trade"]:
                        trade = self.state["active_trade"]
                        if self.is_mock: 
                            delta = (spot - self.state["spot"]) * (0.5 if trade['type'] == "CE" else -0.5) 
                            ltp = trade['entry'] + delta + np.random.uniform(-1, 2)
                        else:
                            ltp = self.get_live_price(trade['exch'], trade['symbol'], trade['token'])
                            
                        if ltp:
                            pnl = (ltp - trade['entry']) * trade['qty'] if trade['type'] == "CE" else (trade['entry'] - ltp) * trade['qty']
                            self.state["active_trade"]["current_ltp"] = ltp
                            self.state["active_trade"]["floating_pnl"] = pnl
                            
                            if ltp > trade.get('highest_price', trade['entry']):
                                trade['highest_price'] = ltp
                                new_sl = ltp - s['tsl_pts']
                                if new_sl > trade['sl']: trade['sl'] = new_sl
                            
                            if current_time >= cutoff_time or ltp >= trade['tgt'] or ltp <= trade['sl']:
                                if not paper and not self.is_mock: self.place_real_order(trade['symbol'], trade['token'], trade['qty'], "SELL", trade['exch'])
                                exit_msg = f"PnL: ₹{round(pnl, 2)} @ ₹{ltp:.2f}"
                                
                                self.log(f"🛑 EXIT {trade['symbol']} | {exit_msg}")
                                self.push_notify("🛑 Trade Closed", f"{trade['symbol']} exited.\n{exit_msg}")
                                
                                save_trade({"Date": dt.date.today().strftime('%Y-%m-%d'), "Time": dt.datetime.now().strftime('%H:%M:%S'), "Symbol": trade['symbol'], "Type": trade['type'], "Qty": trade['qty'], "Entry Price": trade['entry'], "Exit Price": ltp, "PnL (₹)": round(pnl, 2)})
                                
                                self.state["last_trade"] = trade.copy()
                                self.state["last_trade"]["exit_price"] = ltp
                                self.state["last_trade"]["final_pnl"] = pnl
                                self.state["active_trade"] = None
            except Exception as e:
                self.log(f"⚠️ Loop Error: {str(e)}")
            time.sleep(2)

# ==========================================
# 4. STREAMLIT UI 
# ==========================================
st.set_page_config(page_title="Pro Scalper Bot", page_icon="⚡", layout="wide")
is_mkt_open, mkt_status_msg = get_market_status()

# 👉 Audio Notification Function (Beep)
def play_sound():
    components.html(
        """<audio autoplay><source src="https://media.geeksforgeeks.org/wp-content/uploads/20190531135120/beep.mp3" type="audio/mpeg"></audio>""", height=0
    )

# 👉 BATCH PROCESS ALL PENDING UI POPUPS
if st.session_state.bot and st.session_state.bot.state.get("ui_popups"):
    play_sound() 
    
    js_notifications = []
    
    # Process the entire queue so we don't drop multiple alerts
    while st.session_state.bot.state["ui_popups"]:
        alert = st.session_state.bot.state["ui_popups"].popleft()
        
        # Parse title and message securely
        if isinstance(alert, dict):
            raw_title = alert.get("title", "Pro Scalper")
            raw_msg = alert.get("message", "")
        else:
            raw_title = "Pro Scalper Alert ⚡"
            raw_msg = str(alert)
            
        st.toast(raw_msg, icon="🔔")
        
        # Escape characters for JavaScript rendering
        safe_title = raw_title.replace('"', '\\"').replace("'", "\\'")
        safe_body = raw_msg.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
        
        js_notifications.append(f"""
            if (targetWindow.Notification && targetWindow.Notification.permission === "granted") {{
                new targetWindow.Notification("{safe_title}", {{ body: "{safe_body}", icon: "https://cdn-icons-png.flaticon.com/512/2952/2952865.png" }});
            }}
        """)
        
    if js_notifications:
        final_js = f"""
        <script>
        try {{
            const targetWindow = window.parent || window;
            {' '.join(js_notifications)}
        }} catch(e) {{ console.error("Chrome Notification Error: ", e); }}
        </script>
        """
        components.html(final_js, height=0)

with st.sidebar:
    st.header("🔐 Connection Setup")
    auth_mode = st.radio("Operating Mode", ["📝 Paper Trading", "⚡ Real Trading"])
    
    if not st.session_state.bot:
        if auth_mode == "⚡ Real Trading":
            creds = load_creds()
            API_KEY = st.text_input("SmartAPI Key", value=creds.get("api_key", ""), type="password")
            CLIENT_ID = st.text_input("Client ID", value=creds.get("client_id", ""))
            PIN = st.text_input("PIN", value=creds.get("pwd", ""), type="password")
            TOTP = st.text_input("TOTP secret", value=creds.get("totp_secret", ""), type="password")
            
            st.markdown("---")
            st.markdown("**📱 Telegram Alerts (Optional)**")
            TG_TOKEN = st.text_input("Bot Token", value=creds.get("tg_token", ""), type="password")
            TG_CHAT = st.text_input("Chat ID", value=creds.get("tg_chat", ""))
            
            st.markdown("---")
            st.markdown("**💬 WhatsApp Alerts (Optional)**")
            WA_PHONE = st.text_input("WhatsApp Number (incl. +91)", value=creds.get("wa_phone", ""))
            WA_API = st.text_input("CallMeBot API Key", value=creds.get("wa_api", ""), type="password")
            
            SAVE_CREDS = st.checkbox("Remember Details", value=True)
            
            if st.button("Connect to Live Exchange", type="primary"):
                temp_bot = SniperBot(API_KEY, CLIENT_ID, PIN, TOTP, TG_TOKEN, TG_CHAT, WA_PHONE, WA_API, is_mock=False)
                with st.spinner("Authenticating..."):
                    if temp_bot.login():
                        if SAVE_CREDS: save_creds(CLIENT_ID, PIN, TOTP, API_KEY, TG_TOKEN, TG_CHAT, WA_PHONE, WA_API)
                        st.session_state.bot = temp_bot
                        st.rerun()
                    else: st.error("Login Failed. Check credentials or logs.")
        else:
            if st.button("Start Paper Session", type="primary"):
                temp_bot = SniperBot(is_mock=True)
                temp_bot.login()
                st.session_state.bot = temp_bot
                st.rerun()
    else:
        st.success(f"👤 Owner: **{st.session_state.bot.client_name}**")
        if st.button("Logout & Clear"):
            st.session_state.bot.state["is_running"] = False
            st.session_state.clear()
            st.rerun()

    st.divider()

    st.markdown("**🌐 Browser Desktop Notifications**")
    if st.button("🔔 Enable Web Notifications", use_container_width=True):
        if st.session_state.bot:
            st.session_state.bot.push_notify("Notifications Active!", "Browser alerts are properly connected.")
        st.success("Please click 'Allow' if your browser prompts you at the top left.")
        components.html("""
        <script>
            const targetWindow = window.parent || window;
            if (targetWindow.Notification && targetWindow.Notification.permission !== "denied") {
                targetWindow.Notification.requestPermission();
            }
        </script>
        """, height=0)
        
    st.divider()

    # DYNAMIC ASSET LIST CONFIG
    st.header("➕ Add Custom Stock")
    CUSTOM_STOCK = st.text_input("Custom NSE/BSE Symbol (e.g. RELIANCE)", value=st.session_state.custom_stock).upper().strip()
    st.session_state.custom_stock = CUSTOM_STOCK

    asset_list = list(DEFAULT_LOTS.keys())
    if CUSTOM_STOCK and CUSTOM_STOCK not in asset_list:
        asset_list.append(CUSTOM_STOCK)
        DEFAULT_LOTS[CUSTOM_STOCK] = 1 # Fallback 1 quantity for unknown symbols

    st.session_state.asset_options = asset_list

    if st.session_state.sb_index_input not in asset_list:
        st.session_state.sb_index_input = asset_list[0]
        
    STRATEGY = st.selectbox("Trading Strategy", STRAT_LIST, index=STRAT_LIST.index(st.session_state.sb_strat_input), key="sb_strat_input")
    INDEX = st.selectbox("Watchlist", asset_list, index=asset_list.index(st.session_state.sb_index_input), key="sb_index_input")
    TIMEFRAME = st.selectbox("Timeframe", ["1m", "3m", "5m", "15m"], index=2)
    
    st.header("🛡️ Risk Management")
    LOTS = st.number_input("Base Lots", 1, 100, 1)
    MAX_CAPITAL = st.number_input("Max Capital / Trade (₹)", 1000, 500000, 15000, step=1000)
    SL_PTS = st.number_input("Fallback Stop Loss (Pts)", 5, 200, 20)
    TSL_PTS = st.number_input("Trailing Stop Loss (Pts)", 5, 200, 15)
    TGT_PTS = st.number_input("Fallback Target (Points)", 10, 500, 40)
    
    st.header("⚙️ Advanced Execution")
    PAPER = st.toggle("📝 Paper Trade Execution", True, disabled=True if (st.session_state.bot and st.session_state.bot.is_mock) else False)
    MTF_CONFIRM = st.toggle("🔍 Multi-Timeframe Confirmation", False, help="Checks the next higher timeframe to ensure trend alignment before entering.")
    HERO_ZERO = st.toggle("🚀 Enable Hero/Zero", False)
    HZ_PREMIUM = st.number_input("Max H/Z Premium (₹)", 1, 100, 15)
    
    render_signature()

if not st.session_state.bot:
    st.title("Welcome to Pro Scalper Bot ⚡")
    st.warning("Please configure your connection in the sidebar to begin.")
else:
    bot = st.session_state.bot
    bot.settings = {"strategy": STRATEGY, "index": INDEX, "timeframe": TIMEFRAME, "lots": LOTS, "max_capital": MAX_CAPITAL, "sl_pts": SL_PTS, "tsl_pts": TSL_PTS, "tgt_pts": TGT_PTS, "paper_mode": PAPER, "hero_zero": HERO_ZERO, "hz_premium": HZ_PREMIUM, "mtf_confirm": MTF_CONFIRM}

    if bot.state['latest_data'] is None or st.session_state.prev_index != INDEX:
        st.session_state.prev_index = INDEX
        with st.spinner(f"Fetching Previous Closing Data for {INDEX}..."):
            exch, token = bot.get_token_info(INDEX)
            df_preload = bot.get_historical_data(exch, token, symbol=INDEX, interval=TIMEFRAME)
            if df_preload is not None and not df_preload.empty:
                bot.state["spot"] = df_preload['close'].iloc[-1]
                if "Institutional FVG" in STRATEGY:
                    t, s, v, e, df_c, atr, fib = bot.analyzer.apply_fvg_strategy(df_preload, INDEX)
                elif "Combined" in STRATEGY:
                    t, s, v, e, df_c, atr, fib = bot.analyzer.apply_combined_strategy(df_preload, INDEX)
                else:
                    t, s, v, e, df_c, atr, fib = bot.analyzer.apply_vwap_ema_strategy(df_preload, INDEX)
                bot.state.update({"current_trend": t, "current_signal": s, "vwap": v, "ema": e, "atr": atr, "fib_data": fib, "latest_data": df_c})

    if not is_mkt_open: st.error(f"😴 {mkt_status_msg}")
        
    tab1, tab2, tab3 = st.tabs(["⚡ Live Dashboard", "🔎 Scanners", "📜 PnL Reports & Logs"])

    with tab1:
        st.subheader(f"Trading Terminal: {INDEX}")
        
        c1, c2, c3 = st.columns([1, 1, 3])
        is_running = bot.state["is_running"]
        
        with c1:
            if st.button("▶️ START ENGINE", use_container_width=True, type="primary", disabled=is_running):
                bot.state["is_running"] = True
                t = threading.Thread(target=bot.trading_loop, daemon=True)
                add_script_run_ctx(t)
                t.start()
                st.rerun()
        with c2:
            if st.button("🛑 STOP ENGINE", use_container_width=True, disabled=not is_running):
                bot.state["is_running"] = False
                st.rerun()
        with c3:
            if is_running:
                st.success("🟢 **ENGINE IS RUNNING** - Scanning Market for Entries...")
            else:
                st.error("🛑 **ENGINE STOPPED** - Click Start to begin trading.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"LTP", f"₹ {round(bot.state['spot'], 2)}")
        
        fib_text = "Wait for Calc"
        if bot.state['fib_data']:
            fib = bot.state['fib_data']
            fib_text = f"₹ {round(fib['fib_low'], 1)} - {round(fib['fib_high'], 1)}"
            
        m2.metric("Fib Golden Zone (0.5 - 0.618)", fib_text)
        
        if "SIDEWAYS" in bot.state["current_trend"]:
            m3.metric("Volatility (ATR)", f"{round(bot.state['atr'], 2)}", "Low Volatility")
            m4.error(f"Sentiment: {bot.state['current_trend']}")
        elif "MTF" in bot.state["current_trend"]:
            m3.metric("Volatility (ATR)", f"{round(bot.state['atr'], 2)}")
            m4.warning(f"Sentiment: {bot.state['current_trend']}")
        else:
            m3.metric("Volatility (ATR)", f"{round(bot.state['atr'], 2)}")
            m4.success(f"Sentiment: {bot.state['current_trend']}")
        
        chart_col, trade_col = st.columns([3, 1])

        with chart_col:
            
            c_header_col1, c_header_col2 = st.columns([3, 1])
            with c_header_col2:
                SHOW_CHART = st.toggle("📊 Enable Chart", True)
                
            if bot.state["active_trade"]:
                active_sym = bot.state["active_trade"]["symbol"]
                with c_header_col1:
                    st.markdown(f"### 🎯 Tracking Option Strike: **{active_sym}**")
                
                if SHOW_CHART:
                    opt_df = bot.get_historical_data(bot.state["active_trade"]["exch"], bot.state["active_trade"]["token"], symbol=active_sym, interval=TIMEFRAME)
                    df_to_plot = opt_df if opt_df is not None and not opt_df.empty else bot.state["latest_data"]
                else:
                    df_to_plot = None
            else:
                with c_header_col1:
                    st.markdown(f"### 📈 Tracking Underlying: **{INDEX}**")
                df_to_plot = bot.state["latest_data"]

            if SHOW_CHART and df_to_plot is not None and not df_to_plot.empty:
                chart_df = df_to_plot.copy()
                chart_df = chart_df.sort_index()
                for col in ['open', 'high', 'low', 'close']: chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
                chart_df = chart_df.dropna(subset=['open', 'high', 'low', 'close'])
                chart_df['time'] = (pd.to_datetime(chart_df.index).astype('int64') // 10**9) - 19800
                
                candles = chart_df[['time', 'open', 'high', 'low', 'close']].to_dict('records')
                
                fib_lines = []
                if not bot.state["active_trade"]:
                    fib_data = bot.state.get('fib_data', {})
                    if fib_data:
                        fib_lines = [
                            {"price": fib_data.get('major_high', 0), "color": '#ef4444', "lineWidth": 1, "lineStyle": 0, "title": 'Major Res'},
                            {"price": fib_data.get('fib_high', 0), "color": '#fbbf24', "lineWidth": 2, "lineStyle": 2, "title": 'Fib 0.5'},
                            {"price": fib_data.get('fib_low', 0), "color": '#fbbf24', "lineWidth": 2, "lineStyle": 2, "title": 'Fib 0.618'},
                            {"price": fib_data.get('major_low', 0), "color": '#22c55e', "lineWidth": 1, "lineStyle": 0, "title": 'Major Sup'}
                        ]
                
                chartOptions = {
                    "height": 450,
                    "layout": { "textColor": '#d1d5db', "background": { "type": 'solid', "color": '#0b0f19' } },
                    "grid": { "vertLines": { "color": 'rgba(42, 46, 57, 0.5)' }, "horzLines": { "color": 'rgba(42, 46, 57, 0.5)' } },
                    "crosshair": { "mode": 0 }, "timeScale": { "timeVisible": True, "secondsVisible": False }
                }
                
                chart_series = [{"type": 'Candlestick', "data": candles, "options": {"upColor": '#26a69a', "downColor": '#ef5350', "borderVisible": False, "wickUpColor": '#26a69a', "wickDownColor": '#ef5350'}, "priceLines": fib_lines}]
                
                if 'ema_short' in chart_df.columns:
                    chart_df['ema_short'] = pd.to_numeric(chart_df['ema_short'], errors='coerce')
                    ema_data = chart_df[['time', 'ema_short']].dropna().rename(columns={'ema_short': 'value'}).to_dict('records')
                    if ema_data:
                        chart_series.append({
                            "type": 'Line',
                            "data": ema_data,
                            "options": { "color": '#2962ff', "lineWidth": 2, "title": 'EMA 9' }
                        })
                        
                if 'psar' in chart_df.columns:
                    chart_df['psar'] = pd.to_numeric(chart_df['psar'], errors='coerce')
                    psar_data = chart_df[['time', 'psar']].dropna().rename(columns={'psar': 'value'}).to_dict('records')
                    if psar_data:
                        chart_series.append({
                            "type": 'Line',
                            "data": psar_data,
                            "options": { "color": '#e1bee7', "lineWidth": 1, "lineStyle": 2, "title": 'PSAR' }
                        })
                
                renderLightweightCharts([{"chart": chartOptions, "series": chart_series}], 'tv_chart')
            elif not SHOW_CHART:
                st.info("📊 Chart is disabled. Turn the toggle on to view price action.")
            else:
                st.info(f"💡 Waiting for API Data connection for **{INDEX}**. Ensure market is open or check logs.")

        with trade_col:
            st.markdown("### 🎯 Order Info")
            
            if bot.state["active_trade"]:
                t = bot.state["active_trade"]
                pnl = t.get('floating_pnl', 0.0)
                ltp = t.get('current_ltp', t['entry'])
                indicator = "🟢" if pnl >= 0 else "🔴"
                
                st.success(f"**🟢 ACTIVE TRADE**\n\n**Strike:** `{t['symbol']}`\n\n**Type:** `{t['type']}` (Qty: {t['qty']})")
                
                cA, cB = st.columns(2)
                cA.metric("Entry", f"₹{t['entry']:.2f}")
                cB.metric("LTP", f"₹{ltp:.2f}", f"{indicator} ₹{round(pnl, 2)}")
                
                st.info(f"🛑 **Stop Loss:** `₹{t['sl']:.2f}` (Dynamic)\n\n🎯 **Target:** `₹{t['tgt']:.2f}`")
                
                if st.button("Close Trade Manually", use_container_width=True):
                    if not bot.settings['paper_mode'] and not bot.is_mock: bot.place_real_order(t['symbol'], t['token'], t['qty'], "SELL", t['exch'])
                    save_trade({"Date": dt.date.today().strftime('%Y-%m-%d'), "Time": dt.datetime.now().strftime('%H:%M:%S'), "Symbol": t['symbol'], "Type": t['type'], "Qty": t['qty'], "Entry Price": t['entry'], "Exit Price": ltp, "PnL (₹)": round(pnl, 2)})
                    bot.state["last_trade"] = t.copy()
                    bot.state["last_trade"]["exit_price"] = ltp
                    bot.state["last_trade"]["final_pnl"] = pnl
                    bot.state["active_trade"] = None
                    st.rerun()
            else:
                if bot.state.get("last_trade"):
                    lt = bot.state["last_trade"]
                    indicator = "🟢" if lt['final_pnl'] >= 0 else "🔴"
                    st.warning(f"**🛑 LAST CLOSED TRADE**\n\n**Strike:** `{lt['symbol']}`")
                    cA, cB = st.columns(2)
                    cA.metric("Entry", f"₹{lt['entry']:.2f}")
                    cB.metric("Exit", f"₹{lt['exit_price']:.2f}", f"{indicator} ₹{round(lt['final_pnl'], 2)}")
                    st.markdown("*Waiting for next entry setup...*")
                else:
                    st.info("No active positions. Engine will open trades automatically based on Strategy.")

    with tab2:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("📊 52W High/Low & Intraday Scanner")
            st.write("Scans top NIFTY 50 stocks for breakouts and intraday momentum.")
            
            if st.button("🔍 Scan Top NSE Stocks"):
                with st.spinner("Analyzing Volatility and Price Action..."):
                    try:
                        watch_list = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "M&M.NS"]
                        scan_results = []
                        for ticker in watch_list:
                            tk = yf.Ticker(ticker)
                            hist_1y = tk.history(period="1y")
                            if hist_1y.empty: continue
                            
                            high_52 = hist_1y['High'].max()
                            low_52 = hist_1y['Low'].min()
                            ltp = hist_1y['Close'].iloc[-1]
                            
                            intra = tk.history(period="5d", interval="5m")
                            rating = "Neutral ⚖️"
                            
                            if not intra.empty:
                                df_intra = intra.copy()
                                df_intra['vwap'] = (df_intra['Close'] * df_intra['Volume']).cumsum() / df_intra['Volume'].cumsum()
                                df_intra['ema9'] = df_intra['Close'].ewm(span=9).mean()
                                df_intra['ema21'] = df_intra['Close'].ewm(span=21).mean()
                                
                                c = df_intra['Close'].iloc[-1]
                                v = df_intra['vwap'].iloc[-1]
                                e9 = df_intra['ema9'].iloc[-1]
                                e21 = df_intra['ema21'].iloc[-1]
                                
                                if c > e9 > e21 and c > v: rating = "Strong Buy 🚀"
                                elif c > v and c > e21: rating = "Buy 🟢"
                                elif c < e9 < e21 and c < v: rating = "Strong Sell 🩸"
                                elif c < v and c < e21: rating = "Sell 🔴"

                            scan_results.append({
                                "Stock": ticker.replace(".NS", ""),
                                "LTP": round(ltp, 2),
                                "52W High": round(high_52, 2),
                                "52W Low": round(low_52, 2),
                                "Intraday Signal": rating
                            })
                            
                        res_df = pd.DataFrame(scan_results)
                        st.dataframe(res_df, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Scanner Failed. Error: {e}")
                        
        with colB:
            st.subheader(f"📡 Multi-Stock BTST Scanner")
            if st.button("🔄 Scan Market"):
                with st.spinner("Analyzing Watchlist..."):
                    scan_results = []
                    for sym in list(DEFAULT_LOTS.keys()):
                        df = bot.get_historical_data("NSE", "99926000", symbol=sym, interval="1d" if not is_mkt_open else "5m")
                        if df is not None and not df.empty:
                            trend, sig, v, e, _, _, _ = bot.analyzer.apply_vwap_ema_strategy(df, sym)
                            btst_sig = check_btst_stbt(df) if not is_mkt_open else "N/A"
                            scan_results.append({"Symbol": sym, "LTP": round(df['close'].iloc[-1], 2), "Trend": trend, "BTST/STBT": btst_sig})
                    st.dataframe(pd.DataFrame(scan_results), use_container_width=True, hide_index=True)

    with tab3:
        log_col, pnl_col = st.columns([1, 2])
        with log_col:
            st.subheader("System Logs")
            for l in bot.state["logs"]: st.text(l)
        with pnl_col:
            st.subheader("📊 Export PnL Reports")
            if os.path.exists(TRADE_FILE):
                df_all = pd.read_csv(TRADE_FILE)
                st.dataframe(df_all.iloc[::-1], use_container_width=True)
                
                c1, c2 = st.columns(2)
                
                with c1:
                    if st.button("⚙️ Generate Excel File", use_container_width=True):
                        with st.spinner("Generating Report..."):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df_all.to_excel(writer, index=False, sheet_name='Trade_Log')
                            st.session_state['ready_excel'] = output.getvalue()
                    
                    if 'ready_excel' in st.session_state:
                        st.download_button(
                            label="📥 Download Excel (.xlsx)", 
                            data=st.session_state['ready_excel'], 
                            file_name=f"Scalper_Trade_Log_{dt.datetime.now().strftime('%Y%m%d')}.xlsx", 
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                            use_container_width=True
                        )
                        
                with c2:
                    if st.button("🗑️ Clear Logs", use_container_width=True):
                        os.remove(TRADE_FILE)
                        if 'ready_excel' in st.session_state:
                            del st.session_state['ready_excel']
                        st.rerun()
            else:
                st.info("No trade history available.")

# ==========================================
# FULL-WIDTH BOTTOM NAVIGATION DOCK 
# ==========================================

# 👉 GLOBALLY REFRESH UI FOR BACKGROUND ALERTS
if getattr(st.session_state, "bot", None) and st.session_state.bot.state.get("is_running"):
    time.sleep(2)
    st.rerun()

def cycle_asset():
    assets = st.session_state.get('asset_options', list(DEFAULT_LOTS.keys()))
    if st.session_state.sb_index_input in assets:
        current_idx = assets.index(st.session_state.sb_index_input)
        st.session_state.sb_index_input = assets[(current_idx + 1) % len(assets)]
    else:
        st.session_state.sb_index_input = assets[0]

def cycle_strat():
    current_idx = STRAT_LIST.index(st.session_state.sb_strat_input)
    st.session_state.sb_strat_input = STRAT_LIST[(current_idx + 1) % len(STRAT_LIST)]

dock_c1, dock_c2, dock_c3 = st.columns(3)

with dock_c1: 
    st.button(f"🔄 Switch: {st.session_state.sb_index_input}", key="btn_asset_switch", on_click=cycle_asset, use_container_width=True)

with dock_c2: 
    display_strat = st.session_state.sb_strat_input.split(" ")[0] 
    st.button(f"🧠 Strat: {display_strat}", key="btn_strat_switch", on_click=cycle_strat, use_container_width=True)

with dock_c3:
    if st.button("👆 Quick Login", use_container_width=True):
        if not st.session_state.bot:
            st.toast("Simulating biometric login...", icon="🔓")
            creds = load_creds()
            is_mock_login = not bool(creds.get("api_key"))
            temp_bot = SniperBot(creds.get("api_key", ""), creds.get("client_id", ""), creds.get("pwd", ""), creds.get("totp_secret", ""), creds.get("tg_token", ""), creds.get("tg_chat", ""), creds.get("wa_phone", ""), creds.get("wa_api", ""), is_mock=is_mock_login)
            if temp_bot.login():
                st.session_state.bot = temp_bot
                st.rerun()

components.html(
    """<script>
    const doc = window.parent.document;
    const blocks = doc.querySelectorAll('div[data-testid="stHorizontalBlock"]');
    if (blocks.length > 0) {
        const navBlock = blocks[blocks.length - 1]; 
        const wrapper = navBlock.closest('.element-container');
        if (wrapper) {
            wrapper.style.position = 'fixed'; wrapper.style.bottom = '0px'; wrapper.style.left = '0px'; wrapper.style.width = '100%';
            wrapper.style.zIndex = '999999'; wrapper.style.backgroundColor = 'rgba(11, 15, 25, 1)'; wrapper.style.padding = '10px 15px'; wrapper.style.borderTop = '2px solid #38bdf8';
        }
    }
    </script>""", height=0)
