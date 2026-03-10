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
        f'🚀 Algo trade<br>Developed by: Vijayakumar Suryavanshi</div>', 
        unsafe_allow_html=True
    )
    st.session_state['_dev_sig'] = "AUTH_OWNER_VIJAYAKUMAR SURYAVANSHI"

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

DEFAULT_LOTS = {"NIFTY": 25, "BANKNIFTY": 15, "FINNIFTY": 25, "SENSEX": 20, "CRUDEOIL": 100, "NATURALGAS": 1250, "GOLD": 100, "SILVER": 30, "INDIA VIX": 1}
YF_TICKERS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "SENSEX": "^BSESN", "CRUDEOIL": "CL=F", "GOLD": "GC=F", "SILVER": "SI=F", "INDIA VIX": "^INDIAVIX"}

INDEX_TOKENS = {
    "NIFTY": ("NSE", "26000"),
    "BANKNIFTY": ("NSE", "26009"),
    "FINNIFTY": ("NSE", "26037"),
    "INDIA VIX": ("NSE", "26017"),
    "SENSEX": ("BSE", "99919000")
}

# ALL IN ONE AS FIRST, ICT RESTORED
STRAT_LIST = ["All in One", "ICT", "Momentum Breakout + S&R", "Institutional FVG + SMC", "Combined Convergence"]

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

        # INSIDE BAR DETECTION
        df['inside_bar'] = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))

        # CIT ANCHORED VALUE (Anchored VWAP Daily) 
        try:
            df['date'] = df.index.date
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['vol_price'] = df['close'] * df['volume']
                df['avwap'] = df.groupby('date')['vol_price'].cumsum() / df.groupby('date')['volume'].cumsum()
            else:
                # Time-weighted average fallback for indices lacking volume
                df['avwap'] = df.groupby('date')['close'].transform(lambda x: x.expanding().mean())
        except:
            df['avwap'] = df['close'] # Ultimate fallback
            
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
        
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
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
        
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
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
            "global_alerts": deque(maxlen=5), "ui_popups": deque(maxlen=10), "loop_count": 0,
            "daily_pnl": 0.0, "trades_today": 0
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

    def get_market_data_oi(self, exchange, token):
        if self.is_mock: return np.random.randint(50000, 150000), np.random.randint(1000, 10000)
        if not self.api: return 0, 0
        try:
            params = {
                "mode": "FULL",
                "exchangeTokens": { exchange: [str(token)] }
            }
            res = self.api.marketData(params)
    
