import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import datetime as dt
import time
import requests
import pyotp
import hashlib
import uuid
import threading
import io
import yfinance as yf
from SmartApi import SmartConnect
from streamlit.runtime.scriptrunner import add_script_run_ctx
from collections import deque
from streamlit_lightweight_charts import renderLightweightCharts
from supabase import create_client, Client

# Windows 11 Native Notifications
try:
    from plyer import notification
    HAS_NOTIFY = True
except ImportError:
    HAS_NOTIFY = False

# ==========================================
# 0. DATABASE & GLOBAL HELPERS
# ==========================================
@st.cache_resource
def init_supabase() -> Client:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        return None

supabase = init_supabase()
HAS_DB = supabase is not None

def get_ist():
    """Forces strict IST timezone globally (Indian Market Time) using native Python."""
    ist_offset = dt.timezone(dt.timedelta(hours=5, minutes=30))
    return dt.datetime.now(ist_offset)

def get_client_ip():
    try:
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            return st.context.headers.get("X-Forwarded-For", st.context.headers.get("X-Real-IP", "Unknown IP"))
    except: pass
    return "Unknown IP"

def render_signature():
    st.sidebar.markdown(
        f'<div style="text-align: center; color: #ffffff; font-size: 0.8rem; font-weight: bold; border-top: 1px solid #bae6fd; padding-top: 10px; margin-top: 10px;">'
        f'🚀 Algo trade<br>Developed by: Vijayakumar Suryavanshi</div>', 
        unsafe_allow_html=True
    )
    st.session_state['_dev_sig'] = "AUTH_OWNER_VIJAYAKUMAR SURYAVANSHI"

def check_btst_stbt(df):
    if df is None or len(df) < 5: return "NO DATA"
    df = df.copy()
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    last = df.iloc[-1]
    if last['close'] > last['ema9'] > last['ema21'] and (last['high'] - last['close']) < (last['close'] - last['open']): return "🔥 BTST Suggested"
    elif last['close'] < last['ema9'] < last['ema21'] and (last['close'] - last['low']) < (last['open'] - last['close']): return "🩸 STBT Suggested"
    return "⚖️ Neutral (No Hold)"

# ==========================================
# 1. DATABASE FUNCTIONS
# ==========================================
def get_user_hash(api_key):
    if not api_key: return "guest"
    return hashlib.md5(api_key.encode()).hexdigest()[:8]

def load_creds(api_key):
    if not api_key: return {}
    if HAS_DB:
        try:
            res = supabase.table("user_credentials").select("*").eq("api_key", api_key).execute()
            if res.data: return res.data[0]
        except Exception as e:
            st.toast(f"DB Load Error: {e}")
    return {"client_id": "", "pwd": "", "totp_secret": "", "api_key": api_key, "tg_token": "", "tg_chat": "", "wa_phone": "", "wa_api": ""}

def save_creds(client_id, pwd, totp_secret, api_key, tg_token, tg_chat, wa_phone, wa_api):
    if HAS_DB:
        data = {
            "api_key": api_key, "client_id": client_id, "pwd": pwd, 
            "totp_secret": totp_secret, "tg_token": tg_token, "tg_chat": tg_chat, 
            "wa_phone": wa_phone, "wa_api": wa_api
        }
        try:
            supabase.table("user_credentials").upsert(data).execute()
        except: pass

def save_trade(api_key, trade_date, trade_time, symbol, t_type, qty, entry, exit_price, pnl, result):
    if HAS_DB and api_key and api_key != "mock_key_123":
        data = {
            "api_key": api_key, "trade_date": trade_date, "trade_time": trade_time,
            "symbol": symbol, "trade_type": t_type, "qty": qty,
            "entry_price": float(entry), "exit_price": float(exit_price),
            "pnl": float(pnl), "result": result
        }
        try:
            supabase.table("trade_logs").insert(data).execute()
        except: pass

# ==========================================
# 2. UI & CONFIG
# ==========================================
st.set_page_config(page_title="Pro Scalper Bot", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #0f111a; font-family: 'Inter', sans-serif; }
    [data-testid="stHeader"] { background-color: #0284c7 !important; border-bottom: 2px solid #0369a1; }
    [data-testid="stHeader"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #0284c7 !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > input, input[type="number"], input[type="password"], input[type="text"] {
        color: #0f111a !important; font-weight: 600 !important; background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important;
    }
    [data-testid="metric-container"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    [data-testid="metric-container"] label { color: #64748b !important; font-weight: 600 !important; }
    [data-testid="metric-container"] div { color: #0f111a !important; }
    [data-testid="stTable"] th { background-color: #f1f5f9 !important; color: #0284c7 !important; border: 1px solid #e2e8f0 !important; }
    .main .block-container { padding-bottom: 120px; }
</style>
""", unsafe_allow_html=True)

DEFAULT_LOTS = {"NIFTY": 25, "BANKNIFTY": 15, "FINNIFTY": 25, "SENSEX": 20, "CRUDEOIL": 100, "NATURALGAS": 1250, "GOLD": 100, "SILVER": 30, "INDIA VIX": 1}
YF_TICKERS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS", "SENSEX": "^BSESN", "CRUDEOIL": "CL=F", "GOLD": "GC=F", "SILVER": "SI=F", "INDIA VIX": "^INDIAVIX"}
INDEX_SYMBOLS = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank", "FINNIFTY": "Nifty Fin Service", "SENSEX": "BSE SENSEX", "INDIA VIX": "INDIA VIX"}
INDEX_TOKENS = {"NIFTY": ("NSE", "26000"), "BANKNIFTY": ("NSE", "26009"), "FINNIFTY": ("NSE", "26037"), "INDIA VIX": ("NSE", "26017"), "SENSEX": ("BSE", "99919000")}

STRAT_LIST = ["Intraday Trend Rider", "All in One", "ICT", "Momentum Breakout + S&R", "Institutional FVG + SMC"]

if 'sb_index_input' not in st.session_state: st.session_state.sb_index_input = list(DEFAULT_LOTS.keys())[0]
if 'sb_strat_input' not in st.session_state: st.session_state.sb_strat_input = STRAT_LIST[0]
if 'bot' not in st.session_state: st.session_state.bot = None
if 'prev_index' not in st.session_state: st.session_state.prev_index = "NIFTY"
if 'custom_stock' not in st.session_state: st.session_state.custom_stock = ""

def get_market_status():
    now_ist = get_ist()
    if now_ist.weekday() >= 5: return False, "Market Closed (Weekend)"
    if dt.time(9, 15) <= now_ist.time() <= dt.time(15, 30): return True, "Market Live 🟢"
    if dt.time(17, 00) <= now_ist.time() <= dt.time(23, 30): return True, "Commodity Live 🟠"
    return False, "Market Closed (After Hours)"

@st.cache_data(ttl=43200) 
def get_angel_scrip_master():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df = pd.DataFrame(requests.get(url, timeout=15).json())
        df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce') / 100 
        return df
    except Exception: return pd.DataFrame()

# ==========================================
# 3. ADVANCED TECHNICAL ANALYZER 
# ==========================================
class TechnicalAnalyzer:
    def get_atr(self, df, period=14):
        df = df.copy()
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        return df['tr'].rolling(period).mean()

    def calculate_indicators(self, df, is_index=False):
        df = df.copy()
        df['vol_sma'] = df['volume'].rolling(20).mean()
        if is_index: df['vol_spike'] = True
        else: df['vol_spike'] = df['volume'] > (df['vol_sma'] * 1.5) 
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        
        atr = self.get_atr(df, 14)
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['is_sideways'] = (df['rsi'].between(45, 55)) & (abs(df['close'] - df['ema21']) < (atr * 0.5))
        df['inside_bar'] = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))
        
        try:
            df['date'] = df.index.date
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['vol_price'] = df['close'] * df['volume']
                df['avwap'] = df.groupby('date')['vol_price'].cumsum() / df.groupby('date')['volume'].cumsum()
            else: df['avwap'] = df.groupby('date')['close'].transform(lambda x: x.expanding().mean())
        except: df['avwap'] = df['close']
        return df

    def calculate_fib_zones(self, df, lookback=100):
        major_high = df['high'].rolling(lookback).max().iloc[-1]
        major_low = df['low'].rolling(lookback).min().iloc[-1]
        diff = major_high - major_low
        return major_high, major_low, min(major_high - (diff*0.5), major_high - (diff*0.618)), max(major_high - (diff*0.5), major_high - (diff*0.618))

    def detect_order_blocks(self, df):
        df = df.copy()
        df['body'] = abs(df['close'] - df['open'])
        avg_body = df['body'].rolling(10).mean()
        strong_up = (df['close'] > df['open']) & (df['body'] > avg_body * 1.8)
        strong_down = (df['close'] < df['open']) & (df['body'] > avg_body * 1.8)
        bob_h, bob_l, beob_h, beob_l = 0.0, 0.0, 0.0, 0.0
        
        for i in range(len(df)-1, max(0, len(df)-50), -1):
            if strong_up.iloc[i] and bob_h == 0.0:
                for j in range(i-1, max(0, i-10), -1):
                    if df['close'].iloc[j] < df['open'].iloc[j]: bob_h, bob_l = df['high'].iloc[j], df['low'].iloc[j]; break
            if strong_down.iloc[i] and beob_h == 0.0:
                for j in range(i-1, max(0, i-10), -1):
                    if df['close'].iloc[j] > df['open'].iloc[j]: beob_h, beob_l = df['high'].iloc[j], df['low'].iloc[j]; break
            if bob_h != 0.0 and beob_h != 0.0: break
        return {"bob_high": bob_h, "bob_low": bob_l, "beob_high": beob_h, "beob_low": beob_l}

    def apply_all_in_one_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']
        
        min_rsi = df['rsi'].rolling(14).min()
        max_rsi = df['rsi'].rolling(14).max()
        df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-10)
        df['stoch_k'] = df['stoch_rsi'].rolling(3).mean()
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
        df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])
        
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        atr = self.get_atr(df).iloc[-1]
        df['alpha_up'] = df['close'].ewm(span=14).mean() + atr
        df['sar_bull'] = df['close'] > df['close'].shift(1).rolling(5).min()
        
        smc_blocks = self.detect_order_blocks(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high, **smc_blocks}
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        signal, trend = "WAIT", "RANGING 🟡"
        
        bullish_confluence = (
            last['close'] > last['vwap'] and 
            last['stoch_k'] > last['stoch_d'] and
            (last['fvg_bull'] or prev['fvg_bull'] or smc_blocks['bob_high'] != 0.0) and
            last['sar_bull']
        )
        
        bearish_confluence = (
            last['close'] < last['vwap'] and 
            last['stoch_k'] < last['stoch_d'] and
            (last['fvg_bear'] or prev['fvg_bear'] or smc_blocks['beob_low'] != 0.0) and
            not last['sar_bull']
        )
        
        if bullish_confluence:
            trend = "ALL-IN-ONE CONVERGENCE UPTREND 🟢"
            signal = "BUY_CE"
        elif bearish_confluence:
            trend = "ALL-IN-ONE CONVERGENCE DOWNTREND 🔴"
            signal = "BUY_PE"
            
        return trend, signal, last['vwap'], df['alpha_up'].iloc[-1], df, atr, fib_data

    def apply_trend_rider_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']
        df['ema_trend'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_fast'] = df['close'].ewm(span=13, adjust=False).mean()
        
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        smc_blocks = self.detect_order_blocks(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high, **smc_blocks}
        
        last, prev = df.iloc[-1], df.iloc[-2]
        signal, trend = "WAIT", "RANGING 🟡"
        
        if last['ema_fast'] > last['ema_trend'] and last['close'] > last['vwap']:
            trend = "STRONG UPTREND 🟢"
            if last['close'] > prev['high'] and last['rsi'] > 55:
                if (smc_blocks['bob_high'] != 0.0 and last['close'] > smc_blocks['bob_high']) or last.get('vol_spike', False):
                    signal, trend = "BUY_CE", "SMC/VOL UPTREND CONFIRMED 🚀"
                else: trend = "UPTREND (Awaiting Base/Volume) ⏳"
        elif last['ema_fast'] < last['ema_trend'] and last['close'] < last['vwap']:
            trend = "STRONG DOWNTREND 🔴"
            if last['close'] < prev['low'] and last['rsi'] < 45:
                if (smc_blocks['beob_low'] != 0.0 and last['close'] < smc_blocks['beob_low']) or last.get('vol_spike', False):
                    signal, trend = "BUY_PE", "SMC/VOL DOWNTREND CONFIRMED 🩸"
                else: trend = "DOWNTREND (Awaiting Base/Volume) ⏳"
        return trend, signal, last['vwap'], last['ema_fast'], df, atr, fib_data

    def apply_vwap_ema_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high}
        last = df.iloc[-1]
        signal, trend = "WAIT", "FLAT"
        if not last['is_sideways']:
            benchmark = last['ema_long'] if is_index else last['vwap']
            if (last['ema_short'] > last['ema_long']) and (last['close'] > benchmark): trend, signal = "BULLISH MOMENTUM 🟢", "BUY_CE"
            elif (last['ema_short'] < last['ema_long']) and (last['close'] < benchmark): trend, signal = "BEARISH MOMENTUM 🔴", "BUY_PE"
        return trend, signal, last['vwap'], last['ema_short'], df, atr, fib_data

# ==========================================
# 4. CORE BOT ENGINE
# ==========================================
class SniperBot:
    def __init__(self, api_key="", client_id="", pwd="", totp_secret="", tg_token="", tg_chat="", wa_phone="", wa_api="", is_mock=False):
        self.api_key, self.client_id, self.pwd, self.totp_secret = api_key, client_id, pwd, totp_secret
        self.tg_token, self.tg_chat, self.wa_phone, self.wa_api = tg_token, tg_chat, wa_phone, wa_api
        self.api, self.token_map, self.is_mock = None, None, is_mock
        self.client_name = "Offline User"
        self.client_ip = get_client_ip()
        self.user_hash = get_user_hash(self.api_key)
        self.analyzer = TechnicalAnalyzer()
        
        self.state = {
            "is_running": False, "order_in_flight": False, "active_trade": None, "last_trade": None,
            "logs": deque(maxlen=50), "current_trend": "WAIT", "current_signal": "WAIT",
            "spot": 0.0, "vwap": 0.0, "ema": 0.0, "atr": 0.0, "fib_data": {}, "latest_data": None,
            "ui_popups": deque(maxlen=10), "loop_count": 0, "daily_pnl": 0.0, "trades_today": 0
        }
        self.settings = {}

    def push_notify(self, title, message):
        self.state["ui_popups"].append({"title": title, "message": message})
        if HAS_NOTIFY:
            try: notification.notify(title=title, message=message, app_name="Pro Scalper", timeout=5)
            except: pass

    def log(self, msg):
        self.state["logs"].appendleft(f"[{get_ist().strftime('%H:%M:%S')}] {msg}")

    def login(self):
        if self.is_mock: 
            self.client_name, self.api_key = "Paper Trading User", "mock_key_123"
            self.user_hash = get_user_hash(self.api_key)
            self.push_notify("🟢 Session Started", f"Paper Trading from IP: {self.client_ip}")
            return True
        try:
            obj = SmartConnect(api_key=self.api_key)
            res = obj.generateSession(self.client_id, self.pwd, pyotp.TOTP(self.totp_secret).now())
            if res['status']:
                self.api = obj
                self.client_name = res['data'].get('name', self.client_id)
                self.log(f"✅ Exchange Connected | IP: {self.client_ip}")
                self.push_notify("🟢 Exchange Connected", f"Live session started for user: {self.client_name}")
                return True
            self.log(f"❌ Login failed: {res.get('message', 'Check credentials')}")
            return False
        except Exception as e: 
            self.log(f"❌ Login Exception: {e}")
            return False

    def get_master(self):
        if self.token_map is None or self.token_map.empty: self.token_map = get_angel_scrip_master()
        return self.token_map

    def get_token_info(self, index_name):
        if index_name in INDEX_TOKENS: return INDEX_TOKENS[index_name]
        df_map = self.get_master()
        if df_map is not None and not df_map.empty:
            # FIX: Strip timezone awareness to avoid pandas comparison errors
            today_date = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
            futs = df_map[(df_map['name'] == index_name) & (df_map['instrumenttype'].isin(['FUTCOM', 'FUTIDX', 'FUTSTK', 'EQ']))]
            if not futs.empty:
                eqs = futs[futs['instrumenttype'] == 'EQ']
                if not eqs.empty: return eqs.iloc[0]['exch_seg'], eqs.iloc[0]['token']
                futs = futs[futs['expiry'] >= today_date]
                if not futs.empty: return futs[futs['expiry'] == futs['expiry'].min()].iloc[0]['exch_seg'], futs[futs['expiry'] == futs['expiry'].min()].iloc[0]['token']
        return "NSE", "12345"

    def get_market_data_oi(self, exchange, token):
        if self.is_mock: return np.random.randint(50000, 150000), np.random.randint(1000, 10000)
        if not self.api: return 0, 0
        try:
            res = self.api.marketData({"mode": "FULL", "exchangeTokens": { exchange: [str(token)] }})
            if res and res.get('status') and res.get('data'): return res['data']['fetched'][0].get('opnInterest', 0), res['data']['fetched'][0].get('totMacVal', 0)
        except: pass
        return 0, 0

    def get_live_price(self, exchange, symbol, token):
        if self.is_mock: 
            base_prices = {"NIFTY": 22000, "BANKNIFTY": 47000, "SENSEX": 73000}
            base = base_prices.get(symbol, 500)
            return float(np.random.uniform(base - 10, base + 10))
            
        if not self.api: return None
        try:
            trading_symbol = INDEX_SYMBOLS.get(symbol, symbol)
            res = self.api.ltpData(exchange, trading_symbol, str(token))
            if res and res.get('status'): return float(res['data']['ltp'])
        except: pass
        return None

    def get_historical_data(self, exchange, token, symbol="NIFTY", interval="5m"):
        if self.is_mock: return self._fallback_yfinance(symbol, interval)
        if not self.api: return None
        try:
            interval_map = {"1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE"}
            api_interval = interval_map.get(interval, "FIVE_MINUTE")
            now_ist = get_ist()
            fromdate = now_ist - dt.timedelta(days=10) 
            res = self.api.getCandleData({"exchange": exchange, "symboltoken": str(token), "interval": api_interval, "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"), "todate": now_ist.strftime("%Y-%m-%d %H:%M")})
            if res and res.get('status') and res.get('data'):
                df = pd.DataFrame(res['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                if df.empty: return self._fallback_yfinance(symbol, interval)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.index = df['timestamp']
                return df
            return self._fallback_yfinance(symbol, interval)
        except: return self._fallback_yfinance(symbol, interval)
            
    def _fallback_yfinance(self, symbol, interval):
        yf_int = interval if interval in ["1m", "5m", "15m"] else "5m" 
        yf_ticker = YF_TICKERS.get(symbol)
        if yf_ticker:
            try:
                df = yf.Ticker(yf_ticker).history(period="5d" if interval == "1m" else "10d", interval=yf_int)
                if not df.empty:
                    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                    return df
            except: pass
        times = [get_ist() - dt.timedelta(minutes=5*i) for i in range(150)][::-1]
        base = 22000
        close_prices = base + np.random.normal(0, 5, 150).cumsum()
        df = pd.DataFrame({'timestamp': times, 'open': close_prices - 2, 'high': close_prices + 5, 'low': close_prices - 5, 'close': close_prices, 'volume': np.random.randint(1000, 50000, 150)})
        df.index = df['timestamp']
        return df

    def place_real_order(self, symbol, token, qty, side="BUY", exchange="NFO"):
        if self.is_mock: return "MOCK_" + uuid.uuid4().hex[:6].upper()
        try: return self.api.placeOrder({"variety": "NORMAL", "tradingsymbol": symbol, "symboltoken": str(token), "transactiontype": side, "exchange": exchange, "ordertype": "MARKET", "producttype": "INTRADAY", "duration": "DAY", "quantity": str(qty)})
        except Exception as e: 
            self.log(f"Order failed: {e}")
            return None

    def get_strike(self, symbol, spot, signal, max_premium):
        opt_type = "CE" if "BUY_CE" in signal else "PE"
        if self.is_mock: return f"{symbol}28FEB{int(spot)}{opt_type}", "12345", "NFO", min(100.0, max_premium) 
        
        df = self.get_master()
        if df is None or df.empty: return None, None, None, 0.0
        
        # FIX: Strip timezone awareness to avoid pandas comparison errors
        today = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
        
        mask = (df['name'] == symbol) & (df['exch_seg'].isin(["NFO", "MCX", "BFO"])) & (df['expiry'] >= today) & (df['symbol'].str.endswith(opt_type))
        subset = df[mask].copy()
        if subset.empty: return None, None, None, 0.0
        closest_expiry = subset['expiry'].min()
        subset = subset[subset['expiry'] == closest_expiry]
        subset['dist_to_spot'] = abs(subset['strike'] - spot)
        candidates = subset.sort_values('dist_to_spot', ascending=True).head(10)
        
        for _, row in candidates.iterrows():
            ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
            if ltp and ltp <= max_premium: return row['symbol'], row['token'], row['exch_seg'], ltp
            
        return None, None, None, 0.0

    def trading_loop(self):
        self.log("▶️ Engine thread started.")
        while self.state["is_running"]:
            try:
                s = self.settings
                current_time = get_ist().time()
                today_date = get_ist().strftime('%Y-%m-%d')
                time_str = get_ist().strftime('%H:%M:%S')
                self.state["loop_count"] = self.state.get("loop_count", 0) + 1
                
                if self.state["trades_today"] >= s['max_trades'] or self.state.get("daily_pnl", 0.0) <= -s.get("capital_protect", 999999):
                    self.state["is_running"] = False
                    break

                is_open, mkt_msg = get_market_status()
                if not is_open:
                    time.sleep(10)
                    continue

                index, timeframe, paper, strategy = s['index'], s['timeframe'], s['paper_mode'], s['strategy']
                cutoff_time = dt.time(15, 15) if index not in ["CRUDEOIL", "GOLD", "SILVER"] else dt.time(23, 15)
                if self.is_mock: cutoff_time = dt.time(23, 59) 
                
                exch, token = self.get_token_info(index)
                spot = self.get_live_price(exch, index, token)
                if spot is None and self.is_mock: spot = self.get_live_price("NSE", index, "12345")
                
                df_candles = self.get_historical_data(exch, token, symbol=index, interval=timeframe) if not self.is_mock else self.get_historical_data("MOCK", "12345", symbol=index, interval=timeframe)
                
                user_lots_dict = s.get('user_lots', DEFAULT_LOTS)
                base_lot_size = user_lots_dict.get(index, 25)
                
                if spot and df_candles is not None and not df_candles.empty:
                    self.state["spot"] = spot
                    last_candle = df_candles.iloc[-1]
                    
                    if "All in One" in strategy: trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_all_in_one_strategy(df_candles, index)
                    elif "Trend Rider" in strategy: trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_trend_rider_strategy(df_candles, index)
                    else: trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_vwap_ema_strategy(df_candles, index)

                    if s.get("fomo_entry"):
                        body = abs(last_candle['close'] - last_candle['open'])
                        avg_body = df_candles['close'].diff().abs().rolling(14).mean().iloc[-1]
                        if body > (avg_body * 3) and last_candle.get('vol_spike', False):
                            signal = "BUY_CE" if last_candle['close'] > last_candle['open'] else "BUY_PE"
                            trend = "🚨 FOMO BREAKOUT ACTIVE"

                    if s.get("hero_zero") and signal != "WAIT" and not self.is_mock:
                        live_oi, live_vol = self.get_market_data_oi(exch, token)
                        if live_vol < 50000: signal, trend = "WAIT", "Hero/Zero Blocked: Low Volume/OI"

                    self.state.update({"current_trend": trend, "current_signal": signal, "vwap": vwap, "ema": ema, "atr": current_atr, "fib_data": fib_data, "latest_data": df_chart})

                    if self.state["active_trade"] is None and signal in ["BUY_CE", "BUY_PE"] and current_time < cutoff_time:
                        qty = s['lots'] * base_lot_size
                        max_prem = s['max_capital'] / qty if qty > 0 else 0
                        strike_sym, strike_token, strike_exch, entry_ltp = self.get_strike(index, spot, signal, max_prem)
                        
                        if strike_sym and entry_ltp:
                            dynamic_sl = entry_ltp - s['sl_pts'] 
                            tp1 = entry_ltp + s['tgt_pts']
                            tp2 = entry_ltp + (s['tgt_pts'] * 2)
                            tp3 = entry_ltp + (s['tgt_pts'] * 3)
                            
                            new_trade = {
                                "symbol": strike_sym, "token": strike_token, "exch": strike_exch, 
                                "type": "CE" if "CE" in strike_sym else "PE", "entry": entry_ltp, 
                                "highest_price": entry_ltp, "qty": qty, "sl": dynamic_sl, 
                                "tp1": tp1, "tp2": tp2, "tp3": tp3, "tgt": tp3
                            }
                            if not paper and not self.is_mock: self.place_real_order(strike_sym, strike_token, qty, "BUY", strike_exch)
                            self.log(f"🟢 ENTRY: {strike_sym} @ ₹{entry_ltp}")
                            self.push_notify("Trade Entered", f"Bought {qty} {strike_sym} @ {entry_ltp}")
                            self.state["active_trade"] = new_trade
                            self.state["trades_today"] += 1

                    elif self.state["active_trade"]:
                        trade = self.state["active_trade"]
                        
                        if not self.is_mock: ltp = self.get_live_price(trade['exch'], trade['symbol'], trade['token'])
                        else:
                            delta = (spot - self.state["spot"]) * (0.5 if trade['type'] == "CE" else -0.5) 
                            ltp = trade['entry'] + delta + np.random.uniform(-1, 2)
                            
                        if ltp:
                            pnl = (ltp - trade['entry']) * trade['qty'] if trade['type'] == "CE" else (trade['entry'] - ltp) * trade['qty']
                            self.state["active_trade"]["current_ltp"] = ltp
                            self.state["active_trade"]["floating_pnl"] = pnl
                            
                            if ltp > trade.get('highest_price', trade['entry']):
                                trade['highest_price'] = ltp
                                tsl_buffer = s['tsl_pts'] * 1.5 if "Trend Rider" in strategy else s['tsl_pts']
                                new_sl = ltp - tsl_buffer
                                if new_sl > trade['sl']: trade['sl'] = new_sl
                            
                            hit_tp = False if "Trend Rider" in strategy else (ltp >= trade['tgt'])
                            hit_sl = ltp <= trade['sl']
                            market_close = current_time >= cutoff_time
                            
                            if hit_tp or hit_sl or market_close:
                                if not paper and not self.is_mock: self.place_real_order(trade['symbol'], trade['token'], trade['qty'], "SELL", trade['exch'])
                                
                                highest_reached = trade['highest_price']
                                if highest_reached >= trade['tp3']: win_text = "tp3❤"
                                elif highest_reached >= trade['tp2']: win_text = "tp2✔✔"
                                elif highest_reached >= trade['tp1']: win_text = "tp1✔"
                                elif pnl > 0: win_text = "profit👍"
                                else: win_text = "sl hit 🛑"
                                
                                if market_close: win_text += " (Auto Sq-off)"
                                
                                self.log(f"🛑 EXIT {trade['symbol']} | PnL: ₹{round(pnl, 2)} [{win_text}]")
                                self.push_notify("Trade Closed", f"Closed {trade['symbol']} | PnL: ₹{round(pnl, 2)}")
                                
                                # ROUTE DATA BASED ON MODE
                                if not self.is_mock:
                                    save_trade(self.api_key, today_date, time_str, trade['symbol'], trade['type'], trade['qty'], trade['entry'], ltp, round(pnl, 2), win_text)
                                else:
                                    if "paper_history" not in self.state:
                                        self.state["paper_history"] = []
                                    self.state["paper_history"].append({
                                        "Date": today_date, "Time": time_str, "Symbol": trade['symbol'],
                                        "Type": trade['type'], "Qty": trade['qty'], "Entry Price": trade['entry'],
                                        "Exit Price": ltp, "PnL (₹)": round(pnl, 2), "Result": win_text
                                    })
                                
                                self.state["last_trade"] = trade.copy()
                                self.state["last_trade"].update({"exit_price": ltp, "final_pnl": pnl, "win_text": win_text})
                                self.state["daily_pnl"] += pnl
                                self.state["active_trade"] = None
            except Exception as e:
                self.log(f"⚠️ Loop Error: {str(e)}")
            time.sleep(2)

# ==========================================
# 5. STREAMLIT UI (MOBILE/TOP-BAR STYLE)
# ==========================================
is_mkt_open, mkt_status_msg = get_market_status()

def play_sound():
    components.html("""<audio autoplay><source src="https://media.geeksforgeeks.org/wp-content/uploads/20190531135120/beep.mp3" type="audio/mpeg"></audio>""", height=0)

if st.session_state.bot and st.session_state.bot.state.get("ui_popups"):
    play_sound()
    while st.session_state.bot.state["ui_popups"]:
        alert = st.session_state.bot.state["ui_popups"].popleft()
        st.toast(alert.get("message", ""), icon="🔔")

if not st.session_state.bot:
    if not HAS_DB:
        st.error("⚠️ Database connection missing. Add SUPABASE_URL and SUPABASE_KEY to Streamlit Secrets to enable login saving & PnL logs.")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    spacer1, login_col, spacer2 = st.columns([1, 1.5, 1])
    
    with login_col:
        st.markdown("""
            <div style='text-align: center; background: linear-gradient(135deg, #0b1120, #0284c7); padding: 20px; border-radius: 12px 12px 0 0; border: 1px solid #0284c7; border-bottom: none;'>
                <h1 style='color: white; margin:0; font-weight: 800; letter-spacing: 2px;'>⚡ PRO SCALPER</h1>
                <p style='color: #bae6fd; margin:0; font-size: 0.9rem;'>SECURE CLOUD GATEWAY</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            auth_mode = st.radio("Operating Mode", ["📝 Paper Trading", "⚡ Real Trading"], horizontal=True, label_visibility="collapsed")
            st.divider()
            
            if auth_mode == "⚡ Real Trading":
                API_KEY = st.text_input("SmartAPI Key (Required)", type="password", placeholder="Enter Angel One API Key")
                if API_KEY:
                    creds = load_creds(API_KEY)
                    col_id, col_pin = st.columns(2)
                    with col_id: CLIENT_ID = st.text_input("Client ID", value=creds.get("client_id", ""))
                    with col_pin: PIN = st.text_input("PIN", value=creds.get("pwd", ""), type="password")
                    
                    TOTP = st.text_input("TOTP Secret", value=creds.get("totp_secret", ""), type="password")
                    SAVE_CREDS = st.checkbox("Remember Credentials Securely (Cloud DB)", value=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("CONNECT LIVE EXCHANGE 🚀", type="primary", use_container_width=True):
                        temp_bot = SniperBot(API_KEY, CLIENT_ID, PIN, TOTP, is_mock=False)
                        with st.spinner("Authenticating Secure Connection..."):
                            if temp_bot.login():
                                if SAVE_CREDS: save_creds(CLIENT_ID, PIN, TOTP, API_KEY, "", "", "", "")
                                st.session_state.bot = temp_bot
                                st.rerun()
            else:
                st.info("Paper Trading simulates live market movement without risking real capital.")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("START PAPER SESSION 📝", type="primary", use_container_width=True):
                    temp_bot = SniperBot(is_mock=True)
                    temp_bot.login()
                    st.session_state.bot = temp_bot
                    st.rerun()
        
        render_signature()

else:
    bot = st.session_state.bot
    
    head_c1, head_c2 = st.columns([3, 1])
    with head_c1: st.markdown(f"**👤 Owner:** `{bot.client_name}` | **🔒 Session:** `{bot.user_hash}`")
    with head_c2:
        if st.button("Logout & Terminate", use_container_width=True):
            bot.state["is_running"] = False
            st.session_state.clear()
            st.rerun()

    # --- SIDEBAR SETTINGS ---
    with st.sidebar:
        st.header("⚙️ SYSTEM CONFIGURATION")
        
        st.markdown("**1. Market Setup**")
        if 'user_lots' not in st.session_state:
            st.session_state.user_lots = DEFAULT_LOTS.copy()
        
        CUSTOM_STOCK = st.text_input("Add Custom Stock (NSE/BSE)", value=st.session_state.custom_stock, placeholder="e.g. RELIANCE").upper().strip()
        st.session_state.custom_stock = CUSTOM_STOCK
        
        asset_list = list(st.session_state.user_lots.keys())
        if CUSTOM_STOCK and CUSTOM_STOCK not in asset_list:
            asset_list.append(CUSTOM_STOCK)
            st.session_state.user_lots[CUSTOM_STOCK] = 1 
        
        st.session_state.asset_options = asset_list
        if st.session_state.sb_index_input not in asset_list: 
            st.session_state.sb_index_input = asset_list[0]
        
        INDEX = st.selectbox("Watchlist Asset", asset_list, index=asset_list.index(st.session_state.sb_index_input), key="sb_index_input")
        STRATEGY = st.selectbox("Trading Strategy", STRAT_LIST, index=STRAT_LIST.index(st.session_state.sb_strat_input), key="sb_strat_input")
        TIMEFRAME = st.selectbox("Candle Timeframe", ["1m", "3m", "5m", "15m"], index=2)
        
        st.divider()
        st.markdown("**2. Risk Management**")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            LOTS = st.number_input("Base Lots", 1, 100, 1)
            MAX_TRADES = st.number_input("Max Trades/Day", 1, 50, 5)
            MAX_CAPITAL = st.number_input("Max Cap/Trade (₹)", 1000, 500000, 15000, step=1000)
        with col_r2:
            SL_PTS = st.number_input("SL Points", 5, 200, 20)
            TSL_PTS = st.number_input("Trail SL Pts", 5, 200, 15)
            TGT_PTS = st.number_input("Target Steps", 5, 500, 15)
        CAPITAL_PROTECT = st.number_input("Capital Protection (Max Loss/Day ₹)", 500, 500000, 2000, step=500)
        
        st.divider()
        st.markdown("**3. Advanced Triggers**")
        PAPER = st.toggle("📝 Paper Trade Mode", True, disabled=True if bot.is_mock else False)
        HERO_ZERO = st.toggle("🚀 Hero/Zero Setup", False)
        FOMO_ENTRY = st.toggle("🚨 FOMO Momentum Entry", False)
        
        render_signature()

    bot.settings = {"strategy": STRATEGY, "index": INDEX, "timeframe": TIMEFRAME, "lots": LOTS, "max_trades": MAX_TRADES, "max_capital": MAX_CAPITAL, "capital_protect": CAPITAL_PROTECT, "sl_pts": SL_PTS, "tsl_pts": TSL_PTS, "tgt_pts": TGT_PTS, "paper_mode": PAPER, "hero_zero": HERO_ZERO, "fomo_entry": FOMO_ENTRY, "user_lots": st.session_state.user_lots.copy()}

    if bot.state['latest_data'] is None or st.session_state.prev_index != INDEX:
        st.session_state.prev_index = INDEX
        if bot.state.get("is_running"): bot.state["spot"] = 0.0 
        else:
            with st.spinner(f"Fetching Live Market Data for {INDEX}..."):
                exch, token = bot.get_token_info(INDEX)
                df_preload = bot.get_historical_data(exch, token, symbol=INDEX, interval=TIMEFRAME) if not bot.is_mock else bot.get_historical_data("MOCK", "12345", symbol=INDEX, interval=TIMEFRAME)
                if df_preload is not None and not df_preload.empty:
                    bot.state["spot"] = df_preload['close'].iloc[-1]
                    if "All in One" in STRATEGY: t, s, v, e, df_c, atr, fib = bot.analyzer.apply_all_in_one_strategy(df_preload, INDEX)
                    elif "Trend Rider" in STRATEGY: t, s, v, e, df_c, atr, fib = bot.analyzer.apply_trend_rider_strategy(df_preload, INDEX)
                    else: t, s, v, e, df_c, atr, fib = bot.analyzer.apply_vwap_ema_strategy(df_preload, INDEX)
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
            if is_running: st.success(f"🟢 **ENGINE IS RUNNING** (Trades Today: {bot.state['trades_today']}/{MAX_TRADES})")
            else: st.error("🛑 **ENGINE STOPPED**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"LTP", f"₹ {round(bot.state['spot'], 2)}")
        m2.metric("SMC / Fib Filter", "Active" if bot.state['fib_data'] else "Wait")
        m3.metric("Volatility (ATR)", f"{round(bot.state['atr'], 2)}")
        m4.info(f"Sentiment: {bot.state['current_trend']}")
        
        chart_col, trade_col = st.columns([3, 1])

        with chart_col:
            c_header_col1, c_header_col2, c_header_col3 = st.columns([2, 1, 1])
            with c_header_col2: SHOW_CHART = st.toggle("📊 Enable Chart", True)
            with c_header_col3: FULL_CHART = st.toggle("⛶ Large Mode", False)
            
            if SHOW_CHART and bot.state["latest_data"] is not None:
                chart_df = bot.state["latest_data"].copy()
                chart_df['time'] = (pd.to_datetime(chart_df.index).astype('int64') // 10**9) - 19800
                candles = chart_df[['time', 'open', 'high', 'low', 'close']].to_dict('records')
                
                fib_lines = []
                if not bot.state["active_trade"] and bot.state.get('fib_data'):
                    fib = bot.state['fib_data']
                    fib_lines = [
                        {"price": fib.get('major_high', 0), "color": '#ef4444', "lineWidth": 1, "lineStyle": 0, "title": 'Major Res'},
                        {"price": fib.get('fib_high', 0), "color": '#fbbf24', "lineWidth": 2, "lineStyle": 2, "title": 'Fib 0.5'},
                        {"price": fib.get('fib_low', 0), "color": '#fbbf24', "lineWidth": 2, "lineStyle": 2, "title": 'Fib 0.618'},
                        {"price": fib.get('major_low', 0), "color": '#22c55e', "lineWidth": 1, "lineStyle": 0, "title": 'Major Sup'}
                    ]
                
                chartOptions = {
                    "height": 800 if FULL_CHART else 450,
                    "layout": { "textColor": '#1e293b', "background": { "type": 'solid', "color": '#ffffff' } },
                    "grid": { "vertLines": { "color": 'rgba(226, 232, 240, 0.8)' }, "horzLines": { "color": 'rgba(226, 232, 240, 0.8)' } },
                    "crosshair": { "mode": 0 }, "timeScale": { "timeVisible": True, "secondsVisible": False }
                }
                
                chart_series = [{"type": 'Candlestick', "data": candles, "options": {"upColor": '#26a69a', "downColor": '#ef5350'}, "priceLines": fib_lines}]

                if 'avwap' in chart_df.columns:
                    avwap_data = chart_df[['time', 'avwap']].dropna().rename(columns={'avwap': 'value'}).to_dict('records')
                    if avwap_data: chart_series.append({"type": 'Line', "data": avwap_data, "options": { "color": '#9c27b0', "lineWidth": 2, "title": 'ICT AVWAP' }})

                if 'vwap' in chart_df.columns:
                    vwap_data = chart_df[['time', 'vwap']].dropna().rename(columns={'vwap': 'value'}).to_dict('records')
                    if vwap_data: chart_series.append({"type": 'Line', "data": vwap_data, "options": { "color": '#ff9800', "lineWidth": 2, "title": 'VWAP' }})

                ema_col = 'ema_fast' if 'ema_fast' in chart_df.columns else 'ema_short'
                if ema_col in chart_df.columns:
                    ema_data = chart_df[['time', ema_col]].dropna().rename(columns={ema_col: 'value'}).to_dict('records')
                    if ema_data: chart_series.append({"type": 'Line', "data": ema_data, "options": { "color": '#0ea5e9', "lineWidth": 2, "title": 'EMA' }})

                renderLightweightCharts([{"chart": chartOptions, "series": chart_series}], key="static_tv_chart")

        with trade_col:
            st.markdown("### 🎯 Order Info")
            daily_pnl = bot.state.get("daily_pnl", 0.0)
            st.markdown(f"**Daily PnL:** {'🟢' if daily_pnl >= 0 else '🔴'} ₹{round(daily_pnl, 2)}")
            
            if bot.state["active_trade"]:
                t = bot.state["active_trade"]
                ltp = t.get('current_ltp', t['entry'])
                pnl = t.get('floating_pnl', 0.0)
                indicator = "🟢" if pnl >= 0 else "🔴"
                
                st.success(f"**🟢 ACTIVE**\n`{t['symbol']}`\nQty: {t['qty']}")
                
                cA, cB = st.columns(2)
                cA.metric("Entry", f"₹{t['entry']:.2f}")
                cB.metric("LTP", f"₹{ltp:.2f}", f"{indicator} ₹{round(pnl, 2)}")
                
                st.info(f"🛑 **SL (Trailing):** `₹{t['sl']:.2f}`\n\n🎯 **TP1:** `₹{t.get('tp1', 0):.2f}`\n\n🎯 **TP2:** `₹{t.get('tp2', 0):.2f}`\n\n🎯 **TP3:** `₹{t.get('tp3', 0):.2f}`")
            else: st.info("Waiting for entry setup...")

    with tab2:
        colA, colB, colC = st.columns(3)
        with colA:
            st.subheader("📊 52W High/Low & Intraday")
            if st.button("🔍 Scan Top NSE Stocks"):
                with st.spinner("Analyzing Volatility..."):
                    try:
                        watch_list = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]
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
                                c, v, e9, e21 = df_intra['Close'].iloc[-1], df_intra['vwap'].iloc[-1], df_intra['ema9'].iloc[-1], df_intra['ema21'].iloc[-1]
                                if c > e9 > e21 and c > v: rating = "Strong Buy 🚀"
                                elif c > v and c > e21: rating = "Buy 🟢"
                                elif c < e9 < e21 and c < v: rating = "Strong Sell 🩸"
                                elif c < v and c < e21: rating = "Sell 🔴"
                            scan_results.append({"Stock": ticker.replace(".NS", ""), "LTP": round(ltp, 2), "52W High": round(high_52, 2), "52W Low": round(low_52, 2), "Signal": rating})
                        st.dataframe(pd.DataFrame(scan_results), use_container_width=True, hide_index=True)
                    except: st.error("Scanner failed.")
                        
        with colB:
            st.subheader(f"📡 Multi-Stock & Inside Bar")
            if st.button("🔄 Scan Market"):
                with st.spinner("Analyzing Watchlist..."):
                    scan_results = []
                    for sym in list(DEFAULT_LOTS.keys()):
                        df = bot.get_historical_data("NSE", "99926000", symbol=sym, interval="1d" if not is_mkt_open else "5m")
                        if df is not None and len(df) > 2:
                            inside_bar = "Yes 🟡" if (df['high'].iloc[-1] <= df['high'].iloc[-2] and df['low'].iloc[-1] >= df['low'].iloc[-2]) else "No"
                            scan_results.append({"Symbol": sym, "LTP": round(df['close'].iloc[-1], 2), "Inside Bar": inside_bar, "BTST/STBT": check_btst_stbt(df) if not is_mkt_open else "N/A"})
                    st.dataframe(pd.DataFrame(scan_results), use_container_width=True, hide_index=True)

        with colC:
            st.subheader("🪙 Low Cap / Penny Stocks")
            if st.button("🚀 Scan Penny Stocks"):
                with st.spinner("Scanning penny stocks..."):
                    penny_list = ["SUZLON.NS", "YESBANK.NS", "IDEA.NS", "JPPOWER.NS", "RPOWER.NS", "GTLINFRA.NS"]
                    p_results = []
                    for pt in penny_list:
                        try:
                            ptk = yf.Ticker(pt)
                            phist = ptk.history(period="5d", interval="5m")
                            if phist.empty: continue
                            phist['vwap'] = (phist['Close'] * phist['Volume']).cumsum() / phist['Volume'].cumsum()
                            c, v, e9 = phist['Close'].iloc[-1], phist['vwap'].iloc[-1], phist['Close'].ewm(span=9).mean().iloc[-1]
                            sig = "Bullish 🟢" if (c > v and c > e9) else ("Bearish 🔴" if (c < v and c < e9) else "Neutral ⚖️")
                            p_results.append({"Stock": pt.replace(".NS", ""), "LTP": round(c, 2), "Signal": sig})
                        except: pass
                    st.dataframe(pd.DataFrame(p_results), use_container_width=True, hide_index=True)

    with tab3:
        log_col, pnl_col = st.columns([1, 2])
        with log_col:
            st.subheader("System Logs")
            for l in bot.state["logs"]: st.text(l)
            
        with pnl_col:
            if bot.is_mock:
                st.subheader("📊 Paper Trade History (Session Memory)")
                if bot.state.get("paper_history"):
                    df_paper = pd.DataFrame(bot.state["paper_history"])
                    st.dataframe(df_paper.iloc[::-1], use_container_width=True)
                else:
                    st.info("No paper trades recorded yet in this session.")
            else:
                st.subheader("📊 Live Trade History (Cloud DB)")
                if HAS_DB:
                    try:
                        res = supabase.table("trade_logs").select("*").eq("api_key", bot.api_key).execute()
                        if res.data:
                            df_db = pd.DataFrame(res.data)
                            df_db = df_db.drop(columns=["id", "api_key"], errors="ignore")
                            st.dataframe(df_db.iloc[::-1], use_container_width=True)
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df_db.to_excel(writer, index=False)
                            st.download_button("📥 Download Excel (.xlsx)", data=output.getvalue(), file_name=f"Cloud_Trade_Log.xlsx")
                        else: st.info("No real trades recorded yet.")
                    except Exception as e: st.error(f"Could not load trades: {e}")
                else:
                    st.error("Cloud DB not connected.")

def cycle_asset():
    assets = st.session_state.get('asset_options', list(DEFAULT_LOTS.keys()))
    if st.session_state.sb_index_input in assets: st.session_state.sb_index_input = assets[(assets.index(st.session_state.sb_index_input) + 1) % len(assets)]
    else: st.session_state.sb_index_input = assets[0]

def cycle_strat():
    st.session_state.sb_strat_input = STRAT_LIST[(STRAT_LIST.index(st.session_state.sb_strat_input) + 1) % len(STRAT_LIST)]

dock_container = st.container()
with dock_container:
    st.markdown('<div id="bottom-dock-anchor"></div>', unsafe_allow_html=True)
    dock_c1, dock_c2, dock_c3 = st.columns(3)
    with dock_c1: st.button(f"🔄 Switch: {st.session_state.sb_index_input}", key="btn_asset_switch", on_click=cycle_asset, use_container_width=True)
    with dock_c2: st.button(f"🧠 Strat: {st.session_state.sb_strat_input.split(' ')[0]}", key="btn_strat_switch", on_click=cycle_strat, use_container_width=True)
    with dock_c3:
        if st.button("👆 Quick Login", use_container_width=True):
            if not st.session_state.bot:
                temp_bot = SniperBot(is_mock=True)
                temp_bot.login()
                st.session_state.bot = temp_bot

components.html(
    """<script>
    const doc = window.parent.document;
    const anchor = doc.getElementById('bottom-dock-anchor');
    if (anchor) {
        const wrapper = anchor.closest('.element-container').parentElement;
        wrapper.style.position = 'fixed'; 
        wrapper.style.bottom = '0px'; 
        wrapper.style.left = '0px'; 
        wrapper.style.width = '100%';
        wrapper.style.zIndex = '999999'; 
        wrapper.style.backgroundColor = 'rgba(255, 255, 255, 0.95)'; 
        wrapper.style.padding = '10px 15px'; 
        wrapper.style.borderTop = '1px solid #e2e8f0';
        wrapper.style.boxShadow = '0 -4px 6px -1px rgba(0, 0, 0, 0.1)';
    }
    </script>""", height=0)

if getattr(st.session_state, "bot", None) and st.session_state.bot.state.get("is_running"):
    time.sleep(2)
    st.rerun()
