import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import datetime as dt
import time
import requests
import pyotp
import hashlib
import hmac
import json
import uuid
import threading
import io
import yfinance as yf
from SmartApi import SmartConnect
from streamlit.runtime.scriptrunner import add_script_run_ctx
from collections import deque
from streamlit_lightweight_charts import renderLightweightCharts
from supabase import create_client, Client

try:
    import pandas_ta as ta
    HAS_PTA = True
except ImportError:
    HAS_PTA = False

try:
    from kiteconnect import KiteConnect
    HAS_ZERODHA = True
except ImportError:
    HAS_ZERODHA = False

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False

try:
    from plyer import notification
    HAS_NOTIFY = True
except ImportError:
    HAS_NOTIFY = False

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

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
        f'<div style="text-align: center; color: #f8fafc; font-size: 0.8rem; font-weight: bold; border-top: 1px solid #bae6fd; padding-top: 15px; margin-top: 15px;">'
        f'🚀 Algo trade<br>Developed by: Vijayakumar Suryavanshi</div>', 
        unsafe_allow_html=True
    )
    st.session_state['_dev_sig'] = "AUTH_OWNER_VIJAYAKUMAR_SURYAVANSHI"

def check_btst_stbt(df):
    if df is None or len(df) < 5: return "NO DATA"
    df = df.copy()
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    last = df.iloc[-1]
    if last['close'] > last['ema9'] > last['ema21'] and (last['high'] - last['close']) < (last['close'] - last['open']): return "🔥 BTST Suggested"
    elif last['close'] < last['ema9'] < last['ema21'] and (last['close'] - last['low']) < (last['open'] - last['close']): return "🩸 STBT Suggested"
    return "⚖️ Neutral (No Hold)"

def generate_delta_signature(method, endpoint, payload_string, secret):
    timestamp = str(int(time.time()))
    signature_data = method + timestamp + endpoint + payload_string
    signature = hmac.new(secret.encode('utf-8'), signature_data.encode('utf-8'), hashlib.sha256).hexdigest()
    return timestamp, signature

def get_usdt_inr_rate():
    try:
        res = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5).json()
        for coin in res:
            if coin['market'] == 'USDTINR': return float(coin['last_price'])
    except: pass
    return 86.50 

@st.cache_data(ttl=3600)
def get_all_crypto_pairs():
    pairs = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "BNBUSD"]
    try:
        res = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5).json()
        for coin in res:
            mkt = coin.get('market', '')
            if mkt.endswith('USDT'):
                base = mkt.replace('USDT', 'USD')
                if base not in pairs:
                    pairs.append(base)
    except: pass
    return sorted(pairs)

def get_market_status(asset_name):
    now_ist = get_ist()
    if "USD" in asset_name or "USDT" in asset_name or "INR" in asset_name and asset_name != "INDIA VIX":
        return True, "Crypto/Forex Live 🌍"
    if now_ist.weekday() >= 5: 
        return False, "Market Closed (Weekend)"
    if asset_name in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"]:
        if dt.time(9, 0) <= now_ist.time() <= dt.time(23, 30): 
            return True, "Commodity Live 🟠"
        return False, "Commodity Market Closed"
    if dt.time(9, 15) <= now_ist.time() <= dt.time(15, 30): 
        return True, "Equity Market Live 🟢"
    return False, "Equity Market Closed (After Hours)"

# ==========================================
# 1. DATABASE FUNCTIONS 
# ==========================================
def get_user_hash(user_id):
    if not user_id: return "guest"
    return hashlib.md5(user_id.encode()).hexdigest()[:8]

def load_creds(user_id):
    if not user_id: return {}
    if HAS_DB:
        try:
            res = supabase.table("user_credentials").select("*").eq("user_id", user_id).execute()
            if res.data: return res.data[0]
        except Exception as e:
            st.toast(f"DB Load Error: {e}")
    return {
        "user_id": user_id, "angel_api": "", "client_id": "", "pwd": "", "totp_secret": "", 
        "tg_token": "", "tg_chat": "", "wa_phone": "", "wa_api": "", 
        "mt5_acc": "", "mt5_pass": "", "mt5_server": "",
        "zerodha_api": "", "zerodha_secret": "", "coindcx_api": "", "coindcx_secret": "",
        "delta_api": "", "delta_secret": ""
    }

def save_creds(user_id, angel_api, client_id, pwd, totp_secret, tg_token, tg_chat, wa_phone, wa_api, mt5_acc, mt5_pass, mt5_server, zerodha_api, zerodha_secret, coindcx_api, coindcx_secret, delta_api, delta_secret):
    if HAS_DB:
        data = {
            "user_id": user_id, "angel_api": angel_api, "client_id": client_id, "pwd": pwd, 
            "totp_secret": totp_secret, "tg_token": tg_token, "tg_chat": tg_chat, 
            "wa_phone": wa_phone, "wa_api": wa_api,
            "mt5_acc": mt5_acc, "mt5_pass": mt5_pass, "mt5_server": mt5_server,
            "zerodha_api": zerodha_api, "zerodha_secret": zerodha_secret,
            "coindcx_api": coindcx_api, "coindcx_secret": coindcx_secret,
            "delta_api": delta_api, "delta_secret": delta_secret
        }
        try: supabase.table("user_credentials").upsert(data).execute()
        except: pass

def save_trade(user_id, trade_date, trade_time, symbol, t_type, qty, entry, exit_price, pnl, result):
    if HAS_DB and user_id and user_id != "mock_user":
        data = {
            "user_id": user_id, "trade_date": trade_date, "trade_time": trade_time,
            "symbol": symbol, "trade_type": t_type, "qty": qty,
            "entry_price": float(entry), "exit_price": float(exit_price),
            "pnl": float(pnl), "result": result
        }
        try: supabase.table("trade_logs").insert(data).execute()
        except: pass

# ==========================================
# 2. UI & CUSTOM CSS 
# ==========================================
st.set_page_config(page_title="SHREE", page_icon="🕉️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #f4f7f6; color: #0f111a; font-family: 'Inter', sans-serif; }
    
    @media (max-width: 850px) {
        header[data-testid="stHeader"] { visibility: visible !important; height: auto !important; background-color: #0284c7 !important; }
        header[data-testid="stHeader"] svg { fill: white !important; }
        .main .block-container { padding-top: 50px !important; }
    }
    
    [data-testid="stSidebar"] { background-color: #0284c7 !important; transition: all 0.4s ease; border-right: 1px solid #0369a1; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    
    div[data-baseweb="select"] * { color: #0f111a !important; font-weight: 600 !important; }
    div[data-baseweb="select"] { background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 2px !important; }
    div[data-baseweb="base-input"] > input, input[type="number"], input[type="password"], input[type="text"], textarea {
        color: #0f111a !important; font-weight: 600 !important; background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 2px !important;
    }

    div[data-testid="stTabs"] {
        background: transparent !important;
    }
    
    div[data-baseweb="tab-list"] {
        background: #e2e8f0 !important; 
        padding: 6px !important;
        border-radius: 12px !important;
        display: flex !important;
        gap: 8px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05) !important;
        border: none !important;
        margin-bottom: 20px !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        flex: 1 !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        background: transparent !important;
        color: #64748b !important;
        border: none !important;
        margin: 0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        white-space: nowrap !important;
        text-align: center !important;
        justify-content: center !important;
        box-shadow: none !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important; 
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4) !important;
        transform: scale(1.02) !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: #cbd5e1 !important;
        color: #0f111a !important;
    }

    div[data-baseweb="tab-highlight"] { display: none !important; }
    
    @media (max-width: 768px) {
        div[data-baseweb="tab-list"] {
            flex-wrap: wrap !important;
            border-radius: 14px !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-size: 0.85rem !important;
            padding: 10px 8px !important;
            min-width: 45% !important; 
        }
    }

    .glass-panel { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06); padding: 30px; }
    .bottom-dock-container { position: fixed !important; bottom: -500px !important; opacity: 0.01 !important; z-index: -1 !important; }
</style>
""", unsafe_allow_html=True)

DEFAULT_LOTS = {
    "NIFTY": 65, "BANKNIFTY": 30, "SENSEX": 20, "CRUDEOIL": 100, "NATURALGAS": 1250, 
    "GOLD": 100, "SILVER": 30, "XAUUSD": 0.01, "EURUSD": 0.01, "BTCUSD": 0.01, 
    "ETHUSD": 0.1, "SOLUSD": 1.0, "INDIA VIX": 1,
    "XRPUSD": 10.0, "ARBUSD": 1.0, "ADAUSD": 10.0, "XAGUSD": 0.1, "DOGEUSD": 100.0, 
    "BNBUSD": 0.05, "1000PEPEUSD": 1.0, "SUIUSD": 1.0, "NEARUSD": 1.0, "ENAUSD": 1.0, 
    "TIAUSD": 1.0, "1000BONKUSD": 1.0, "MEUSD": 1.0,
    "FETUSD": 1.0, "RNDRUSD": 1.0, "TAOUSD": 0.01, "INJUSD": 1.0, "AGIXUSD": 1.0
}
YF_TICKERS = {
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "CRUDEOIL": "CL=F", 
    "NATURALGAS": "NG=F", "GOLD": "GC=F", "SILVER": "SI=F", "XAUUSD": "GC=F", "EURUSD": "EURUSD=X", 
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD",
    "XRPUSD": "XRP-USD", "ARBUSD": "ARB-USD", "ADAUSD": "ADA-USD", "XAGUSD": "SI=F", 
    "DOGEUSD": "DOGE-USD", "BNBUSD": "BNB-USD", "1000PEPEUSD": "PEPE-USD", "SUIUSD": "SUI-USD", 
    "NEARUSD": "NEAR-USD", "ENAUSD": "ENA-USD", "TIAUSD": "TIA-USD", "1000BONKUSD": "BONK-USD", 
    "MEUSD": "ME-USD",
    "FETUSD": "FET-USD", "RNDRUSD": "RNDR-USD", "TAOUSD": "TAO-USD", "INJUSD": "INJ-USD", "AGIXUSD": "AGIX-USD"
}
INDEX_SYMBOLS = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank", "SENSEX": "BSE SENSEX", "INDIA VIX": "INDIA VIX"}
INDEX_TOKENS = {"NIFTY": ("NSE", "26000"), "BANKNIFTY": ("NSE", "26009"), "INDIA VIX": ("NSE", "26017"), "SENSEX": ("BSE", "99919000")}

STRAT_LIST = ["VIJAY & RFF All-In-One", "Intraday Trend Rider", "ICT", "Momentum Breakout + S&R", "Institutional FVG + SMC", "Keyword Rule Builder", "TradingView Webhook"]

if 'sb_index_input' not in st.session_state: st.session_state.sb_index_input = list(DEFAULT_LOTS.keys())[0]
if 'sb_strat_input' not in st.session_state: st.session_state.sb_strat_input = STRAT_LIST[0]
if 'bot' not in st.session_state: st.session_state.bot = None
if 'prev_index' not in st.session_state: st.session_state.prev_index = "NIFTY"
if 'custom_stock' not in st.session_state: st.session_state.custom_stock = ""
if 'custom_code_input' not in st.session_state: st.session_state.custom_code_input = "EMA Crossover (9 & 21)"

@st.cache_data(ttl=43200) 
def get_angel_scrip_master():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        df = pd.DataFrame(requests.get(url, timeout=45).json())
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
        else: df['vol_spike'] = df['volume'] >= (df['vol_sma'] * 0.9) 
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        
        atr = self.get_atr(df, 14)
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['is_sideways'] = (df['rsi'].between(48, 52)) & (abs(df['close'] - df['ema21']) < (atr * 0.3)) 
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
        
        fib_618 = major_high - (diff * 0.618)
        fib_650 = major_high - (diff * 0.650)
        
        return major_high, major_low, min(fib_618, fib_650), max(fib_618, fib_650)

    def detect_order_blocks(self, df):
        df = df.copy()
        df['body'] = abs(df['close'] - df['open'])
        avg_body = df['body'].rolling(10).mean()
        strong_up = (df['close'] > df['open']) & (df['body'] > avg_body * 1.5) 
        strong_down = (df['close'] < df['open']) & (df['body'] > avg_body * 1.5)
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

    def apply_ict_smc_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']

        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

        min_rsi = df['rsi'].rolling(14).min()
        max_rsi = df['rsi'].rolling(14).max()
        df['stoch_rsi'] = (df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-10)
        df['stoch_k'] = df['stoch_rsi'].rolling(3).mean() * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
        df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])

        latest_bull_top = df['low'].where(df['fvg_bull']).ffill()
        latest_bull_bot = df['high'].shift(2).where(df['fvg_bull']).ffill()
        latest_bear_bot = df['high'].where(df['fvg_bear']).ffill()
        latest_bear_top = df['low'].shift(2).where(df['fvg_bear']).ffill()

        last = df.iloc[-1]
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        smc_blocks = self.detect_order_blocks(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high, **smc_blocks}

        trend, signal = "AWAITING FVG REVERSAL 🟡", "WAIT"

        mitigated_bull = (last['low'] <= latest_bull_top.iloc[-1] * 1.002) and (last['low'] >= latest_bull_bot.iloc[-1] * 0.995)
        bull_reversal = last['close'] > last['open'] 

        in_golden_zone = (last['low'] <= f_high and last['high'] >= f_low)

        if in_golden_zone and bull_reversal and (last['stoch_k'] > last['stoch_d']):
            trend, signal = "SMC GOLDEN ZONE BULLISH REVERSAL 🟢", "BUY_CE"
        elif in_golden_zone and (not bull_reversal) and (last['stoch_k'] < last['stoch_d']):
            trend, signal = "SMC GOLDEN ZONE BEARISH REVERSAL 🔴", "BUY_PE"
        elif mitigated_bull and bull_reversal and (last['stoch_k'] > last['stoch_d']):
            trend, signal = "ICT BULL FVG REVERSAL CONFIRMED 🟢", "BUY_CE"
        elif (last['high'] >= latest_bear_bot.iloc[-1] * 0.998 and last['high'] <= latest_bear_top.iloc[-1] * 1.005) and (not bull_reversal) and (last['stoch_k'] < last['stoch_d']):
            trend, signal = "ICT BEAR FVG REVERSAL CONFIRMED 🔴", "BUY_PE"

        return trend, signal, last['vwap'], last['ema9'], df, atr, fib_data

    # 🔥 FIX: VIJAY RFF Strategy logic properly triggers on fresh EMA Crossovers with RSI to match TradingView logic precisely.
    def apply_vijay_rff_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]
        
        df = self.calculate_indicators(df, is_index)
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        smc_blocks = self.detect_order_blocks(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high, **smc_blocks}
        
        if not HAS_PTA:
            return "WAIT (pandas_ta required)", "WAIT", df['close'].iloc[-1], df['close'].iloc[-1], df, atr, fib_data

        df_ta = df.copy()
        df_ta.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        df_ta['EMA_5'] = ta.ema(df_ta['Close'], length=5)
        df_ta['EMA_13'] = ta.ema(df_ta['Close'], length=13)
        df_ta['EMA_21'] = ta.ema(df_ta['Close'], length=21)
        
        df_ta['RSI_14'] = ta.rsi(df_ta['Close'], length=14)
        if df_ta['RSI_14'] is None: df_ta['RSI_14'] = df_ta['Close'] * 0 + 50
        
        df_ta['VWAP'] = ta.vwap(df_ta['High'], df_ta['Low'], df_ta['Close'], df_ta['Volume'])
        if df_ta['VWAP'] is None or df_ta['VWAP'].isnull().all() or is_index:
            df_ta['VWAP'] = df_ta['Close']

        # 🔥 CROSSOVER DETECTION: Only triggers when the fast EMA explicitly crosses the slow EMA
        df_ta['EMA_Cross_Up'] = (df_ta['EMA_5'] > df_ta['EMA_13']) & (df_ta['EMA_5'].shift(1) <= df_ta['EMA_13'].shift(1))
        df_ta['EMA_Cross_Dn'] = (df_ta['EMA_5'] < df_ta['EMA_13']) & (df_ta['EMA_5'].shift(1) >= df_ta['EMA_13'].shift(1))

        df_ta['Buy_Signal'] = df_ta['EMA_Cross_Up'] & (df_ta['RSI_14'] >= 50)
        df_ta['Sell_Signal'] = df_ta['EMA_Cross_Dn'] & (df_ta['RSI_14'] <= 50)
        
        df['vwap'] = df_ta['VWAP']
        df['ema_fast'] = df_ta['EMA_13']
        
        last = df_ta.iloc[-1]
        
        signal = "WAIT"
        trend = "RANGING 🟡 (VIJAY_RFF)"
        
        if last['Buy_Signal']:
            signal = "BUY_CE"
            trend = "VIJAY_RFF UPTREND CROSSOVER 🟢"
        elif last['Sell_Signal']:
            signal = "BUY_PE"
            trend = "VIJAY_RFF DOWNTREND CROSSOVER 🔴"
            
        return trend, signal, last['VWAP'], last['EMA_13'], df, atr, fib_data

    # 🔥 FIX: Intraday Trend Rider now triggers specifically on price pullbacks to the fast EMA, creating precision entries.
    def apply_trend_rider_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']
        df['ema_trend'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_fast'] = df['close'].ewm(span=13, adjust=False).mean()
        
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        smc_blocks = self.detect_order_blocks(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high, **smc_blocks}
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        signal, trend = "WAIT", "RANGING 🟡"
        
        # Detect Price Crossover against the Fast EMA while maintaining the macro trend
        price_cross_up = (last['close'] > last['ema_fast']) and (prev['close'] <= prev['ema_fast'])
        price_cross_dn = (last['close'] < last['ema_fast']) and (prev['close'] >= prev['ema_fast'])
        
        if price_cross_up and last['ema_fast'] > last['ema_trend'] and last['rsi'] > 50:
            signal, trend = "BUY_CE", "TREND RIDER PULLBACK UPTREND 🚀"
        elif price_cross_dn and last['ema_fast'] < last['ema_trend'] and last['rsi'] < 50:
            signal, trend = "BUY_PE", "TREND RIDER PULLBACK DOWNTREND 🩸"
                    
        return trend, signal, last['vwap'], last['ema_fast'], df, atr, fib_data

    def apply_vwap_ema_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 50: return "WAIT", "WAIT", 0, 0, df, 0, {}
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high}
        last = df.iloc[-1]
        signal, trend = "WAIT", "FLAT"
        
        benchmark = last['ema_long'] if is_index else last['vwap']
        if (last['ema_short'] > last['ema_long']) and (last['close'] > benchmark) and last['rsi'] > 50: 
            trend, signal = "BULLISH MOMENTUM 🟢", "BUY_CE"
        elif (last['ema_short'] < last['ema_long']) and (last['close'] < benchmark) and last['rsi'] < 50: 
            trend, signal = "BEARISH MOMENTUM 🔴", "BUY_PE"
            
        return trend, signal, last['vwap'], last['ema_short'], df, atr, fib_data

    def apply_keyword_strategy(self, df, keywords, index_name):
        if df is None or len(df) < 30: return "WAIT", "WAIT", 0, 0, df, 0, {}
        df = df.copy()
        
        df['ema9'] = ta.ema(df['close'], 9)
        df['ema21'] = ta.ema(df['close'], 21)
        df['rsi'] = ta.rsi(df['close'], 14)
        
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macds'] = macd['MACDs_12_26_9']
            
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            df['bbl'] = bb['BBL_20_2.0']
            df['bbh'] = bb['BBU_20_2.0']

        last = df.iloc[-1]
        prev = df.iloc[-2]

        buy_conds, sell_conds = [], []
        keys = keywords.split(',') if keywords else []

        if "EMA Crossover (9 & 21)" in keys:
            buy_conds.append(last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21'])
            sell_conds.append(last['ema9'] < last['ema21'] and prev['ema9'] >= prev['ema21'])

        if "RSI Breakout (>60/<40)" in keys:
            buy_conds.append(last['rsi'] > 60)
            sell_conds.append(last['rsi'] < 40)

        if "MACD Crossover" in keys:
            if 'macd' in df.columns:
                buy_conds.append(last['macd'] > last['macds'] and prev['macd'] <= prev['macds'])
                sell_conds.append(last['macd'] < last['macds'] and prev['macd'] >= prev['macds'])

        if "Bollinger Bands Bounce" in keys:
            if 'bbl' in df.columns:
                buy_conds.append(last['close'] > last['bbl'] and prev['close'] <= prev['bbl'])
                sell_conds.append(last['close'] < last['bbh'] and prev['close'] >= prev['bbh'])

        signal, trend = "WAIT", "Awaiting Keyword Match 🟡"

        if buy_conds and all(buy_conds):
            signal, trend = "BUY_CE", "Keyword Setup Met: BULLISH 🟢"
        elif sell_conds and all(sell_conds):
            signal, trend = "BUY_PE", "Keyword Setup Met: BEARISH 🔴"

        return trend, signal, last['close'], last['ema9'], df, self.get_atr(df).iloc[-1], {}

# ==========================================
# 4. CORE BOT ENGINE
# ==========================================
class SniperBot:
    def __init__(self, api_key="", client_id="", pwd="", totp_secret="", tg_token="", tg_chat="", wa_phone="", wa_api="", mt5_acc="", mt5_pass="", mt5_server="", zerodha_api="", zerodha_secret="", request_token="", coindcx_api="", coindcx_secret="", delta_api="", delta_secret="", is_mock=False):
        self.api_key, self.client_id, self.pwd, self.totp_secret = api_key, client_id, pwd, totp_secret
        self.tg_token, self.tg_chat, self.wa_phone, self.wa_api = tg_token, tg_chat, wa_phone, wa_api
        self.mt5_acc, self.mt5_pass, self.mt5_server = mt5_acc, mt5_pass, mt5_server
        self.zerodha_api, self.zerodha_secret, self.request_token = zerodha_api, zerodha_secret, request_token
        self.coindcx_api, self.coindcx_secret = coindcx_api, coindcx_secret
        self.delta_api, self.delta_secret = delta_api, delta_secret
        
        self.api, self.kite, self.token_map, self.is_mock = None, None, None, is_mock
        self.is_mt5_connected = False
        self.client_name = "Offline User"
        self.client_ip = get_client_ip()
        self.user_hash = get_user_hash(self.api_key)
        self.analyzer = TechnicalAnalyzer()
        
        self.state = {
            "is_running": False, "order_in_flight": False, "active_trade": None, "last_trade": None,
            "logs": deque(maxlen=50), "current_trend": "WAIT", "current_signal": "WAIT",
            "spot": 0.0, "vwap": 0.0, "ema": 0.0, "atr": 0.0, "fib_data": {}, "latest_data": None,
            "latest_candle": None,
            "ui_popups": deque(maxlen=10), "loop_count": 0, "daily_pnl": 0.0, "trades_today": 0,
            "manual_exit": False,
            "ghost_memory": {},
            "tv_signal": {"action": "WAIT", "symbol": "", "timestamp": 0}
        }
        self.settings = {}
        
    def start_webhook_listener(self):
        if not HAS_FLASK:
            self.log("⚠️ Flask not installed. Webhook listener won't work.")
            return
            
        app = Flask(__name__)
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        @app.route('/tv_webhook', methods=['POST'])
        def webhook():
            data = request.json
            if data and data.get("passphrase") == self.settings.get("tv_passphrase", "SHREE123"):
                action = data.get("action", "WAIT").upper()
                symbol = data.get("symbol", "").upper()
                self.state["tv_signal"] = {"action": action, "symbol": symbol, "timestamp": time.time()}
                self.log(f"🔔 TV Webhook Alert: {action} on {symbol}")
                return jsonify({"status": "success"}), 200
            return jsonify({"status": "unauthorized"}), 401
            
        threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'use_reloader': False}, daemon=True).start()
        self.log("🌐 TradingView Webhook Listener Active on Port 5000")

    def push_notify(self, title, message):
        self.state["ui_popups"].append({"title": title, "message": message})
        if HAS_NOTIFY:
            try: notification.notify(title=title, message=message, app_name="QUANT", timeout=5)
            except: pass
        if self.tg_token and self.tg_chat:
            try: requests.get(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", params={"chat_id": self.tg_chat, "text": f"<b>{title}</b>\n{message}", "parse_mode": "HTML"}, timeout=3)
            except: pass
        if self.wa_phone and self.wa_api:
            try: requests.get("https://api.callmebot.com/whatsapp.php", params={"phone": self.wa_phone, "text": f"{title}\n{message}", "apikey": self.wa_api}, timeout=3)
            except: pass

    def log(self, msg):
        self.state["logs"].appendleft(f"[{get_ist().strftime('%H:%M:%S')}] {msg}")

    def get_balance(self):
        if self.is_mock: 
            cap = self.settings.get("max_capital", 15000.0) if self.settings else 15000.0
            return f"₹ {cap:,.2f} (Paper)"
        b_str = []
        
        if self.api:
            try:
                rms = self.api.rms()
                if rms and rms.get('status') and rms.get('data'):
                    data = rms['data']
                    bal = data.get('availablecash', data.get('net', 0))
                    try: bal = float(bal)
                    except: bal = 0.0
                    b_str.append(f"Angel: ₹ {round(bal, 2)}")
            except: pass
            
        if self.kite:
            try:
                margins = self.kite.margins()
                eq = margins.get('equity', {})
                bal = eq.get('available', {}).get('live_balance', eq.get('net', 0))
                try: bal = float(bal)
                except: bal = 0.0
                b_str.append(f"Zerodha: ₹ {round(bal, 2)}")
            except: pass
            
        if self.coindcx_api:
            try:
                ts = int(round(time.time() * 1000))
                payload = {"timestamp": ts}
                secret_bytes = bytes(self.coindcx_secret, 'utf-8')
                signature = hmac.new(secret_bytes, json.dumps(payload, separators=(',', ':')).encode('utf-8'), hashlib.sha256).hexdigest()
                res = requests.post("https://api.coindcx.com/exchange/v1/users/balances", headers={'X-AUTH-APIKEY': self.coindcx_api, 'X-AUTH-SIGNATURE': signature}, json=payload)
                if res.status_code == 200:
                    for b in res.json():
                        if b['currency'] == 'USDT': 
                            bal = float(b['balance'])
                            if self.settings.get('show_inr_crypto', True): b_str.append(f"DCX: ₹ {round(bal * get_usdt_inr_rate(),2)}")
                            else: b_str.append(f"DCX: $ {round(bal,2)}")
            except: pass
            
        if self.delta_api:
            try:
                ts, sig = generate_delta_signature('GET', '/v2/wallet/balances', '', self.delta_secret)
                headers = {'api-key': self.delta_api, 'signature': sig, 'timestamp': ts}
                res = requests.get("https://api.delta.exchange/v2/wallet/balances", headers=headers)
                if res.status_code == 200:
                    for b in res.json().get('result', []):
                        if b['asset_symbol'] == 'USDT': 
                            bal = float(b['balance'])
                            if self.settings.get('show_inr_crypto', True): b_str.append(f"Delta: ₹ {round(bal * get_usdt_inr_rate(),2)}")
                            else: b_str.append(f"Delta: $ {round(bal,2)}")
            except: pass
            
        if self.is_mt5_connected:
            try:
                acc = mt5.account_info()
                if acc: b_str.append(f"MT5: $ {round(acc.balance, 2)}")
            except: pass
            
        return " | ".join(b_str) if b_str else "N/A"

    def login(self):
        if self.is_mock: 
            self.client_name, self.api_key = "Paper Trading User", "mock_user"
            self.user_hash = get_user_hash(self.api_key)
            self.push_notify("🟢 Session Started", f"Paper Trading active.")
            self.start_webhook_listener()
            return True
        
        success = False
        if self.api_key and self.totp_secret:
            try:
                obj = SmartConnect(api_key=self.api_key)
                totp = pyotp.TOTP(self.totp_secret).now()
                res = obj.generateSession(self.client_id, self.pwd, totp)
                if res and res.get('status'):
                    self.api = obj
                    self.client_name = res.get('data', {}).get('name', self.client_id)
                    self.log(f"✅ Angel One Connected")
                    success = True
                else: self.log(f"❌ Angel Login failed: {res.get('message', 'Check credentials')}")
            except Exception as e: self.log(f"❌ Angel Login Exception: {e}")

        if self.zerodha_api and self.zerodha_secret and self.request_token and HAS_ZERODHA:
            try:
                self.kite = KiteConnect(api_key=self.zerodha_api)
                data = self.kite.generate_session(self.request_token, api_secret=self.zerodha_secret)
                self.kite.set_access_token(data["access_token"])
                self.log(f"✅ Zerodha Kite Connected")
                success = True
            except Exception as e: self.log(f"❌ Zerodha Exception: {e}")

        if self.mt5_acc and self.mt5_server and HAS_MT5:
            try:
                if mt5.initialize():
                    if mt5.login(int(self.mt5_acc), password=self.mt5_pass, server=self.mt5_server):
                        self.log(f"✅ MT5 Connected")
                        self.is_mt5_connected = True
                        success = True
                    else: self.log(f"❌ MT5 Login failed: {mt5.last_error()}")
                else: self.log(f"❌ MT5 Init failed: {mt5.last_error()}")
            except Exception as e: self.log(f"❌ MT5 Exception: {e}")

        if self.coindcx_api and self.coindcx_secret:
            self.log(f"✅ CoinDCX Credentials Loaded")
            success = True
            
        if self.delta_api and self.delta_secret:
            self.log(f"✅ Delta Exchange Credentials Loaded")
            success = True
                
        if success:
            self.push_notify("🟢 Gateway Active", f"Connections established.")
            self.start_webhook_listener()
            return True
        return False

    def get_master(self):
        if self.token_map is None or self.token_map.empty: self.token_map = get_angel_scrip_master()
        return self.token_map

    def get_token_info(self, index_name):
        if index_name in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "SOLUSD"] and self.settings.get("primary_broker") == "MT5":
            return "MT5", index_name
        
        if self.settings.get("primary_broker") == "Delta Exchange" or ("USD" in index_name and "Delta" in self.settings.get("primary_broker", "")):
            return "DELTA", index_name
        if self.settings.get("primary_broker") == "CoinDCX" or "USDT" in index_name or "INR" in index_name:
            return "COINDCX", index_name
            
        if index_name in INDEX_TOKENS: return INDEX_TOKENS[index_name]
        df_map = self.get_master()
        if df_map is not None and not df_map.empty:
            today_date = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
            futs = df_map[(df_map['name'] == index_name) & (df_map['instrumenttype'].isin(['FUTCOM', 'FUTIDX', 'FUTSTK', 'EQ']))]
            if not futs.empty:
                eqs = futs[futs['instrumenttype'] == 'EQ']
                if not eqs.empty: return eqs.iloc[0]['exch_seg'], eqs.iloc[0]['token']
                futs = futs[futs['expiry'] >= today_date]
                if not futs.empty: return futs[futs['expiry'] == futs['expiry'].min()].iloc[0]['exch_seg'], futs[futs['expiry'] == futs['expiry'].min()].iloc[0]['token']
        return "NSE", "12345"

    def get_market_data_oi(self, exchange, token):
        if self.is_mock or exchange in ["MT5", "COINDCX", "DELTA"]: return np.random.randint(50000, 150000), np.random.randint(1000, 10000)
        if not self.api: return 0, 0
        try:
            res = self.api.marketData({"mode": "FULL", "exchangeTokens": { exchange: [str(token)] }})
            if res and res.get('status') and res.get('data'): return res['data']['fetched'][0].get('opnInterest', 0), res['data']['fetched'][0].get('totMacVal', 0)
        except: pass
        return 0, 0

    def get_live_price(self, exchange, symbol, token):
        if self.is_mock and token == "12345": 
            base_prices = {"NIFTY": 22000, "BANKNIFTY": 47000, "SENSEX": 73000, "NATURALGAS": 145.0, "CRUDEOIL": 6500.0, "GOLD": 62000.0, "SILVER": 72000.0, "XAUUSD": 2050.0, "EURUSD": 1.0850, "BTCUSD": 65000.0, "ETHUSD": 3500.0, "SOLUSD": 150.0}
            base = base_prices.get(symbol, 500)
            return float(np.random.uniform(base - 10, base + 10))
            
        if exchange == "MT5" and self.is_mt5_connected:
            tick = mt5.symbol_info_tick(symbol)
            if tick: return (tick.bid + tick.ask) / 2.0
            return None
            
        if exchange == "COINDCX" and self.coindcx_api:
            try:
                res = requests.get(f"https://api.coindcx.com/exchange/ticker").json()
                for coin in res:
                    target = symbol.replace("USD", "USDT") if symbol.endswith("USD") and not symbol.endswith("USDT") else symbol
                    if coin['market'] == target or target in coin['market'].replace('_', ''): 
                        price = float(coin['last_price'])
                        return price
            except: return None
            
        if exchange == "DELTA" and self.delta_api:
            try:
                target = symbol if symbol.endswith("USD") or symbol.endswith("USDT") else f"{symbol}USD"
                res = requests.get(f"https://api.delta.exchange/v2/products/ticker/24hr?symbol={target}").json()
                if res.get('success'): 
                    price = float(res['result']['close'])
                    return price
            except: return None

        if self.kite and self.settings.get("primary_broker") == "Zerodha":
            try:
                tsym = f"{exchange}:{symbol}"
                res = self.kite.quote([tsym])
                return float(res[tsym]['last_price'])
            except: pass

        if self.api:
            try:
                trading_symbol = INDEX_SYMBOLS.get(symbol, symbol)
                res = self.api.ltpData(exchange, trading_symbol, str(token))
                if res and res.get('status'): return float(res['data']['ltp'])
            except: pass
        return None

    def get_historical_data(self, exchange, token, symbol="NIFTY", interval="5m"):
        if self.is_mock and token == "12345": return self._fallback_yfinance(symbol, interval)
        if exchange == "MT5" and self.is_mt5_connected:
            try:
                mt5_interval_map = {"1m": mt5.TIMEFRAME_M1, "3m": mt5.TIMEFRAME_M3, "5m": mt5.TIMEFRAME_M5, "15m": mt5.TIMEFRAME_M15}
                tf = mt5_interval_map.get(interval, mt5.TIMEFRAME_M5)
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, 500)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.rename(columns={'time': 'timestamp', 'real_volume': 'volume'}, inplace=True)
                    df.index = df['timestamp']
                    return df
            except: pass
            return self._fallback_yfinance(symbol, interval)
            
        if not self.api and not self.kite: return self._fallback_yfinance(symbol, interval)
        
        try:
            now_ist = get_ist()
            fromdate = now_ist - dt.timedelta(days=10) 
            
            if self.kite and self.settings.get("primary_broker") == "Zerodha":
                z_int_map = {"1m": "minute", "3m": "3minute", "5m": "5minute", "15m": "15minute"}
                records = self.kite.historical_data(int(token), fromdate.strftime("%Y-%m-%d"), now_ist.strftime("%Y-%m-%d"), z_int_map.get(interval, "5minute"))
                df = pd.DataFrame(records)
                df.rename(columns={'date': 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.index = df['timestamp']
                return df
            elif self.api:
                interval_map = {"1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE"}
                api_interval = interval_map.get(interval, "FIVE_MINUTE")
                res = self.api.getCandleData({"exchange": exchange, "symboltoken": str(token), "interval": api_interval, "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"), "todate": now_ist.strftime("%Y-%m-%d %H:%M")})
                if res and res.get('status') and res.get('data'):
                    df = pd.DataFrame(res['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    if df.empty: return self._fallback_yfinance(symbol, interval)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.index = df['timestamp']
                    return df
        except: pass
        return self._fallback_yfinance(symbol, interval)
            
    def _fallback_yfinance(self, symbol, interval):
        yf_int = interval if interval in ["1m", "5m", "15m"] else "5m" 
        yf_ticker = YF_TICKERS.get(symbol)
        
        if not yf_ticker and ("USD" in symbol or "USDT" in symbol):
            base_coin = symbol.replace("USDT", "").replace("USD", "")
            yf_ticker = f"{base_coin}-USD"
            
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

    def analyze_oi_and_greeks(self, df, is_hero_zero, signal):
        if not is_hero_zero or df is None or len(df) < 14: return True, ""
        
        last = df.iloc[-1]
        atr = self.analyzer.get_atr(df).iloc[-1]
        body = abs(last['close'] - last['open'])
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        
        recent_range = df['high'].rolling(5).max().iloc[-1] - df['low'].rolling(5).min().iloc[-1]
        is_choppy = recent_range < (atr * 1.0)
        
        if is_choppy and last['volume'] < vol_sma:
            return False, "⚠️ Blocked: Consolidating Market. High Theta Decay Risk."
            
        if last['volume'] > (vol_sma * 1.2) and body > (atr * 0.5):
            if signal == "BUY_CE" and last['close'] > last['open']: 
                return True, "🔥 Gamma Blast Detected! Buying OTM Bottom."
            elif signal == "BUY_PE" and last['close'] < last['open']: 
                return True, "🩸 Gamma Blast Detected! Buying OTM Bottom."
                
        return False, "⏳ Waiting for Volatility Expansion (Avoid Theta)."

    def place_real_order(self, symbol, token, qty, side="BUY", exchange="NFO"):
        if self.is_mock: return "MOCK_" + uuid.uuid4().hex[:6].upper()
        broker = self.settings.get("primary_broker", "Angel One")
        
        formatted_qty = str(int(float(qty))) if exchange in ["NFO", "NSE", "BFO", "MCX"] else str(qty)
        self.log(f"⚙️ Executing Real API: {symbol} | Qty: {qty} | Side: {side} | Exchange: {exchange}")

        if exchange == "DELTA":
            try:
                target = symbol if symbol.endswith("USD") or symbol.endswith("USDT") else f"{symbol}USD"
                payload = {"product_id": target, "size": int(float(qty)), "side": "buy" if side == "BUY" else "sell", "order_type": "market"}
                payload_str = json.dumps(payload, separators=(',', ':'))
                ts, sig = generate_delta_signature('POST', '/v2/orders', payload_str, self.delta_secret)
                headers = {'api-key': self.delta_api, 'signature': sig, 'timestamp': ts, 'Content-Type': 'application/json'}
                res = requests.post("https://api.delta.exchange/v2/orders", headers=headers, data=payload_str)
                if res.status_code == 200: 
                    self.log(f"✅ Delta Order Success! ID: {res.json().get('result', {}).get('id')}")
                    return res.json().get('result', {}).get('id')
                else: 
                    self.log(f"❌ Delta API Rejected: {res.text}"); return None
            except Exception as e: 
                self.log(f"❌ Delta Exception: {e}"); return None

        if exchange == "COINDCX":
            try:
                ts = int(round(time.time() * 1000))
                market_type = self.settings.get("crypto_mode", "Spot")
                base_coin = symbol.replace("USDT", "").replace("USD", "").replace("INR", "")
                clean_qty = float(round(float(qty), 4))
                
                exact_market = f"{base_coin}USDT"
                exact_pair = f"B-{base_coin}_USDT"
                
                try:
                    ticker_data = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5).json()
                    for coin in ticker_data:
                        mkt = coin.get('market', '')
                        if market_type in ["Futures", "Options"] and mkt.startswith(f"B-{base_coin}_"):
                            exact_pair = mkt
                            break
                        elif market_type == "Spot" and mkt.startswith(base_coin) and mkt.endswith("USDT"):
                            exact_market = mkt
                            break
                except: pass

                if market_type in ["Futures", "Options"]:
                    payload = {
                        "side": side.lower(), 
                        "order_type": "market", 
                        "pair": exact_pair,     
                        "total_quantity": clean_qty, 
                        "timestamp": ts
                    }
                    endpoint = "https://api.coindcx.com/exchange/v1/derivatives/futures/orders/create"
                else:
                    payload = {
                        "side": side.lower(), 
                        "order_type": "market_order", 
                        "market": exact_market,       
                        "total_quantity": clean_qty, 
                        "timestamp": ts
                    }
                    endpoint = "https://api.coindcx.com/exchange/v1/orders/create"

                payload_str = json.dumps(payload, separators=(',', ':'))
                secret_bytes = bytes(self.coindcx_secret, 'utf-8')
                signature = hmac.new(secret_bytes, payload_str.encode('utf-8'), hashlib.sha256).hexdigest()
                
                res = requests.post(endpoint, headers={'X-AUTH-APIKEY': self.coindcx_api, 'X-AUTH-SIGNATURE': signature, 'Content-Type': 'application/json'}, data=payload_str)
                
                if res.status_code == 404 and market_type in ["Futures", "Options"]:
                    self.log(f"⚠️ Futures 404 on {exact_pair}. Falling back to Margin API...")
                    payload = {"side": side.lower(), "order_type": "market_order", "market": exact_market, "total_quantity": clean_qty, "timestamp": ts}
                    endpoint = "https://api.coindcx.com/exchange/v1/margin/create"
                    payload_str = json.dumps(payload, separators=(',', ':'))
                    signature = hmac.new(secret_bytes, payload_str.encode('utf-8'), hashlib.sha256).hexdigest()
                    res = requests.post(endpoint, headers={'X-AUTH-APIKEY': self.coindcx_api, 'X-AUTH-SIGNATURE': signature, 'Content-Type': 'application/json'}, data=payload_str)

                if res.status_code == 200: 
                    response_data = res.json()
                    order_id = response_data.get('orders', [{}])[0].get('id', response_data.get('id', 'DCX_ORDER_OK'))
                    self.log(f"✅ CoinDCX Order Success! ID: {order_id}")
                    return order_id
                else: 
                    self.log(f"❌ CoinDCX API Rejected [{res.status_code}]: {res.text}")
                    return None
            except Exception as e: 
                self.log(f"❌ CoinDCX Exception: {e}"); return None

        if exchange == "MT5" and self.is_mt5_connected:
            try:
                action_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
                tick = mt5.symbol_info_tick(symbol)
                price = tick.ask if side == "BUY" else tick.bid
                request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(qty), "type": action_type, "price": price, "deviation": 20, "magic": 234000, "comment": "QUANT Algo", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE: 
                    self.log(f"❌ MT5 Order Failed: {result.comment}"); return None
                self.log(f"✅ MT5 Order Success! ID: {result.order}")
                return result.order
            except Exception as e: self.log(f"❌ MT5 Exception: {e}"); return None

        if broker == "Zerodha" and self.kite:
            try:
                z_side = self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL
                order_id = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, exchange=exchange, tradingsymbol=symbol, transaction_type=z_side, quantity=int(float(qty)), product=self.kite.PRODUCT_MIS, order_type=self.kite.ORDER_TYPE_MARKET)
                self.log(f"✅ Zerodha Order Pushed! ID: {order_id}")
                return order_id
            except Exception as e: self.log(f"❌ Zerodha Order Error: {str(e)}"); return None

  # 🔥 ANGEL ONE EXECUTION BLOCK
        try: 
            p_type = "CARRYFORWARD" if exchange in ["NFO", "BFO", "MCX"] else "INTRADAY"
            
            # Default to MARKET, but we will convert Options to safe LIMIT orders
            order_type = "MARKET"
            exec_price = 0.0
            
            # 🛡️ Anti-Freak Trade Protection for Options
            if exchange in ["NFO", "BFO"]:
                ltp = self.get_live_price(exchange, symbol, token)
                if ltp and ltp > 0:
                    order_type = "LIMIT"
                    if side.upper() == "BUY":
                        safe_price = ltp * 1.05  # 5% buffer above LTP
                    else:
                        safe_price = ltp * 0.95  # 5% buffer below LTP
                    
                    # Round strictly to NSE's 0.05 tick size
                    exec_price = round(round(safe_price / 0.05) * 0.05, 2)
                else:
                    self.log(f"⚠️ Could not fetch LTP for {symbol}. Retrying as pure MARKET.")

            # 🔥 FIX: Angel API Gateway strictly requires native numbers, NOT strings for price, squareoff, quantity
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": str(symbol),
                "symboltoken": str(token),
                "transactiontype": str(side.upper()),
                "exchange": str(exchange.upper()),
                "ordertype": str(order_type),
                "producttype": str(p_type),
                "duration": "DAY",
                "price": float(exec_price),
                "squareoff": 0.0,
                "stoploss": 0.0,
                "quantity": int(float(qty))
            }
            
            self.log(f"📡 Sending Angel Payload: {order_params}")
            res = self.api.placeOrder(order_params)
            
            if res is None:
                # The Python SDK failed to parse the response. Doing a direct request to catch the raw text.
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api.access_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "X-UserType": "USER",
                        "X-SourceID": "WEB",
                        "X-ClientLocalIP": self.client_ip,
                        "X-ClientPublicIP": self.client_ip,
                        "X-MACAddress": "00:00:00:00:00:00",
                        "X-PrivateKey": self.api_key
                    }
                    url = "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/placeOrder"
                    raw_res = requests.post(url, headers=headers, json=order_params, timeout=5)
                    self.log(f"❌ Gateway Error [{raw_res.status_code}]: {raw_res.text}")
                except Exception as e:
                    self.log(f"❌ Angel API Timeout. Network issue: {e}")
                return None
                
            elif isinstance(res, str):
                self.log(f"✅ Angel Order Placed! ID: {res}")
                return res
            elif isinstance(res, dict):
                if res.get('status'):
                    o_id = res.get('data', {}).get('orderid', 'UNKNOWN_ID') if res.get('data') else 'UNKNOWN_ID'
                    self.log(f"✅ Angel Order Placed! ID: {o_id}")
                    return o_id
                else:
                    self.log(f"❌ Angel API Validation Error: {res.get('message')}")
                    return None
            else:
                self.log(f"❌ Angel Unknown Response Type: {type(res)} -> {res}")
                return None
        except Exception as e: 
            self.log(f"❌ Exception placing Angel order: {str(e)}"); return None
    def get_strike(self, symbol, spot, signal, max_premium):
        opt_type = "CE" if "BUY_CE" in signal else "PE"
        
        if self.settings.get("primary_broker") in ["CoinDCX", "Delta Exchange"] and self.settings.get("crypto_mode") == "Options":
            rounder = 500 if "BTC" in symbol else (50 if "ETH" in symbol else 1)
            strike_price = round(spot / rounder) * rounder
            expiry_str = (get_ist() + dt.timedelta(days=(4 - get_ist().weekday()) % 7)).strftime("%d%b%y").upper() 
            crypto_sym = f"{symbol.replace('USDT', '').replace('USD', '')}-{expiry_str}-{int(strike_price)}-{opt_type}"
            exch_target = "COINDCX" if self.settings.get("primary_broker") == "CoinDCX" else "DELTA"
            return crypto_sym, crypto_sym, exch_target, spot * 0.02
            
        df = self.get_master()
        if df is None or df.empty: 
            if self.is_mock: return f"{symbol}28FEB{int(spot)}{opt_type}", "12345", "NFO", min(100.0, max_premium)
            self.log("⚠️ Option Chain JSON is empty. Cannot compute Angel strikes.")
            return None, None, None, 0.0

        today = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
        mask = (df['name'] == symbol) & (df['exch_seg'].isin(["NFO", "MCX", "BFO"])) & (df['expiry'] >= today) & (df['symbol'].str.endswith(opt_type))
        subset = df[mask].copy()
        
        if subset.empty: 
            if self.is_mock: return f"{symbol}28FEB{int(spot)}{opt_type}", "12345", "NFO", min(100.0, max_premium)
            return None, None, None, 0.0

        closest_expiry = subset['expiry'].min()
        subset = subset[subset['expiry'] == closest_expiry]
        subset['dist_to_spot'] = abs(subset['strike'] - spot)
        
        if self.settings.get("hero_zero"):
            otm_margin = spot * 0.003 
            if opt_type == "CE": candidates = subset[subset['strike'] > (spot + otm_margin)]
            else: candidates = subset[subset['strike'] < (spot - otm_margin)]
            candidates = candidates.sort_values('dist_to_spot', ascending=True).head(15)
        else: 
            candidates = subset.sort_values('dist_to_spot', ascending=True).head(10)
            
        for _, row in candidates.iterrows():
            ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
            if ltp is None and self.is_mock:
                ltp = max_premium * 0.85
            if ltp and ltp <= max_premium: 
                return row['symbol'], row['token'], row['exch_seg'], ltp

        if self.is_mock: return f"{symbol}28FEB{int(spot)}{opt_type}", "12345", "NFO", min(100.0, max_premium)
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

                index, timeframe, is_mock_mode, strategy = s['index'], s['timeframe'], s['paper_mode'], s['strategy']
                
                is_open, mkt_msg = get_market_status(index)
                if not is_open:
                    time.sleep(10)
                    continue

                exch, token = self.get_token_info(index)
                is_mt5_asset = (exch == "MT5")
                is_crypto = (exch in ["COINDCX", "DELTA"])

                if index in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]:
                    cutoff_time = dt.time(15, 15)
                elif index in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"]:
                    cutoff_time = dt.time(23, 15)
                else:
                    cutoff_time = dt.time(23, 59, 59)
                
                spot = self.get_live_price(exch, index, token)
                if spot is None and self.is_mock: spot = self.get_live_price("NSE", index, "12345")
                
                df_candles = self.get_historical_data(exch, token, symbol=index, interval=timeframe) if not self.is_mock else self.get_historical_data("MOCK", "12345", symbol=index, interval=timeframe)
                
                user_lots_dict = s.get('user_lots', DEFAULT_LOTS)
                base_lot_size = user_lots_dict.get(index, 25) if not (is_mt5_asset or is_crypto) else 0.01
                
                # ---- STRATEGY LOGIC ROUTING ----
                if strategy == "TradingView Webhook":
                    trend = "Listening for TV Webhook 📡"
                    signal = "WAIT"
                    vwap, ema, df_chart, current_atr, fib_data = spot, spot, df_candles, 0, {}
                    
                    tv_action = self.state.get("tv_signal", {}).get("action")
                    tv_symbol = self.state.get("tv_signal", {}).get("symbol")
                    tv_time = self.state.get("tv_signal", {}).get("timestamp", 0)
                    
                    if tv_action in ["BUY_CE", "BUY_PE"] and (time.time() - tv_time) < 60:
                        if tv_symbol == index or tv_symbol == "ALL":
                            signal = tv_action
                            trend = f"TV Alert Triggered: {signal} 🚀"
                            self.state["tv_signal"]["action"] = "WAIT" 
                
                elif spot and df_candles is not None and not df_candles.empty:
                    self.state["spot"] = spot
                    last_candle = df_candles.iloc[-1]
                    self.state["latest_candle"] = last_candle.to_dict()
                    
                    if strategy == "Keyword Rule Builder":
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_keyword_strategy(df_candles, s.get('custom_code', ''), index)
                    elif "VIJAY & RFF" in strategy: 
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_vijay_rff_strategy(df_candles, index)
                    elif "Institutional FVG" in strategy or "ICT" in strategy: 
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_ict_smc_strategy(df_candles, index)
                    elif "Trend Rider" in strategy: 
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_trend_rider_strategy(df_candles, index)
                    else: 
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data = self.analyzer.apply_vwap_ema_strategy(df_candles, index)

                    if s.get("fomo_entry") and strategy != "TradingView Webhook":
                        body = abs(last_candle['close'] - last_candle['open'])
                        avg_body = abs(df_candles['close'] - df_candles['open']).rolling(14).mean().iloc[-1]
                        vol_spike = last_candle['volume'] > (df_candles['volume'].rolling(20).mean().iloc[-1] * 1.2)
                        
                        if body > (avg_body * 2.0) and vol_spike: 
                            signal = "BUY_CE" if last_candle['close'] > last_candle['open'] else "BUY_PE"
                            trend = "🚨 FOMO BREAKOUT ACTIVE"
                            if self.state["active_trade"] is None:
                                self.push_notify("🚨 FOMO ALERT", f"High Momentum Detected on {index}!")

                    if signal != "WAIT":
                        last_trade_time = self.state["ghost_memory"].get(f"{index}_{signal}")
                        if last_trade_time and (get_ist() - last_trade_time).seconds < 900:
                            signal = "WAIT"
                            trend += " | 👻 Ghost Blocked"

                    if s.get("mtf_confirm") and signal != "WAIT" and strategy != "TradingView Webhook":
                        df_htf = self.get_historical_data(exch, token, symbol=index, interval="15m") if not self.is_mock else self.get_historical_data("MOCK", "12345", symbol=index, interval="15m")
                        if df_htf is not None and len(df_htf) > 5:
                            htf_ema = df_htf['close'].ewm(span=9).mean().iloc[-1]
                            htf_close = df_htf['close'].iloc[-1]
                            if signal == "BUY_CE" and htf_close < htf_ema: signal = "WAIT"; trend = "MTF Blocked: 15m Bearish"
                            elif signal == "BUY_PE" and htf_close > htf_ema: signal = "WAIT"; trend = "MTF Blocked: 15m Bullish"

                    is_hz = s.get("hero_zero")
                    if is_hz and signal != "WAIT" and strategy != "TradingView Webhook":
                        if not self.is_mock and not (is_mt5_asset or is_crypto):
                            live_oi, live_vol = self.get_market_data_oi(exch, token)
                            if live_vol < 50000: signal, trend = "WAIT", "Hero/Zero Blocked: Low Volume/OI"
                        
                        greek_pass, greek_msg = self.analyze_oi_and_greeks(df_candles, is_hz, signal)
                        if not greek_pass: signal = "WAIT"; trend = greek_msg
                        else: trend += f" | {greek_msg}"

                else:
                    trend, signal, vwap, ema, df_chart, current_atr, fib_data = "Waiting for Market Data", "WAIT", 0, 0, df_candles, 0, {}

                self.state.update({"current_trend": trend, "current_signal": signal, "vwap": vwap, "ema": ema, "atr": current_atr, "fib_data": fib_data, "latest_data": df_chart})

                # ---- EXECUTION ----
                if self.state["active_trade"] is None and signal in ["BUY_CE", "BUY_PE"] and current_time < cutoff_time:
                    
                    qty = max(float(s['lots']), float(base_lot_size))
                    
                    if is_mt5_asset or (is_crypto and s.get('crypto_mode') != "Options"):
                        strike_sym = index
                        if is_crypto and s.get('crypto_mode') == "Futures":
                            if exch == "DELTA" and not strike_sym.endswith("USD"):
                                strike_sym = f"{strike_sym}USD" 
                        strike_token, strike_exch = strike_sym, exch
                        entry_ltp = spot
                    else:
                        max_prem = s['max_capital'] / qty if qty > 0 else 0
                        strike_sym, strike_token, strike_exch, entry_ltp = self.get_strike(index, spot, signal, max_prem)
                    
                    if strike_sym and entry_ltp:
                        trade_type = "CE" if signal == "BUY_CE" else "PE"
                        if is_mt5_asset or is_crypto: trade_type = "BUY" if signal == "BUY_CE" else "SELL"

                        # Ensure correct TP/SL direction for MT5/Crypto Shorts
                        if trade_type == "SELL":
                            dynamic_sl = entry_ltp + s['sl_pts']
                            tp1 = entry_ltp - s['tgt_pts']
                            tp2 = entry_ltp - (s['tgt_pts'] * 2)
                            tp3 = entry_ltp - (s['tgt_pts'] * 3)
                        else:
                            dynamic_sl = entry_ltp - s['sl_pts'] 
                            tp1 = entry_ltp + s['tgt_pts']
                            tp2 = entry_ltp + (s['tgt_pts'] * 2)
                            tp3 = entry_ltp + (s['tgt_pts'] * 3)

                        new_trade = {
                            "symbol": strike_sym, "token": strike_token, "exch": strike_exch, 
                            "type": trade_type, "entry": entry_ltp, 
                            "highest_price": entry_ltp, "lowest_price": entry_ltp, "qty": qty, "sl": dynamic_sl, 
                            "tp1": tp1, "tp2": tp2, "tp3": tp3, "tgt": tp3,
                            "scaled_out": False, "is_hz": s.get("hero_zero", False)
                        }

                        if not is_mock_mode: 
                            # 🔥 FIX: For option buying (CE or PE), we always BUY to open! 
                            exec_side = "SELL" if new_trade['type'] == "SELL" else "BUY"
                            self.place_real_order(strike_sym, strike_token, qty, exec_side, strike_exch)
                            
                        self.push_notify("Trade Entered", f"Entered {qty} {strike_sym} @ {entry_ltp}")
                        self.state["active_trade"] = new_trade
                        self.state["trades_today"] += 1
                        self.state["ghost_memory"][f"{index}_{signal}"] = get_ist()
                    
                    elif not is_mock_mode:
                        self.log(f"⚠️ Trade Blocked: Failed to fetch valid Strike/Premium for {index}.")

                elif self.state["active_trade"]:
                    trade = self.state["active_trade"]
                    
                    if not self.is_mock or (self.is_mock and trade['token'] != "12345" and self.api):
                        ltp = self.get_live_price(trade['exch'], trade['symbol'], trade['token'])
                        if ltp is None and self.is_mock:
                            delta = (spot - self.state["spot"]) * (0.5 if trade['type'] in ["CE", "BUY"] else -0.5)
                            ltp = trade['entry'] + delta + np.random.uniform(-1, 2)
                    else:
                        delta = (spot - self.state["spot"]) * (0.5 if trade['type'] in ["CE", "BUY"] else -0.5) 
                        ltp = trade['entry'] + delta + np.random.uniform(-1, 2)
                        
                    if ltp:
                        if trade['type'] == "SELL":
                            pnl = (trade['entry'] - ltp) * trade['qty']
                        else:
                            pnl = (ltp - trade['entry']) * trade['qty']
                            
                        if is_mt5_asset: pnl = pnl * 100000 if "USD" in trade['symbol'] else pnl 

                        self.state["active_trade"]["current_ltp"] = ltp
                        self.state["active_trade"]["floating_pnl"] = pnl
                        
                        # Trailing Stoploss logic specific to trade direction
                        if trade['type'] == "SELL":
                            lowest = trade.get('lowest_price', trade['entry'])
                            if ltp < lowest:
                                trade['lowest_price'] = ltp
                                tsl_buffer = s['tsl_pts'] * 1.5 if "Trend Rider" in strategy else s['tsl_pts']
                                new_sl = ltp + tsl_buffer
                                if new_sl < trade['sl']: trade['sl'] = new_sl
                                
                            hit_tp = False if ("Trend Rider" in strategy) else (ltp <= trade['tgt'])
                            hit_sl = ltp >= trade['sl']
                        else:
                            highest = trade.get('highest_price', trade['entry'])
                            if ltp > highest:
                                trade['highest_price'] = ltp
                                
                                if trade.get('is_hz'):
                                    if ltp >= trade['entry'] * 3.0:    new_sl = ltp * 0.85 
                                    elif ltp >= trade['entry'] * 2.0:  new_sl = ltp * 0.80 
                                    elif ltp >= trade['entry'] * 1.5:  new_sl = trade['entry'] * 1.10 
                                    else:                              new_sl = trade['sl']
                                    if new_sl > trade['sl']: trade['sl'] = new_sl
                                else:
                                    tsl_buffer = s['tsl_pts'] * 1.5 if "Trend Rider" in strategy else s['tsl_pts']
                                    new_sl = ltp - tsl_buffer
                                    if new_sl > trade['sl']: trade['sl'] = new_sl
                                    
                            hit_tp = False if ("Trend Rider" in strategy and not trade.get('is_hz')) else (ltp >= trade['tgt'])
                            hit_sl = ltp <= trade['sl']

                        # Partial Target Hits
                        if not trade['scaled_out'] and not trade.get('is_hz'):
                            reach_tp1 = (ltp <= trade['tp1']) if trade['type'] == "SELL" else (ltp >= trade['tp1'])
                            if reach_tp1:
                                if index in ["NIFTY", "SENSEX", "XAUUSD"] or is_crypto:
                                    lots_held = trade['qty'] / base_lot_size
                                    half_lots = int(lots_held / 2) if not (is_mt5_asset or is_crypto) else round(trade['qty']/2, 2)
                                    if half_lots > 0:
                                        qty_to_sell = half_lots * base_lot_size if not (is_mt5_asset or is_crypto) else half_lots
                                        if not is_mock_mode:
                                            # 🔥 FIX: Inverse order exactly aligns execution with partial close requirements
                                            exec_side = "BUY" if trade['type'] == "SELL" else "SELL"
                                            self.place_real_order(trade['symbol'], trade['token'], qty_to_sell, exec_side, trade['exch'])
                                        trade['qty'] -= qty_to_sell
                                        trade['scaled_out'] = True
                                        trade['sl'] = trade['entry'] 
                                        self.log(f"💥 PARTIAL BOOKED 50% at {ltp}. SL trailed to BE.")
                                        self.push_notify("Partial Profit", f"Booked 50% of {trade['symbol']}. Remainder risk-free.")
                        
                        market_close = current_time >= cutoff_time
                        
                        if self.state.get("manual_exit"):
                            hit_tp, market_close = True, True
                            self.state["manual_exit"] = False
                        
                        # Full Exit Target / SL Hits
                        if hit_tp or hit_sl or market_close:
                            if not is_mock_mode: 
                                # 🔥 FIX: Option Longs close by SELLING. Crypto Shorts close by BUYING.
                                exec_side = "BUY" if trade['type'] == "SELL" else "SELL"
                                self.place_real_order(trade['symbol'], trade['token'], trade['qty'], exec_side, trade['exch'])
                            
                            win_text = "profit👍" if pnl > 0 else "sl hit 🛑"
                            if trade['type'] == "SELL":
                                lowest_reached = trade.get('lowest_price', trade['entry'])
                                if lowest_reached <= trade['tp3']: win_text = "tp3❤"
                                elif lowest_reached <= trade['tp2']: win_text = "tp2✔✔"
                                elif lowest_reached <= trade['tp1']: win_text = "tp1✔"
                            else:
                                highest_reached = trade.get('highest_price', trade['entry'])
                                if highest_reached >= trade['tp3']: win_text = "tp3❤"
                                elif highest_reached >= trade['tp2']: win_text = "tp2✔✔"
                                elif highest_reached >= trade['tp1']: win_text = "tp1✔"
                            
                            if market_close: win_text += " (Force Exit 3:15)"
                            if trade['scaled_out']: win_text += " (Scaled Out)"
                            
                            self.log(f"🛑 EXIT {trade['symbol']} | PnL: {round(pnl, 2)} [{win_text}]")
                            self.push_notify("Trade Closed", f"Closed {trade['symbol']} | PnL: {round(pnl, 2)}")
                            
                            if not self.is_mock: 
                                user_id = getattr(self, "system_user_id", self.api_key)
                                save_trade(user_id, today_date, time_str, trade['symbol'], trade['type'], trade['qty'], trade['entry'], ltp, round(pnl, 2), win_text)
                            else:
                                if "paper_history" not in self.state: self.state["paper_history"] = []
                                self.state["paper_history"].append({
                                    "Date": today_date, "Time": time_str, "Symbol": trade['symbol'],
                                    "Type": trade['type'], "Qty": trade['qty'], "Entry Price": trade['entry'],
                                    "Exit Price": ltp, "PnL": round(pnl, 2), "Result": win_text
                                })
                            
                            self.state["last_trade"] = trade.copy()
                            self.state["last_trade"].update({"exit_price": ltp, "final_pnl": pnl, "win_text": win_text})
                            self.state["daily_pnl"] += pnl
                            self.state["active_trade"] = None
            except Exception as e:
                self.log(f"⚠️ Loop Error: {str(e)}")
            time.sleep(2)

# ==========================================
# 5. STREAMLIT UI 
# ==========================================
is_mkt_open, mkt_status_msg = get_market_status(st.session_state.sb_index_input)

def play_sound():
    components.html("""<audio autoplay><source src="https://media.geeksforgeeks.org/wp-content/uploads/20190531135120/beep.mp3" type="audio/mpeg"></audio>""", height=0)

if getattr(st.session_state, "bot", None) and st.session_state.bot.state.get("ui_popups"):
    play_sound()
    while st.session_state.bot.state["ui_popups"]:
        alert = st.session_state.bot.state["ui_popups"].popleft()
        st.toast(alert.get("message", ""), icon="🔔")

# --- LOGIN SCREEN ---
if not getattr(st.session_state, "bot", None):
    if not HAS_DB: st.error("⚠️ Database missing. Add SUPABASE_URL and SUPABASE_KEY to enable saving & logs.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    spacer1, login_col, spacer2 = st.columns([1, 1.5, 1])
    
    with login_col:
        st.markdown("""
            <div style='text-align: center; background: linear-gradient(135deg, #0f111a, #0284c7); padding: 30px; border-radius: 4px 4px 0 0; border-bottom: none;'>
                <h1 style='color: white; margin:0; font-weight: 900; letter-spacing: 2px; font-size: 2.2rem;'>🕉️ SHREE</h1>
                <p style='color: #bae6fd; margin-top:5px; font-size: 1rem; font-weight: 600; letter-spacing: 1px;'>SECURE MULTI-BROKER GATEWAY</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
            auth_mode = st.radio("Operating Mode", ["📝 Paper Trading", "🕉️ Real Trading", "👆 Quick Auth"], horizontal=True, label_visibility="collapsed")
            st.divider()
            
            if auth_mode == "👆 Quick Auth":
                st.info("💡 **Quick Login:** Enter your registered Email or Phone. The system will auto-fetch your Cloud profile.")
                USER_ID = st.text_input("Enter Email ID or Phone Number")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("👆 Authenticate & Connect", type="primary", use_container_width=True):
                    creds = load_creds(USER_ID)
                    if creds and (creds.get("client_id") or creds.get("zerodha_api") or creds.get("coindcx_api") or creds.get("delta_api")):
                        temp_bot = SniperBot(
                            api_key=creds.get("angel_api", ""), client_id=creds.get("client_id"), pwd=creds.get("pwd"), 
                            totp_secret=creds.get("totp_secret"), mt5_acc=creds.get("mt5_acc"), 
                            mt5_pass=creds.get("mt5_pass"), mt5_server=creds.get("mt5_server"),
                            zerodha_api=creds.get("zerodha_api"), zerodha_secret=creds.get("zerodha_secret"),
                            coindcx_api=creds.get("coindcx_api"), coindcx_secret=creds.get("coindcx_secret"),
                            delta_api=creds.get("delta_api"), delta_secret=creds.get("delta_secret"),
                            is_mock=False
                        )
                        temp_bot.system_user_id = USER_ID
                        with st.spinner("Authenticating via Cloud..."):
                            if temp_bot.login():
                                st.session_state.bot = temp_bot
                                st.rerun()
                            else: st.error("❌ Login Failed! Check API details or TOTP.")
                    else: st.error("❌ Profile not found! Please save it once via the Real Trading menu.")
                        
            elif auth_mode == "🕉️ Real Trading":
                USER_ID = st.text_input("System Login ID (Email or Phone Number)")
                creds = load_creds(USER_ID) if USER_ID else {}

                st.markdown("### 🏦 Select Brokers to Connect")
                
                ANGEL_API, CLIENT_ID, PIN, TOTP = "", "", "", ""
                Z_API, Z_SEC, Z_REQ = "", "", ""
                MT5_ACC, MT5_PASS, MT5_SERVER = "", "", ""
                DCX_API, DCX_SEC = "", ""
                DELTA_API, DELTA_SEC = "", ""

                with st.container(border=True):
                    col_img, col_t = st.columns([1, 6])
                    with col_img: st.image("https://www.google.com/s2/favicons?domain=angelone.in&sz=128", width=40)
                    with col_t: use_angel = st.toggle("Angel One India", value=bool(creds.get("client_id")))
                    if use_angel:
                        ANGEL_API = st.text_input("Angel API Key", value=creds.get("angel_api", ""))
                        col_id, col_pin = st.columns(2)
                        with col_id: CLIENT_ID = st.text_input("Client ID", value=creds.get("client_id", ""))
                        with col_pin: PIN = st.text_input("PIN", value=creds.get("pwd", ""), type="password")
                        TOTP = st.text_input("TOTP Secret", value=creds.get("totp_secret", ""), type="password")

                with st.container(border=True):
                    col_img, col_t = st.columns([1, 6])
                    with col_img: st.image("https://www.google.com/s2/favicons?domain=zerodha.com&sz=128", width=40)
                    with col_t: use_zerodha = st.toggle("Zerodha Kite", value=bool(creds.get("zerodha_api")))
                    if use_zerodha:
                        Z_API = st.text_input("Kite API Key", value=creds.get("zerodha_api", ""))
                        Z_SEC = st.text_input("Kite API Secret", type="password", value=creds.get("zerodha_secret", ""))
                        Z_REQ = st.text_input("Today's Request Token", type="password")

                with st.container(border=True):
                    col_img, col_t = st.columns([1, 6])
                    with col_img: st.image("https://www.google.com/s2/favicons?domain=coindcx.com&sz=128", width=40)
                    with col_t: use_coindcx = st.toggle("CoinDCX Crypto", value=bool(creds.get("coindcx_api")))
                    if use_coindcx:
                        DCX_API = st.text_input("CoinDCX API Key", value=creds.get("coindcx_api", ""))
                        DCX_SEC = st.text_input("CoinDCX API Secret", type="password", value=creds.get("coindcx_secret", ""))

                with st.container(border=True):
                    col_img, col_t = st.columns([1, 6])
                    with col_img: st.image("https://www.google.com/s2/favicons?domain=delta.exchange&sz=128", width=40)
                    with col_t: use_delta = st.toggle("Delta Exchange", value=bool(creds.get("delta_api")))
                    if use_delta:
                        DELTA_API = st.text_input("Delta API Key", value=creds.get("delta_api", ""))
                        DELTA_SEC = st.text_input("Delta API Secret", type="password", value=creds.get("delta_secret", ""))

                with st.container(border=True):
                    col_img, col_t = st.columns([1, 6])
                    with col_img: st.image("https://www.google.com/s2/favicons?domain=metatrader5.com&sz=128", width=40)
                    with col_t: use_mt5 = st.toggle("MetaTrader 5 (MT5)", value=bool(creds.get("mt5_acc")))
                    if use_mt5:
                        if not HAS_MT5:
                            st.error("⚠️ MetaTrader5 module is strictly Windows-only. It will not run on Termux or Cloud Servers.")
                        else:
                            col_m1, col_m2 = st.columns(2)
                            with col_m1: MT5_ACC = st.text_input("MT5 Account ID", value=creds.get("mt5_acc", ""))
                            with col_m2: MT5_PASS = st.text_input("MT5 Password", type="password", value=creds.get("mt5_pass", ""))
                            MT5_SERVER = st.text_input("Broker Server (e.g. XMGlobal-MT5 or BTCDana-Live)", value=creds.get("mt5_server", ""))
                
                st.divider()
                with st.expander("📱 Notifications (Telegram/WhatsApp)"):
                    TG_TOKEN = st.text_input("Telegram Bot Token", value=creds.get("tg_token", ""))
                    TG_CHAT = st.text_input("Telegram Chat ID", value=creds.get("tg_chat", ""))
                    WA_PHONE = st.text_input("WhatsApp Phone", value=creds.get("wa_phone", ""))
                    WA_API = st.text_input("WhatsApp API Key", value=creds.get("wa_api", ""))

                SAVE_CREDS = st.checkbox("Remember Credentials Securely (Cloud DB)", value=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("CONNECT MARKETS 🚀", type="primary", use_container_width=True):
                    if not USER_ID: st.error("Please enter your System Login ID (Email or Phone) to proceed.")
                    else:
                        temp_bot = SniperBot(
                            api_key=ANGEL_API if use_angel else "", client_id=CLIENT_ID if use_angel else "", 
                            pwd=PIN if use_angel else "", totp_secret=TOTP if use_angel else "", 
                            tg_token=TG_TOKEN, tg_chat=TG_CHAT, wa_phone=WA_PHONE, wa_api=WA_API, 
                            mt5_acc=MT5_ACC if use_mt5 else "", mt5_pass=MT5_PASS if use_mt5 else "", 
                            mt5_server=MT5_SERVER if use_mt5 else "",
                            zerodha_api=Z_API if use_zerodha else "", zerodha_secret=Z_SEC if use_zerodha else "", 
                            request_token=Z_REQ if use_zerodha else "",
                            coindcx_api=DCX_API if use_coindcx else "", coindcx_secret=DCX_SEC if use_coindcx else "",
                            delta_api=DELTA_API if use_delta else "", delta_secret=DELTA_SEC if use_delta else "",
                            is_mock=False
                        )
                        temp_bot.system_user_id = USER_ID
                        with st.spinner("Authenticating Secure Connections..."):
                            if temp_bot.login():
                                if SAVE_CREDS: save_creds(USER_ID, ANGEL_API, CLIENT_ID, PIN, TOTP, TG_TOKEN, TG_CHAT, WA_PHONE, WA_API, MT5_ACC, MT5_PASS, MT5_SERVER, Z_API, Z_SEC, DCX_API, DCX_SEC, DELTA_API, DELTA_SEC)
                                st.session_state.bot = temp_bot
                                st.rerun()
                            else:
                                err_msg = temp_bot.state['logs'][0] if temp_bot.state['logs'] else "Unknown Error"
                                st.error(f"Login Failed! \n\n**System Log:** {err_msg}")
            else:
                st.info("📝 Paper Trading simulates live market movement without risking real capital.")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("START PAPER SESSION 📝", type="primary", use_container_width=True):
                    temp_bot = SniperBot(is_mock=True)
                    temp_bot.login()
                    st.session_state.bot = temp_bot
                    st.rerun()
                    
            st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN TERMINAL ---
else:
    bot = st.session_state.bot
    
    # --- 1. TOP HEADER & LOGOUT ---
    head_c1, head_c2 = st.columns([3, 1])
    with head_c1: 
        st.markdown(f"**👤 Session:** <span style='color:#0284c7; font-weight:800;'>{bot.client_name}</span> | **IP:** `{bot.client_ip}`", unsafe_allow_html=True)
    
    with head_c2:
        st.markdown("""
        <style>
        div[data-testid="column"]:nth-of-type(2) button {
            background: linear-gradient(135deg, #ff416c, #ff4b2b) !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important; 
            font-weight: 900 !important;
            box-shadow: 0 4px 15px rgba(255, 65, 108, 0.5) !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="column"]:nth-of-type(2) button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 6px 20px rgba(255, 65, 108, 0.7) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🚪 LOGOUT", use_container_width=True):
            bot.state["is_running"] = False
            st.session_state.clear()
            st.rerun()

    # --- 2. THE SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ SYSTEM CONFIGURATION")
        
        st.markdown("**1. Market Setup**")
        BROKER = st.selectbox("Primary Broker", ["Angel One", "Zerodha", "CoinDCX", "Delta Exchange", "MT5"], index=0)
        
        if 'user_lots' not in st.session_state: st.session_state.user_lots = DEFAULT_LOTS.copy()
        
        CUSTOM_STOCK = st.text_input("Add Custom Stock/Coin", value=st.session_state.custom_stock, placeholder="e.g. RELIANCE or BTCUSDT").upper().strip()
        st.session_state.custom_stock = CUSTOM_STOCK
        
        all_assets = list(st.session_state.user_lots.keys())
        if CUSTOM_STOCK and CUSTOM_STOCK not in all_assets:
            all_assets.append(CUSTOM_STOCK)
            st.session_state.user_lots[CUSTOM_STOCK] = 0.01 if len(CUSTOM_STOCK) == 6 or "USD" in CUSTOM_STOCK else 1 
        
        if BROKER in ["CoinDCX", "Delta Exchange"]:
            crypto_pairs = get_all_crypto_pairs()
            valid_assets = [a for a in all_assets if ("USD" in a or "USDT" in a or "INR" in a)]
            for p in crypto_pairs:
                if p not in valid_assets:
                    valid_assets.append(p)
                if p not in st.session_state.user_lots:
                    st.session_state.user_lots[p] = 1.0 
        elif BROKER in ["Angel One", "Zerodha"]:
            valid_assets = [a for a in all_assets if a in ["NIFTY", "BANKNIFTY", "SENSEX", "CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "INDIA VIX"] or ("USD" not in a and "USDT" not in a)]
        else: # MT5
            valid_assets = [a for a in all_assets if a in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "SOLUSD"] or "USD" in a]
            
        if CUSTOM_STOCK and CUSTOM_STOCK not in valid_assets: valid_assets.append(CUSTOM_STOCK)
        if not valid_assets: valid_assets = ["NIFTY"] if BROKER in ["Angel One", "Zerodha"] else ["BTCUSD"]
        
        st.session_state.asset_options = valid_assets
        if st.session_state.sb_index_input not in valid_assets: st.session_state.sb_index_input = valid_assets[0]
        
        INDEX = st.selectbox("Watchlist Asset", valid_assets, index=valid_assets.index(st.session_state.sb_index_input), key="sb_index_input")
        STRATEGY = st.selectbox("Trading Strategy", STRAT_LIST, index=STRAT_LIST.index(st.session_state.sb_strat_input), key="sb_strat_input")
        TIMEFRAME = st.selectbox("Candle Timeframe", ["1m", "3m", "5m", "15m"], index=2)
        
        # --- DYNAMIC KEYWORD BUILDER ---
        CUSTOM_CODE = ""
        TV_PASSPHRASE = "SHREE123"
        
        if STRATEGY == "Keyword Rule Builder":
            st.divider()
            st.markdown("**🧠 Keyword Logic Builder**")
            st.info("Select triggers. If multiple are chosen, ALL must be true to enter a trade.")
            selected_rules = st.multiselect(
                "Select Technical Conditions",
                ["EMA Crossover (9 & 21)", "Bollinger Bands Bounce", "RSI Breakout (>60/<40)", "MACD Crossover"],
                default=["EMA Crossover (9 & 21)"]
            )
            CUSTOM_CODE = ",".join(selected_rules)
            st.session_state.custom_code_input = CUSTOM_CODE

        elif STRATEGY == "TradingView Webhook":
            st.divider()
            st.markdown("**📡 TradingView Integration**")
            if not HAS_FLASK:
                st.error("Flask is required for webhooks. Please run `pip install Flask`.")
            else:
                st.success(f"Webhook URL: `http://{bot.client_ip}:5000/tv_webhook`")
                TV_PASSPHRASE = st.text_input("Webhook Passphrase", value="SHREE123")
                st.code(f"""// Example TradingView JSON Alert
{{
    "passphrase": "{TV_PASSPHRASE}",
    "action": "BUY_CE", // or BUY_PE
    "symbol": "{INDEX}" // Must match app asset
}}""", language="json")

        if BROKER in ["CoinDCX", "Delta Exchange"]:
            st.divider()
            st.markdown("**🪙 Crypto Market Setup**")
            col_c1, col_c2 = st.columns(2)
            with col_c1: CRYPTO_MODE = st.selectbox("Market Type", ["Futures", "Spot", "Options"])
            with col_c2: LEVERAGE = st.number_input("Leverage (x)", min_value=1, max_value=100, value=10, step=1)
            SHOW_INR_CRYPTO = st.toggle("Convert Displays to ₹ INR", True)
        else:
            CRYPTO_MODE = "Options"
            LEVERAGE = 1
            SHOW_INR_CRYPTO = False

        st.divider()
        st.markdown("**2. Risk Management**")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            default_lot_val = float(st.session_state.user_lots.get(INDEX, 1.0))
            LOTS = st.number_input("Base Lots / Vol", 0.01, 10000.0, value=default_lot_val, step=0.01, key=f"lot_input_{INDEX}")
            MAX_TRADES = st.number_input("Max Trades/Day", 1, 50, 5)
            MAX_CAPITAL = st.number_input("Max Cap/Trade (₹/$)", 10.0, 500000.0, 15000.0, step=100.0)
        with col_r2:
            SL_PTS = st.number_input("SL Points/Pips", 5.0, 500.0, 20.0)
            TSL_PTS = st.number_input("Trail SL Pts/Pips", 5.0, 500.0, 15.0)
            TGT_PTS = st.number_input("Target Steps/Pips", 5.0, 1000.0, 15.0)
        CAPITAL_PROTECT = st.number_input("Capital Protection (Max Loss)", 500.0, 500000.0, 2000.0, step=500.0)
        
        st.divider()
        st.markdown("**3. Advanced Triggers**")
        MTF_CONFIRM = st.toggle("⏱️ Multi-TF Confirmation", False)
        HERO_ZERO = st.toggle("🚀 Hero/Zero Setup (Gamma Tracker)", False)
        FOMO_ENTRY = st.toggle("🚨 FOMO Momentum Entry", False)

        st.divider()
        if not bot.is_mock and st.button("🧪 Ping API Connection", use_container_width=True):
            st.toast("Testing exact API parameters...", icon="🧪")
            bot.log(f"🧪 User executed manual Ping API Connection for {BROKER}.")

        render_signature()

    bot.settings = {
        "primary_broker": BROKER, "strategy": STRATEGY, "index": INDEX, "timeframe": TIMEFRAME, 
        "lots": LOTS, "max_trades": MAX_TRADES, "max_capital": MAX_CAPITAL, "capital_protect": CAPITAL_PROTECT, 
        "sl_pts": SL_PTS, "tsl_pts": TSL_PTS, "tgt_pts": TGT_PTS, "paper_mode": bot.is_mock, 
        "mtf_confirm": MTF_CONFIRM, "hero_zero": HERO_ZERO, "fomo_entry": FOMO_ENTRY, 
        "crypto_mode": CRYPTO_MODE, "leverage": LEVERAGE, "show_inr_crypto": SHOW_INR_CRYPTO,
        "user_lots": st.session_state.user_lots.copy(),
        "custom_code": CUSTOM_CODE, "tv_passphrase": TV_PASSPHRASE 
    }

    if bot.state['latest_data'] is None or st.session_state.prev_index != INDEX:
        st.session_state.prev_index = INDEX
        if bot.state.get("is_running"): bot.state["spot"] = 0.0 
        else:
            with st.spinner(f"Fetching Live Market Data for {INDEX}..."):
                exch, token = bot.get_token_info(INDEX)
                df_preload = bot.get_historical_data(exch, token, symbol=INDEX, interval=TIMEFRAME) if not bot.is_mock else bot.get_historical_data("MOCK", "12345", symbol=INDEX, interval=TIMEFRAME)
                if df_preload is not None and not df_preload.empty:
                    bot.state["spot"] = df_preload['close'].iloc[-1]
                    bot.state["latest_candle"] = df_preload.iloc[-1].to_dict()
                    if STRATEGY == "Keyword Rule Builder": t, s, v, e, df_c, atr, fib = bot.analyzer.apply_keyword_strategy(df_preload, CUSTOM_CODE, INDEX)
                    elif STRATEGY == "TradingView Webhook": t, s, v, e, df_c, atr, fib = "Awaiting TradingView Webhook...", "WAIT", df_preload['close'].iloc[-1], df_preload['close'].iloc[-1], df_preload, 0, {}
                    elif "VIJAY & RFF" in STRATEGY: t, s, v, e, df_c, atr, fib = bot.analyzer.apply_vijay_rff_strategy(df_preload, INDEX)
                    elif "Institutional FVG" in STRATEGY or "ICT" in STRATEGY: t, s, v, e, df_c, atr, fib = bot.analyzer.apply_ict_smc_strategy(df_preload, INDEX)
                    elif "Trend Rider" in STRATEGY: t, s, v, e, df_c, atr, fib = bot.analyzer.apply_trend_rider_strategy(df_preload, INDEX)
                    else: t, s, v, e, df_c, atr, fib = bot.analyzer.apply_vwap_ema_strategy(df_preload, INDEX)
                    bot.state.update({"current_trend": t, "current_signal": s, "vwap": v, "ema": e, "atr": atr, "fib_data": fib, "latest_data": df_c})

    if not is_mkt_open: 
        st.error(f"🛑 {mkt_status_msg} - Engine will standby until market opens.")
        
    tab1, tab2, tab3, tab4 = st.tabs(["🕉️ DASHBOARD", "🔎 SCANNERS", "📜 LOGS", "🚀 CRYPTO/FX"])

    with tab1:
        exch, _ = bot.get_token_info(INDEX)
        if exch == "MT5": term_type = "🌍 MT5 Forex Terminal"
        elif exch == "COINDCX": term_type = f"🕉️ CoinDCX {CRYPTO_MODE}"
        elif exch == "DELTA": term_type = f"🔺 Delta Exchange {CRYPTO_MODE}"
        else: term_type = f"🇮🇳 {BROKER} NSE/NFO"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0284c7, #0369a1); padding: 18px; border-radius: 4px; border: 1px solid #e2e8f0; color: white; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h2 style="margin: 0; color: #ffffff; font-weight: 800; letter-spacing: 1px;">🕉️ {INDEX}</h2>
                <p style="margin: 5px 0 0 0; font-size: 0.95rem; color: #e0f2fe; font-weight: 700;">{term_type}</p>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed rgba(255,255,255,0.3);">
                    <span style="font-size: 0.85rem; color: #f8fafc;">Live Balance:</span><br>
                    <span style="font-size: 1.2rem; font-weight: bold; color: #ffffff;">{bot.get_balance()}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        is_running = bot.state["is_running"]
        status_color = "#22c55e" if is_running else "#ef4444"
        status_bg = "#f0fdf4" if is_running else "#fef2f2"
        status_text = f"🟢 ENGINE ACTIVE ({bot.state['trades_today']}/{MAX_TRADES} Trades)" if is_running else "🛑 ENGINE STOPPED"
        
        st.markdown(f"""
            <div style="text-align: center; padding: 10px; border-radius: 4px; background-color: {status_bg}; border: 1.5px solid {status_color}; color: {status_color}; font-weight: 800; font-size: 0.95rem; margin-bottom: 15px; letter-spacing: 0.5px;">
                {status_text}
            </div>
        """, unsafe_allow_html=True)

        c1, c2, c_kill = st.columns([2, 2, 1])
        with c1:
            if st.button("▶️ FIRE ENGINE", use_container_width=True, type="primary", disabled=is_running):
                bot.state["is_running"] = True
                t = threading.Thread(target=bot.trading_loop, daemon=True)
                add_script_run_ctx(t)
                t.start()
                st.rerun()
        with c2:
            if st.button("🛑 HALT ENGINE", use_container_width=True, disabled=not is_running):
                bot.state["is_running"] = False
                st.rerun()
        with c_kill:
            if st.button("☠️", use_container_width=True):
                bot.state["is_running"] = False
                if bot.state["active_trade"]: bot.state["manual_exit"] = True
                st.toast("System Terminated & Trades Closed", icon="☠️")

        ltp_val = round(bot.state['spot'], 4)
        trend_val = bot.state['current_trend']
        
        fib_d = bot.state.get('fib_data', {})
        gz_l = round(fib_d.get('fib_low', 0), 2)
        gz_h = round(fib_d.get('fib_high', 0), 2)
        gz_display = f"{gz_l} - {gz_h}" if gz_l > 0 else "Calculating..."
        
        currency_sym = "$" if exch in ["MT5", "DELTA", "COINDCX"] else "₹"
        if exch in ["DELTA", "COINDCX"] and SHOW_INR_CRYPTO:
            inr_val = ltp_val * get_usdt_inr_rate()
            ltp_display = f"{currency_sym}{ltp_val} (₹ {round(inr_val, 2)})"
        else:
            ltp_display = f"{currency_sym}{ltp_val}"
            
        last_candle = bot.state.get("latest_candle")
        if last_candle is not None:
            o_val = last_candle.get('open', 0.0)
            h_val = last_candle.get('high', 0.0)
            l_val = last_candle.get('low', 0.0)
            c_val = last_candle.get('close', 0.0)
            v_val = last_candle.get('volume', 0.0)
            if c_val > o_val: vol_dom = "Buy Volume Dominant 🟢"
            elif c_val < o_val: vol_dom = "Sell Volume Dominant 🔴"
            else: vol_dom = "Neutral Volume ⚪"
            v_display = f"{round(v_val, 2)} ({vol_dom})"
        else:
            o_val, h_val, l_val, c_val, v_display = 0.0, 0.0, 0.0, 0.0, "N/A"
        
        st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; margin-bottom: 20px;">
                <div style="background: #ffffff; padding: 15px; border-radius: 4px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 800; letter-spacing: 1px;">Live Spot</div>
                    <div style="font-size: 1.4rem; color: #0f111a; font-weight: 900; margin-top: 4px;">{ltp_display}</div>
                </div>
                <div style="background: #ffffff; padding: 15px; border-radius: 4px; border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 800; letter-spacing: 1px;">Fib Golden Zone</div>
                    <div style="font-size: 1.1rem; color: #0f111a; font-weight: 900; margin-top: 4px;">{gz_display}</div>
                </div>
                <div style="background: #ffffff; padding: 15px; border-radius: 4px; border: 1px solid #e2e8f0; text-align: center; grid-column: span 2; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 800; letter-spacing: 1px;">Algorithm Sentiment</div>
                    <div style="font-size: 1.2rem; color: #0284c7; font-weight: 900; margin-top: 4px;">{trend_val}</div>
                </div>
                <div style="background: #ffffff; padding: 15px; border-radius: 4px; border: 1px solid #e2e8f0; text-align: center; grid-column: span 2; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 800; letter-spacing: 1px;">Live OHLCV</div>
                    <div style="font-size: 1.1rem; color: #0f111a; font-weight: 900; margin-top: 4px;">
                        O: {round(o_val, 4)} &nbsp;|&nbsp; H: {round(h_val, 4)} &nbsp;|&nbsp; L: {round(l_val, 4)} &nbsp;|&nbsp; C: {round(c_val, 4)}<br>
                        <span style="font-size: 0.95rem; color: #0284c7;">V: {v_display}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Live Position Tracker")
        daily_pnl = bot.state.get("daily_pnl", 0.0)
        st.markdown(f"**Session Net Yield:** {'<span style="color:#22c55e">🟢' if daily_pnl >= 0 else '<span style="color:#ef4444">🔴'} `{round(daily_pnl, 2)}`</span>", unsafe_allow_html=True)
        
        if bot.state["active_trade"]:
            t = bot.state["active_trade"]
            ltp = t.get('current_ltp', t['entry'])
            pnl = t.get('floating_pnl', 0.0)
            pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
            pnl_bg = "#f0fdf4" if pnl >= 0 else "#fef2f2"
            pnl_sign = "+" if pnl >= 0 else ""
            exec_type = t['exch']
            buy_sell_color = "#22c55e" if t['type'] in ["CE", "BUY"] else "#ef4444"
            
            pnl_display = round(pnl, 2)
            if t['exch'] in ["DELTA", "COINDCX"] and SHOW_INR_CRYPTO:
                inr_pnl = pnl * get_usdt_inr_rate()
                pnl_display = f"{pnl_sign}{round(pnl, 2)} (₹ {round(inr_pnl, 2)})"
            
            st.markdown(f"""
                <div style="background: #ffffff; border: 2px solid {pnl_color}; border-radius: 4px; padding: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px dashed #e2e8f0; padding-bottom: 12px; margin-bottom: 12px;">
                        <div>
                            <span style="background: {buy_sell_color}; color: white; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; font-weight: 800; letter-spacing: 1px;">{t['type']}</span>
                            <strong style="margin-left: 10px; font-size: 1.1rem; color: #0f111a;">{t['symbol']}</strong>
                        </div>
                        <div style="background: {pnl_bg}; color: {pnl_color}; padding: 6px 12px; border-radius: 4px; font-weight: 900; font-size: 1.4rem; border: 1px solid {pnl_color};">
                            {pnl_display}
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 15px;">
                        <div style="background: #f8fafc; padding: 10px; border-radius: 4px;">
                            <span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">Avg Entry</span><br>
                            <b style="font-size: 1.1rem; color: #0f111a;">{t['entry']:.4f}</b>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 4px;">
                            <span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">Live Mark</span><br>
                            <b style="font-size: 1.1rem; color: {pnl_color};">{ltp:.4f}</b>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 4px;">
                            <span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">Lot / Qty</span><br>
                            <b style="font-size: 1.1rem; color: #0f111a;">{t['qty']}</b> <span style="font-size: 0.8rem; color: #64748b;">({exec_type})</span>
                        </div>
                        <div style="background: #fef2f2; padding: 10px; border-radius: 4px; border: 1px solid #fecaca;">
                            <span style="color: #ef4444; font-size: 0.75rem; text-transform: uppercase; font-weight: 800;">Risk Stop</span><br>
                            <b style="font-size: 1.1rem; color: #ef4444;">{t['sl']:.4f}</b>
                        </div>
                    </div>
                    <div style="background: #0f111a; padding: 10px; border-radius: 4px; font-size: 0.9rem; text-align: center; color: #38bdf8; font-weight: 700; letter-spacing: 0.5px;">
                        🎯 TP1: {t.get('tp1', 0):.2f} &nbsp;|&nbsp; TP2: {t.get('tp2', 0):.2f} &nbsp;|&nbsp; TP3: {t.get('tp3', 0):.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🛑 KILL TRADE", type="primary", use_container_width=True):
                bot.state["manual_exit"] = True
                st.toast("Forcing trade closure...", icon="🛑")
        else:
            st.info("⏳ Radar Active: Waiting for High-Probability Setup...")

        st.markdown("<br>### 📈 Technical Engine", unsafe_allow_html=True)
        c_h1, c_h2 = st.columns(2)
        with c_h1: SHOW_CHART = st.toggle("📊 Render Chart", True)
        with c_h2: FULL_CHART = st.toggle("⛶ Cinema Mode", False)
        
        if SHOW_CHART and bot.state["latest_data"] is not None:
            chart_df = bot.state["latest_data"].copy()
            chart_df['time'] = (pd.to_datetime(chart_df.index).astype('int64') // 10**9) - 19800
            candles = chart_df[['time', 'open', 'high', 'low', 'close']].to_dict('records')
            
            fib_lines = []
            if not bot.state["active_trade"] and bot.state.get('fib_data'):
                fib = bot.state['fib_data']
                fib_lines = [
                    {"price": fib.get('major_high', 0), "color": '#ef4444', "lineWidth": 1, "lineStyle": 0, "title": 'Major Res'},
                    {"price": fib.get('fib_high', 0), "color": '#fbbf24', "lineWidth": 2, "lineStyle": 2, "title": 'Golden 0.618'},
                    {"price": fib.get('fib_low', 0), "color": '#fbbf24', "lineWidth": 2, "lineStyle": 2, "title": 'Golden 0.65'},
                    {"price": fib.get('major_low', 0), "color": '#22c55e', "lineWidth": 1, "lineStyle": 0, "title": 'Major Sup'}
                ]
            
            chartOptions = {
                "height": 700 if FULL_CHART else 400, 
                "layout": { "textColor": '#1e293b', "background": { "type": 'solid', "color": '#ffffff' } }, 
                "grid": { "vertLines": { "color": 'rgba(226, 232, 240, 0.8)' }, "horzLines": { "color": 'rgba(226, 232, 240, 0.8)' } }, 
                "crosshair": { "mode": 0 }, 
                "timeScale": { "timeVisible": True, "secondsVisible": False }
            }
            chart_series = [{"type": 'Candlestick', "data": candles, "options": {"upColor": '#26a69a', "downColor": '#ef5350'}, "priceLines": fib_lines}]

            if 'avwap' in chart_df.columns:
                avwap_data = chart_df[['time', 'avwap']].dropna().rename(columns={'avwap': 'value'}).to_dict('records')
                if avwap_data: chart_series.append({"type": 'Line', "data": avwap_data, "options": { "color": '#9c27b0', "lineWidth": 2, "title": 'ICT AVWAP' }})
            if 'vwap' in chart_df.columns:
                vwap_data = chart_df[['time', 'vwap']].dropna().rename(columns={'vwap': 'value'}).to_dict('records')
                if vwap_data: chart_series.append({"type": 'Line', "data": vwap_data, "options": { "color": '#ff9800', "lineWidth": 2, "title": 'VWAP' }})
            ema_col = 'ema_fast' if 'ema_fast' in chart_df.columns else 'ema9'
            if ema_col in chart_df.columns:
                ema_data = chart_df[['time', ema_col]].dropna().rename(columns={ema_col: 'value'}).to_dict('records')
                if ema_data: chart_series.append({"type": 'Line', "data": ema_data, "options": { "color": '#0ea5e9', "lineWidth": 2, "title": 'EMA' }})

            renderLightweightCharts([{"chart": chartOptions, "series": chart_series}], key="static_tv_chart")

    with tab2:
        colA, colB, colC = st.columns(3)
        with colA:
            st.subheader("📊 52W High/Low")
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
                    for sym in ["NIFTY", "BANKNIFTY", "SENSEX", "RELIANCE"]:
                        df = bot.get_historical_data("NSE", "99926000", symbol=sym, interval="1d" if not is_mkt_open else "5m")
                        if df is not None and len(df) > 2:
                            inside_bar = "Yes 🟡" if (df['high'].iloc[-1] <= df['high'].iloc[-2] and df['low'].iloc[-1] >= df['low'].iloc[-2]) else "No"
                            scan_results.append({"Symbol": sym, "LTP": round(df['close'].iloc[-1], 2), "Inside Bar": inside_bar, "BTST/STBT": check_btst_stbt(df) if not is_mkt_open else "N/A"})
                    st.dataframe(pd.DataFrame(scan_results), use_container_width=True, hide_index=True)

        with colC:
            st.subheader("🪙 Penny Stocks")
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
            col_title, col_btn = st.columns([2, 1])
            with col_title:
                st.subheader("System Console")
            with col_btn:
                if st.button("🗑️ Clear", use_container_width=True):
                    bot.state["logs"].clear()
                    if "paper_history" in bot.state:
                        bot.state["paper_history"] = []
                    bot.state["daily_pnl"] = 0.0
                    bot.state["trades_today"] = 0
                    if not bot.is_mock and HAS_DB:
                        try:
                            user_id = getattr(bot, "system_user_id", bot.api_key)
                            supabase.table("trade_logs").delete().eq("user_id", user_id).execute()
                        except Exception as e:
                            pass
                    st.rerun()
                    
            for l in bot.state["logs"]: st.markdown(f"`{l}`")
            
        with pnl_col:
            if bot.is_mock:
                st.subheader(f"📊 Mock Ledger (Session: {bot.client_name})")
                if bot.state.get("paper_history"):
                    df_paper = pd.DataFrame(bot.state["paper_history"])
                    st.dataframe(df_paper.iloc[::-1], use_container_width=True)
                    
                    output_mock = io.BytesIO()
                    with pd.ExcelWriter(output_mock, engine='xlsxwriter') as writer: df_paper.to_excel(writer, index=False)
                    st.download_button("📥 Export Mock Ledger (.xlsx)", data=output_mock.getvalue(), file_name=f"Mock_Trade_Log_{bot.client_name}.xlsx")
                    
                    with st.expander("📈 View Daily & Weekly Reports"):
                        df_paper['temp_date'] = pd.to_datetime(df_paper['Date'], errors='coerce')
                        df_paper['temp_pnl'] = pd.to_numeric(df_paper['PnL'], errors='coerce').fillna(0)
                        rep_mode = st.radio("Select Report Interval", ["Daily", "Weekly"], horizontal=True, label_visibility="collapsed", key="mock_rep")
                        if rep_mode == "Daily":
                            report = df_paper.groupby(df_paper['temp_date'].dt.date)['temp_pnl'].sum().reset_index()
                            report.columns = ["Day", "Total PnL"]
                        else:
                            report = df_paper.groupby(df_paper['temp_date'].dt.strftime('%Y-W%V'))['temp_pnl'].sum().reset_index()
                            report.columns = ["Week", "Total PnL"]
                        st.dataframe(report.style.map(lambda x: 'color: #22c55e; font-weight: bold;' if x > 0 else ('color: #ef4444; font-weight: bold;' if x < 0 else ''), subset=['Total PnL']), use_container_width=True, hide_index=True)
                else: st.info("No paper trades recorded yet in this session.")
            else:
                user_id = getattr(bot, "system_user_id", bot.api_key)
                st.subheader(f"📊 Live Trade Matrix (Cloud DB - ID: {user_id})")
                if HAS_DB:
                    try:
                        res = supabase.table("trade_logs").select("*").eq("user_id", user_id).execute()
                        if res.data:
                            df_db = pd.DataFrame(res.data)
                            df_db = df_db.drop(columns=["id", "user_id"], errors="ignore")
                            st.dataframe(df_db.iloc[::-1], use_container_width=True)
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df_db.to_excel(writer, index=False)
                            st.download_button("📥 Export Live Log (.xlsx)", data=output.getvalue(), file_name=f"Live_Trade_Log_{user_id}.xlsx")
                            
                            with st.expander("📈 View Daily & Weekly Reports"):
                                df_db['temp_date'] = pd.to_datetime(df_db['trade_date'], errors='coerce')
                                df_db['temp_pnl'] = pd.to_numeric(df_db['pnl'], errors='coerce').fillna(0)
                                rep_mode = st.radio("Select Report Interval", ["Daily", "Weekly"], horizontal=True, label_visibility="collapsed", key="live_rep")
                                if rep_mode == "Daily":
                                    report = df_db.groupby(df_db['temp_date'].dt.date)['temp_pnl'].sum().reset_index()
                                    report.columns = ["Day", "Total PnL"]
                                else:
                                    report = df_db.groupby(df_db['temp_date'].dt.strftime('%Y-W%V'))['temp_pnl'].sum().reset_index()
                                    report.columns = ["Week", "Total PnL"]
                                st.dataframe(report.style.map(lambda x: 'color: #22c55e; font-weight: bold;' if x > 0 else ('color: #ef4444; font-weight: bold;' if x < 0 else ''), subset=['Total PnL']), use_container_width=True, hide_index=True)
                                
                        else: st.info("No real trades recorded yet.")
                    except Exception as e: st.error(f"Could not load trades: {e}")
                else: st.error("Cloud DB disconnected.")
                    
    with tab4:
        c_dx, c_bias = st.columns(2)
        with c_dx:
            st.subheader("🕉️ CoinDCX Intraday Momentum")
            if st.button("Scan CoinDCX 🔥", use_container_width=True):
                with st.spinner("Fetching live market data..."):
                    try:
                        response = requests.get("https://api.coindcx.com/exchange/ticker", timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            valid_coins = []
                            for coin in data:
                                market = coin.get('market', '')
                                if market.endswith('USDT') or market.endswith('INR'):
                                    try:
                                        change = float(coin.get('change_24_hour', 0))
                                        ltp = float(coin.get('last_price', 0))
                                        if ltp > 0: valid_coins.append({"Pair": market, "LTP": ltp, "24h Change (%)": change})
                                    except: pass
                            if valid_coins:
                                df_dcx = pd.DataFrame(valid_coins).sort_values(by="24h Change (%)", ascending=False).head(15)
                                df_dcx['24h Change (%)'] = df_dcx['24h Change (%)'].apply(lambda x: f"+{x}%" if x > 0 else f"{x}%")
                                st.dataframe(df_dcx, use_container_width=True, hide_index=True)
                        else: st.error(f"CoinDCX API Error: {response.status_code}")
                    except Exception as e: st.error(f"Failed to fetch CoinDCX data: {e}")

        with c_bias:
            st.subheader("📈 Directional Bias (BTC & Gold)")
            if st.button("Analyze BTC & XAUUSD 🔍", use_container_width=True):
                with st.spinner("Analyzing Market Structure & EMAs..."):
                    assets_to_check = {"Bitcoin (BTC-USD)": "BTC-USD", "Gold (XAUUSD)": "GC=F", "Silver (SI=F)": "SI=F"}
                    bias_results = []
                    for name, ticker in assets_to_check.items():
                        try:
                            tk = yf.Ticker(ticker)
                            df_bias = tk.history(period="5d", interval="15m")
                            if not df_bias.empty:
                                df_bias['ema9'] = df_bias['Close'].ewm(span=9).mean()
                                df_bias['ema21'] = df_bias['Close'].ewm(span=21).mean()
                                c_price = df_bias['Close'].iloc[-1]
                                e9 = df_bias['ema9'].iloc[-1]
                                e21 = df_bias['ema21'].iloc[-1]
                                if c_price > e9 and e9 > e21: bias = "UP 🟢 (Bullish Momentum)"
                                elif c_price < e9 and e9 < e21: bias = "DOWN 🔴 (Bearish Momentum)"
                                else: bias = "RANGING 🟡 (Wait for Breakout)"
                                bias_results.append({"Asset": name, "Current Price": round(c_price, 2), "Next Expected Move": bias})
                        except: pass
                    if bias_results: st.dataframe(pd.DataFrame(bias_results), use_container_width=True, hide_index=True)

    def cycle_asset():
        assets = st.session_state.get('asset_options', list(DEFAULT_LOTS.keys()))
        if st.session_state.sb_index_input in assets: st.session_state.sb_index_input = assets[(assets.index(st.session_state.sb_index_input) + 1) % len(assets)]
        else: st.session_state.sb_index_input = assets[0]

    def cycle_strat():
        st.session_state.sb_strat_input = STRAT_LIST[(STRAT_LIST.index(st.session_state.sb_strat_input) + 1) % len(STRAT_LIST)]

    dock_container = st.container()
    with dock_container:
        st.markdown('<div id="bottom-dock-anchor" class="bottom-dock-container">', unsafe_allow_html=True)
        dock_c1, dock_c2, dock_c3 = st.columns(3)
        with dock_c1: st.button("◀️", key="btn_back", on_click=cycle_asset, use_container_width=True)
        with dock_c2: 
            if st.button("🏠", key="btn_home", use_container_width=True): st.rerun()
        with dock_c3: st.button("🔲", key="btn_recent", on_click=cycle_strat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if bot.state.get("is_running"):
        time.sleep(2)
        st.rerun()

