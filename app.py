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
import socket
import os
import signal
import sys
import subprocess
from threading import Event

# ---------- ML / News / Extra Imports ----------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
    import mt5linux
    HAS_MT5_LINUX = True
except ImportError:
    HAS_MT5_LINUX = False

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

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

# ==========================================
# Interactive Brokers (requires ib_insync)
# ==========================================
try:
    from ib_insync import *
    HAS_IB = True
except ImportError:
    HAS_IB = False

# ==========================================
# SOUND FUNCTIONS (fixed – reliable beep)
# ==========================================
def play_sound_ui(sound_type="entry"):
    """Play a sound via HTML5 audio."""
    if not st.session_state.get("audio_enabled", False):
        return
    beep_url = "https://media.geeksforgeeks.org/wp-content/uploads/20190531135120/beep.mp3"
    components.html(f"""
        <audio autoplay>
            <source src="{beep_url}" type="audio/mpeg">
        </audio>
    """, height=0)

def unlock_audio():
    """Silent sound to unlock audio in browser."""
    components.html("""
        <audio autoplay>
            <source src="data:audio/mp3;base64,SUQzBAAAAAABEVRYWFgAAAAtAAADY29tbWVudABCaWdTb3VuZEJhbmsuY29tIC8gTG9uZWx5TWVkaWEuY29tIFNvdW5kIEVmZmVjdHMAAAAAAFRBTkNFAAAAQ29udGVudCBUaGF0IFdvcmtzLCBJbmMuIAAAADM2MDAwMDAwTk9STUFMAAAAJVRoYXRzIGEgdGVzdCBzb3VuZC4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==" type="audio/mp3">
        </audio>
    """, height=0)

# ==========================================
# LIVE NEWS FUNCTIONS (Kannada + English)
# ==========================================
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

def generate_market_prediction(asset="NIFTY"):
    if st.session_state.bot is None:
        return "⚖️ Market prediction unavailable (bot not started)"
    bot = st.session_state.bot
    signal = bot.state.get("current_signal", "WAIT")
    strength = bot.state.get("signal_strength", 0)
    trend = bot.state.get("current_trend", "Neutral")
    ml_up = bot.state.get("ml_prob_up", 50)
    ml_down = bot.state.get("ml_prob_down", 50)
    if signal == "BUY_CE":
        direction = "🟢 UP"
        confidence = strength
    elif signal == "BUY_PE":
        direction = "🔴 DOWN"
        confidence = strength
    else:
        if ml_up > ml_down + 10:
            direction = "🟢 UP (ML)"
            confidence = ml_up
        elif ml_down > ml_up + 10:
            direction = "🔴 DOWN (ML)"
            confidence = ml_down
        else:
            direction = "⚪ SIDEWAYS"
            confidence = 50
    return f"📈 Prediction: Market likely to move {direction} in next 30 mins | Confidence: {confidence:.0f}% | {trend}"

import xml.etree.ElementTree as ET

def fetch_feed(url):
    """Fetch RSS feed and return list of titles."""
    titles = []
    try:
        if HAS_FEEDPARSER:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                if hasattr(entry, 'title'):
                    titles.append(entry.title)
        else:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for item in root.findall('.//item'):
                title = item.find('title').text
                if title:
                    titles.append(title)
    except Exception:
        return None
    return titles if titles else None

def fetch_english_news(asset="NIFTY"):
    asset_terms = {
        "NIFTY": "Nifty 50",
        "BANKNIFTY": "Bank Nifty",
        "SENSEX": "Sensex",
        "GOLD": "Gold",
        "SILVER": "Silver",
        "CRUDEOIL": "Crude Oil",
        "NATURALGAS": "Natural Gas",
        "XAUUSD": "Gold",
        "BTCUSD": "Bitcoin",
        "ETHUSD": "Ethereum",
        "SOLUSD": "Solana"
    }
    term = asset_terms.get(asset, asset)
    query = f"{term} stock market"
    google_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    titles = fetch_feed(google_url)
    if titles and len(titles) >= 3:
        return [f"📰 {t}" for t in titles[:8]]
    reuters_url = "https://feeds.reuters.com/news/wealth"
    titles = fetch_feed(reuters_url)
    if titles and len(titles) >= 3:
        return [f"📰 {t}" for t in titles[:8]]
    yahoo_url = f"https://finance.yahoo.com/news/rssindex"
    titles = fetch_feed(yahoo_url)
    if titles and len(titles) >= 3:
        return [f"📰 {t}" for t in titles[:8]]
    return [generate_market_prediction(asset)]

def fetch_kannada_news(asset="NIFTY"):
    rss_feeds = [
        "https://kannada.oneindia.com/rss/news-business-feed.xml",
        "https://www.prajavani.net/rss/ವಾಣಿಜ్య",
        "https://vijaykarnataka.com/rss/business",
        "https://kannada.asianetnews.com/rss/business",
        "https://kannada.news18.com/rss/business/",
        "https://www.tv9kannada.com/rss/business",
        "https://kannada.hindustantimes.com/rss/business",
        "https://kannada.goodreturns.in/rss/business/"
    ]
    for feed_url in rss_feeds:
        titles = fetch_feed(feed_url)
        if titles and len(titles) >= 3:
            return [f"📰 {t}" for t in titles[:5]]
    return [generate_market_prediction(asset)]

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
import threading
import time

def news_updater():
    while True:
        asset = st.session_state.get('sb_index_input', 'NIFTY')
        st.session_state.kannada_news = fetch_kannada_news(asset)
        st.session_state.english_news = fetch_english_news(asset)
        time.sleep(300)  # update every 5 minutes

if 'kannada_news' not in st.session_state:
    st.session_state.kannada_news = []
    st.session_state.english_news = []
    if 'news_thread_started' not in st.session_state:
        thread = threading.Thread(target=news_updater, daemon=True)
        add_script_run_ctx(thread)
        thread.start()
        st.session_state.news_thread_started = True

if 'sb_index_input' not in st.session_state:
    st.session_state.sb_index_input = "NIFTY"
if 'sb_strat_input' not in st.session_state:
    st.session_state.sb_strat_input = "Momentum Breakout + S&R"
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'prev_index' not in st.session_state:
    st.session_state.prev_index = "NIFTY"
if 'custom_stock' not in st.session_state:
    st.session_state.custom_stock = ""
if 'custom_code_input' not in st.session_state:
    st.session_state.custom_code_input = "EMA Crossover (9 & 21)"
if 'user_lots' not in st.session_state:
    st.session_state.user_lots = {}
if 'asset_options' not in st.session_state:
    st.session_state.asset_options = ["NIFTY", "BANKNIFTY", "SENSEX", "CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "XAUUSD", "BTCUSD", "ETHUSD", "SOLUSD"]
if 'use_quantity_mode' not in st.session_state:
    st.session_state.use_quantity_mode = False
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = False

# ==========================================
# DATABASE & GLOBAL HELPERS
# ==========================================
def safe_secrets_get(key, default=None):
    try:
        return st.secrets.get(key, default)
    except:
        return default

@st.cache_resource
def init_supabase() -> Client:
    url = safe_secrets_get("SUPABASE_URL")
    key = safe_secrets_get("SUPABASE_KEY")
    if url and key:
        try:
            return create_client(url, key)
        except Exception as e:
            return None
    return None

supabase = init_supabase()
HAS_DB = supabase is not None

BACKGROUND_RUNNING = False
BACKGROUND_THREAD = None
STOP_EVENT = Event()

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
# DATABASE FUNCTIONS 
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
        "mt5_acc": "", "mt5_pass": "", "mt5_server": "", "mt5_api_url": "",
        "zerodha_api": "", "zerodha_secret": "", "coindcx_api": "", "coindcx_secret": "",
        "delta_api": "", "delta_secret": "", "ib_host": "", "ib_port": 7497, "ib_client_id": 1
    }

def save_creds(user_id, angel_api, client_id, pwd, totp_secret, tg_token, tg_chat, wa_phone, wa_api, mt5_acc, mt5_pass, mt5_server, mt5_api_url, zerodha_api, zerodha_secret, coindcx_api, coindcx_secret, delta_api, delta_secret, ib_host, ib_port, ib_client_id):
    if HAS_DB:
        data = {
            "user_id": user_id, "angel_api": angel_api, "client_id": client_id, "pwd": pwd, 
            "totp_secret": totp_secret, "tg_token": tg_token, "tg_chat": tg_chat, 
            "wa_phone": wa_phone, "wa_api": wa_api,
            "mt5_acc": mt5_acc, "mt5_pass": mt5_pass, "mt5_server": mt5_server, "mt5_api_url": mt5_api_url,
            "zerodha_api": zerodha_api, "zerodha_secret": zerodha_secret,
            "coindcx_api": coindcx_api, "coindcx_secret": coindcx_secret,
            "delta_api": delta_api, "delta_secret": delta_secret,
            "ib_host": ib_host, "ib_port": ib_port, "ib_client_id": ib_client_id
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
# UI & CUSTOM CSS
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

    div[data-testid="stTabs"] { background: transparent !important; }
    
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
    
    .signal-strength-tiny {
        font-size: 0.7rem;
        color: #64748b;
        display: inline-block;
        padding: 0 5px;
    }
    
    .risk-low { color: #22c55e; font-weight: bold; }
    .risk-medium { color: #fbbf24; font-weight: bold; }
    .risk-high { color: #ef4444; font-weight: bold; }
    
    .hero-badge { background: #22c55e; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .zero-badge { background: #ef4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    
    .modern-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.04);
        border: 1px solid #edf2f7;
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 20px;
        height: 100%;
    }
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    }
    .modern-card h3 {
        margin-top: 0;
        color: #2563eb;
        font-weight: 700;
        font-size: 1.2rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    
    .broker-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
        background: #2563eb;
        color: white;
    }
    
    .news-ticker {
        background: #1e293b;
        color: white;
        padding: 8px 0;
        overflow: hidden;
        white-space: nowrap;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 1rem;
    }
    .news-ticker span {
        display: inline-block;
        padding-left: 100%;
        animation: ticker 60s linear infinite;
    }
    @keyframes ticker {
        0% { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }

    .bottom-dock {
        position: fixed;
        bottom: 10px;
        left: 0;
        right: 0;
        display: flex;
        justify-content: center;
        gap: 1rem;
        z-index: 999;
    }
    .bottom-dock button {
        background: #0284c7;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 8px 20px;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS AND DEFAULTS
# ==========================================
LOT_SIZES = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "SENSEX": 20,
    "CRUDEOIL": 100,
    "NATURALGAS": 1250,
    "GOLD": 100,
    "SILVER": 30,
    "XAUUSD": 1,
    "EURUSD": 1,
    "BTCUSD": 1,
    "ETHUSD": 1,
    "SOLUSD": 1,
    "XRPUSD": 1,
    "ADAUSD": 1,
    "DOGEUSD": 1,
    "BNBUSD": 1
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

STRAT_LIST = [
    "Momentum Breakout + S&R",
    "Machine Learning",
    "VIJAY & RFF All-In-One", 
    "Institutional FVG + SMC",
    "Lux Algo Institutional ICT",
    "Keyword Rule Builder", 
    "TradingView Webhook"
]

if not st.session_state.user_lots:
    st.session_state.user_lots = LOT_SIZES.copy()

is_mkt_open, mkt_status_msg = get_market_status(st.session_state.sb_index_input)

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
# BROKER BRIDGES (MT5, IB)
# ==========================================
class MT5WebBridge:
    def __init__(self, account=None, password=None, server=None, api_url=None):
        self.account = account
        self.password = password
        self.server = server
        self.api_url = api_url or "https://mt5-web-api.mtapi.io/v1"
        self.session = requests.Session()
        self.connected = False
        self.token = None
        self.use_direct_mt5 = HAS_MT5 and account and server
        
    def connect(self):
        if self.use_direct_mt5:
            try:
                if mt5.initialize():
                    if mt5.login(int(self.account), password=self.password, server=self.server):
                        self.connected = True
                        return True, "Connected via direct MT5"
            except:
                pass
        
        if HAS_MT5_LINUX and not self.connected:
            try:
                client = mt5linux.MT5Linux("localhost")
                if client.initialize():
                    self.client = client
                    self.connected = True
                    self.use_direct_mt5 = False
                    return True, "Connected via MT5 Linux bridge"
            except:
                pass
        
        if not self.connected and self.api_url:
            try:
                response = self.session.post(
                    f"{self.api_url}/auth",
                    json={
                        "login": int(self.account) if self.account else 0,
                        "password": self.password,
                        "server": self.server
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    self.token = data.get('token')
                    self.connected = True
                    self.use_direct_mt5 = False
                    return True, "Connected via MT5 Web API"
            except:
                pass
                
        return False, "Could not connect to MT5"
    
    def get_live_price(self, symbol):
        if self.use_direct_mt5 and mt5.initialize():
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return (tick.bid + tick.ask) / 2.0
        
        if self.token:
            try:
                response = self.session.get(
                    f"{self.api_url}/quote/{symbol}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    return (data.get('bid', 0) + data.get('ask', 0)) / 2
            except:
                pass
        return None
    
    def place_order(self, symbol, volume, order_type, sl=None, tp=None):
        self.log(f"MT5 place_order: symbol={symbol}, volume={volume}, type={order_type}")
        if self.use_direct_mt5 and mt5.initialize():
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": 0.0,
                "sl": sl if sl else 0,
                "tp": tp if tp else 0,
                "deviation": 20,
                "magic": 234000,
                "comment": "SHREE Algo",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result.order, f"Order placed: {result.order}"
            return None, f"Failed: {result.comment}"
        
        if self.token:
            try:
                response = self.session.post(
                    f"{self.api_url}/order",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json={
                        "symbol": symbol,
                        "volume": float(volume),
                        "type": order_type.upper(),
                        "sl": sl if sl else 0,
                        "tp": tp if tp else 0
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get('order_id'), "Order placed via web API"
                else:
                    return None, f"HTTP {response.status_code}: {response.text}"
            except Exception as e:
                return None, f"Exception: {e}"
        
        return None, "All MT5 connection methods failed"
    
    def get_historical_data(self, symbol, timeframe="M5", count=500):
        timeframe_map = {
            "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
            "1h": "H1", "4h": "H4", "1d": "D1"
        }
        mt5_tf = timeframe_map.get(timeframe, "M5")
        
        if self.use_direct_mt5 and mt5.initialize():
            mt5_tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf_map.get(mt5_tf, mt5.TIMEFRAME_M5), 0, count)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.rename(columns={'time': 'timestamp', 'real_volume': 'volume'}, inplace=True)
                df.index = df['timestamp']
                return df
        
        if self.token:
            try:
                response = self.session.get(
                    f"{self.api_url}/history/{symbol}",
                    params={"timeframe": mt5_tf, "count": count},
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data.get('rates', []))
                    if not df.empty:
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                        df.index = df['timestamp']
                        return df
            except:
                pass
        
        return None
    
    def get_account_info(self):
        if self.use_direct_mt5 and mt5.initialize():
            acc = mt5.account_info()
            if acc:
                return {
                    'balance': acc.balance,
                    'equity': acc.equity,
                    'margin': acc.margin,
                    'profit': acc.profit
                }
        
        if self.token:
            try:
                response = self.session.get(
                    f"{self.api_url}/account",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=5
                )
                if response.status_code == 200:
                    return response.json()
            except:
                pass
        return None
    
    def disconnect(self):
        if self.use_direct_mt5:
            mt5.shutdown()
        self.connected = False
        self.token = None

class IBBridge:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False

    def connect(self):
        if not HAS_IB:
            return False, "ib_insync not installed"
        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            return True, "Connected to Interactive Brokers"
        except Exception as e:
            return False, f"IB connection failed: {e}"

    def get_live_price(self, symbol, exchange='SMART', currency='USD'):
        if not self.connected:
            return None
        contract = Stock(symbol, exchange, currency)
        self.ib.reqMktData(contract, '', False, False)
        time.sleep(1)
        ticker = self.ib.ticker(contract)
        return (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else ticker.last

    def place_order(self, symbol, qty, order_type, exchange='SMART', currency='USD'):
        if not self.connected:
            return None, "Not connected"
        contract = Stock(symbol, exchange, currency)
        action = 'BUY' if order_type.upper() == 'BUY' else 'SELL'
        order = MarketOrder(action, qty)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        if trade.orderStatus.status == 'Filled':
            return trade.order.orderId, f"Order filled at {trade.orderStatus.avgFillPrice}"
        return trade.order.orderId, f"Order submitted: {trade.orderStatus.status}"

    def disconnect(self):
        if self.ib:
            self.ib.disconnect()
        self.connected = False

# ==========================================
# FOMO SCANNER (includes indices & commodities)
# ==========================================
class FOMOScanner:
    def __init__(self):
        self.watchlist = {
            "INDIAN_STOCKS": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "WIPRO.NS"],
            "US_STOCKS": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "DIS", "JPM"],
            "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
            "INDICES": ["^NSEI", "^BSESN", "^NSEBANK"],
            "COMMODITIES": ["GC=F", "SI=F", "CL=F", "NG=F"]
        }
        self.signals = deque(maxlen=50)
        self.scan_interval = 60
        self.last_scan = 0
        
    def scan(self):
        signals = []
        for category, tickers in self.watchlist.items():
            for ticker in tickers:
                try:
                    df = yf.Ticker(ticker).history(period="1d", interval="5m")
                    if len(df) < 20:
                        continue
                    df['volume_ma'] = df['Volume'].rolling(20).mean()
                    df['volume_ratio'] = df['Volume'] / df['volume_ma']
                    df['high_20'] = df['High'].rolling(20).max()
                    df['low_20'] = df['Low'].rolling(20).min()
                    last = df.iloc[-1]
                    if last['volume_ratio'] > 1.5 and last['Close'] > last['high_20'] * 0.99:
                        signals.append({
                            "time": get_ist().strftime("%H:%M"),
                            "symbol": ticker,
                            "category": category,
                            "price": round(last['Close'], 2),
                            "volume_spike": f"{last['volume_ratio']:.1f}x",
                            "signal": "BUY 🚀"
                        })
                    elif last['volume_ratio'] > 1.5 and last['Close'] < last['low_20'] * 1.01:
                        signals.append({
                            "time": get_ist().strftime("%H:%M"),
                            "symbol": ticker,
                            "category": category,
                            "price": round(last['Close'], 2),
                            "volume_spike": f"{last['volume_ratio']:.1f}x",
                            "signal": "SELL 🔻"
                        })
                except:
                    continue
        self.signals.extend(signals)
        return signals

fomo_scanner = FOMOScanner()

# ==========================================
# Scalping Module (always active)
# ==========================================
class ScalpingModule:
    def __init__(self, bot):
        self.bot = bot
        self.scalp_trades = deque(maxlen=50)
        self.scalp_pnl = 0
        
    def scalp_signal(self, df):
        if df is None or len(df) < 20:
            return "WAIT"
            
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['tick_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        if last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21'] and last['tick_ratio'] > 1.2:
            return "SCALP_BUY"
        if last['ema9'] < last['ema21'] and prev['ema9'] >= prev['ema21'] and last['tick_ratio'] > 1.2:
            return "SCALP_SELL"
        
        if (last['lower_wick'] > last['body'] * 1.5 and last['tick_ratio'] > 1.2 and last['close'] > last['open']):
            return "SCALP_BUY"
        if (last['upper_wick'] > last['body'] * 1.5 and last['tick_ratio'] > 1.2 and last['close'] < last['open']):
            return "SCALP_SELL"
            
        return "WAIT"
    
    def execute_scalp(self, signal, price, qty):
        if signal == "SCALP_BUY":
            entry = price
            sl = price * 0.998
            tp = price * 1.002
            return {
                "type": "BUY",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "rr": 1
            }
        elif signal == "SCALP_SELL":
            entry = price
            sl = price * 1.002
            tp = price * 0.998
            return {
                "type": "SELL",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "rr": 1
            }
        return None

# ==========================================
# TECHNICAL ANALYZER (full definition)
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
        try:
            df = df.copy()
            df['vol_sma'] = df['volume'].rolling(20).mean()
            if is_index: 
                df['vol_spike'] = True
            else: 
                df['vol_spike'] = df['volume'] >= (df['vol_sma'] * 0.8)
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain / loss)))
            
            df['tr0'] = abs(df['high'] - df['low'])
            df['tr1'] = abs(df['high'] - df['close'].shift())
            df['tr2'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['is_sideways'] = (df['rsi'].between(45, 55)) & (abs(df['close'] - df['ema21']) < (df['atr'] * 0.4))
            df['inside_bar'] = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))
            
            try:
                df['date'] = df.index.date
                if 'volume' in df.columns and df['volume'].sum() > 0:
                    df['vol_price'] = df['close'] * df['volume']
                    df['avwap'] = df.groupby('date')['vol_price'].cumsum() / df.groupby('date')['volume'].cumsum()
                else: 
                    df['avwap'] = df.groupby('date')['close'].transform(lambda x: x.expanding().mean())
            except: 
                df['avwap'] = df['close']
            
            if 'ema9' not in df.columns:
                df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
            if 'ema21' not in df.columns:
                df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            df['returns_1'] = df['close'].pct_change(1)
            df['returns_3'] = df['close'].pct_change(3)
            df['returns_5'] = df['close'].pct_change(5)
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            df['volatility_10'] = df['returns_1'].rolling(10).std()
            
            return df
        except Exception as e:
            print(f"❌ Error in calculate_indicators: {e}")
            import traceback
            traceback.print_exc()
            return df

    def calculate_fib_zones(self, df, lookback=100):
        if df is None or len(df) < 10:
            return 0, 0, 0, 0
        actual_lookback = min(lookback, len(df))
        major_high = df['high'].rolling(actual_lookback).max().iloc[-1]
        major_low = df['low'].rolling(actual_lookback).min().iloc[-1]
        diff = major_high - major_low
        
        fib_618 = major_high - (diff * 0.618)
        fib_650 = major_high - (diff * 0.650)
        
        return major_high, major_low, min(fib_618, fib_650), max(fib_618, fib_650)

    def detect_order_blocks(self, df):
        df = df.copy()
        df['body'] = abs(df['close'] - df['open'])
        avg_body = df['body'].rolling(10).mean()
        strong_up = (df['close'] > df['open']) & (df['body'] > avg_body * 1.2)
        strong_down = (df['close'] < df['open']) & (df['body'] > avg_body * 1.2)
        bob_h, bob_l, beob_h, beob_l = 0.0, 0.0, 0.0, 0.0
        
        for i in range(len(df)-1, max(0, len(df)-50), -1):
            if strong_up.iloc[i] and bob_h == 0.0:
                for j in range(i-1, max(0, i-10), -1):
                    if df['close'].iloc[j] < df['open'].iloc[j]: 
                        bob_h, bob_l = df['high'].iloc[j], df['low'].iloc[j]; break
            if strong_down.iloc[i] and beob_h == 0.0:
                for j in range(i-1, max(0, i-10), -1):
                    if df['close'].iloc[j] > df['open'].iloc[j]: 
                        beob_h, beob_l = df['high'].iloc[j], df['low'].iloc[j]; break
            if bob_h != 0.0 and beob_h != 0.0: break
        return {"bob_high": bob_h, "bob_low": bob_l, "beob_high": beob_h, "beob_low": beob_l}

    def detect_pin_bar(self, df):
        if df is None or len(df) < 2:
            return "None"
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        upper_wick = last['high'] - max(last['close'], last['open'])
        lower_wick = min(last['close'], last['open']) - last['low']
        if lower_wick > body * 1.5 and last['close'] > last['open']:
            return "Bullish Pin 📌"
        elif upper_wick > body * 1.5 and last['close'] < last['open']:
            return "Bearish Pin 📌"
        return "None"

    def apply_institutional_fvg_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 20: return "WAIT", "WAIT", 0, 0, df, 0, {}, 0
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']

        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

        df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
        df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])

        last = df.iloc[-1]
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        smc_blocks = self.detect_order_blocks(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high, **smc_blocks}

        trend, signal = "AWAITING FVG REVERSAL 🟡", "WAIT"
        signal_strength = 50

        latest_bull_fvg = df[df['fvg_bull']].iloc[-1] if any(df['fvg_bull']) else None
        latest_bear_fvg = df[df['fvg_bear']].iloc[-1] if any(df['fvg_bear']) else None

        if latest_bull_fvg is not None:
            mitigated_bull = (last['low'] <= latest_bull_fvg['high'].shift(1)) and (last['low'] >= latest_bull_fvg['low'].shift(1))
            if mitigated_bull and last['close'] > last['open']:
                signal = "BUY_CE"
                trend = "BULL FVG REVERSAL CONFIRMED 🟢"
                signal_strength = 75

        if latest_bear_fvg is not None:
            mitigated_bear = (last['high'] >= latest_bear_fvg['low'].shift(1)) and (last['high'] <= latest_bear_fvg['high'].shift(1))
            if mitigated_bear and last['close'] < last['open']:
                signal = "BUY_PE"
                trend = "BEAR FVG REVERSAL CONFIRMED 🔴"
                signal_strength = 75

        return trend, signal, last['vwap'], last['ema9'], df, atr, fib_data, signal_strength

    def apply_vijay_rff_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 30: return "WAIT", "WAIT", 0, 0, df, 0, {}, 0
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]
        
        df = self.calculate_indicators(df, is_index)
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        smc_blocks = self.detect_order_blocks(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high, **smc_blocks}
        
        if not HAS_PTA:
            return "WAIT (pandas_ta required)", "WAIT", df['close'].iloc[-1], df['close'].iloc[-1], df, atr, fib_data, 0

        df_ta = df.copy()
        df_ta.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        df_ta['EMA_5'] = ta.ema(df_ta['Close'], length=5)
        df_ta['EMA_13'] = ta.ema(df_ta['Close'], length=13)
        df_ta['EMA_21'] = ta.ema(df_ta['Close'], length=21)
        
        df_ta['RSI_14'] = ta.rsi(df_ta['Close'], length=14)
        if df_ta['RSI_14'] is None: 
            df_ta['RSI_14'] = df_ta['Close'] * 0 + 50
        
        df_ta['VWAP'] = ta.vwap(df_ta['High'], df_ta['Low'], df_ta['Close'], df_ta['Volume'])
        if df_ta['VWAP'] is None or df_ta['VWAP'].isnull().all() or is_index:
            df_ta['VWAP'] = df_ta['Close']

        df_ta['EMA_Cross_Up'] = (df_ta['EMA_5'] > df_ta['EMA_13']) & (df_ta['EMA_5'].shift(1) <= df_ta['EMA_13'].shift(1))
        df_ta['EMA_Cross_Dn'] = (df_ta['EMA_5'] < df_ta['EMA_13']) & (df_ta['EMA_5'].shift(1) >= df_ta['EMA_13'].shift(1))

        df_ta['Buy_Signal'] = df_ta['EMA_Cross_Up']
        df_ta['Sell_Signal'] = df_ta['EMA_Cross_Dn']
        
        df['vwap'] = df_ta['VWAP']
        df['ema_fast'] = df_ta['EMA_13']
        
        last = df_ta.iloc[-1]
        
        signal = "WAIT"
        trend = "RANGING 🟡 (VIJAY_RFF)"
        signal_strength = 50
        
        if last['Buy_Signal']:
            signal = "BUY_CE"
            trend = "VIJAY_RFF UPTREND CROSSOVER 🟢"
            signal_strength = 80
        elif last['Sell_Signal']:
            signal = "BUY_PE"
            trend = "VIJAY_RFF DOWNTREND CROSSOVER 🔴"
            signal_strength = 80
            
        return trend, signal, last['VWAP'], last['EMA_13'], df, atr, fib_data, signal_strength

    def apply_lux_algo_ict_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 50:
            return "WAIT (insufficient data)", "WAIT", 0, 0, df, 0, {}, 0
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum() if not is_index else df['close']
        
        df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
        df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])
        
        df['body'] = abs(df['close'] - df['open'])
        avg_body = df['body'].rolling(10).mean()
        df['bull_ob'] = (df['close'] < df['open']) & (df['body'] > avg_body * 1.2)
        df['bear_ob'] = (df['close'] > df['open']) & (df['body'] > avg_body * 1.2)
        
        df['hh'] = df['high'] > df['high'].shift(1)
        df['ll'] = df['low'] < df['low'].shift(1)
        df['bos_up'] = df['high'] > df['high'].rolling(20).max().shift(1)
        df['bos_down'] = df['low'] < df['low'].rolling(20).min().shift(1)
        
        last = df.iloc[-1]
        atr = self.get_atr(df).iloc[-1]
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        smc_blocks = self.detect_order_blocks(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high, **smc_blocks}
        
        signal = "WAIT"
        trend = "ICT Neutral"
        signal_strength = 50
        
        if last['fvg_bull'] and last['close'] > last['vwap'] and last['bos_up']:
            signal = "BUY_CE"
            trend = "LUX ICT BULLISH: FVG + OB + BOS 🟢"
            signal_strength = 85
        elif last['fvg_bear'] and last['close'] < last['vwap'] and last['bos_down']:
            signal = "BUY_PE"
            trend = "LUX ICT BEARISH: FVG + OB + BOS 🔴"
            signal_strength = 85
        
        return trend, signal, last['vwap'], last['ema9'], df, atr, fib_data, signal_strength

    def apply_vwap_ema_strategy(self, df, index_name="NIFTY"):
        if df is None or len(df) < 20: return "WAIT", "WAIT", 0, 0, df, 0, {}, 0
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
        signal_strength = 50
        
        benchmark = last['ema_long'] if is_index else last['vwap']
        if (last['ema_short'] > last['ema_long']) and (last['close'] > benchmark):
            trend, signal = "BULLISH MOMENTUM 🟢", "BUY_CE"
            signal_strength = 70
        elif (last['ema_short'] < last['ema_long']) and (last['close'] < benchmark):
            trend, signal = "BEARISH MOMENTUM 🔴", "BUY_PE"
            signal_strength = 70
            
        return trend, signal, last['vwap'], last['ema_short'], df, atr, fib_data, signal_strength

    def apply_keyword_strategy(self, df, keywords, index_name):
        if df is None or len(df) < 30: return "WAIT", "WAIT", 0, 0, df, 0, {}, 0
        df = self.calculate_indicators(df, index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"])
        
        last = df.iloc[-1]
        prev = df.iloc[-2]

        buy_conds, sell_conds = [], []
        keys = keywords.split(',') if keywords else []
        signal_strength = 50

        if "EMA Crossover (9 & 21)" in keys:
            buy_conds.append(last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21'])
            sell_conds.append(last['ema9'] < last['ema21'] and prev['ema9'] >= prev['ema21'])

        if "RSI Breakout (>60/<40)" in keys:
            buy_conds.append(last['rsi'] > 50)
            sell_conds.append(last['rsi'] < 50)

        if "MACD Crossover" in keys:
            if 'macd' in df.columns:
                buy_conds.append(last['macd'] > last['macds'] and prev['macd'] <= prev['macds'])
                sell_conds.append(last['macd'] < last['macds'] and prev['macd'] >= prev['macds'])

        if "Bollinger Bands Bounce" in keys:
            if 'bbl' in df.columns:
                buy_conds.append(last['close'] > last['bbl'] and prev['close'] <= prev['bbl'])
                sell_conds.append(last['close'] < last['bbh'] and prev['close'] >= prev['bbh'])

        if "Stochastic RSI" in keys:
            if 'stoch_k' in df.columns:
                buy_conds.append(last['stoch_k'] > last['stoch_d'] and last['stoch_k'] < 30)
                sell_conds.append(last['stoch_k'] < last['stoch_d'] and last['stoch_k'] > 70)

        if "FVG ICT" in keys:
            if 'fvg_bull' in df.columns and df['fvg_bull'].iloc[-1]:
                buy_conds.append(True)
            if 'fvg_bear' in df.columns and df['fvg_bear'].iloc[-1]:
                sell_conds.append(True)

        if "VWAP" in keys:
            buy_conds.append(last['close'] > last['vwap'])
            sell_conds.append(last['close'] < last['vwap'])

        signal, trend = "WAIT", "Awaiting Keyword Match 🟡"

        if buy_conds and all(buy_conds):
            signal, trend = "BUY_CE", "Keyword Setup Met: BULLISH 🟢"
            signal_strength = 70 + (len(buy_conds) * 5)
        elif sell_conds and all(sell_conds):
            signal, trend = "BUY_PE", "Keyword Setup Met: BEARISH 🔴"
            signal_strength = 70 + (len(sell_conds) * 5)

        signal_strength = min(100, signal_strength)

        return trend, signal, last['vwap'], last['ema9'], df, self.get_atr(df).iloc[-1], {}, signal_strength

    def apply_ml_strategy(self, df, index_name, prob_threshold=0.3, persistence=1):
        global ml_predictor
        if df is None or len(df) < 50:
            return "WAIT (insufficient data)", "WAIT", 0, 0, df, 0, {}, 0
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"]
        df = self.calculate_indicators(df, is_index)
        
        if not ml_predictor.is_trained:
            ml_predictor.train(df)
        prob_up, prob_down = ml_predictor.predict(df)
        
        last = df.iloc[-1]
        atr = self.get_atr(df).iloc[-1]
        
        up_thresh = prob_threshold
        down_thresh = prob_threshold
        
        trend = f"ML Ensemble: Up {prob_up:.2f} / Down {prob_down:.2f}"
        signal = "WAIT"
        strength = int(max(prob_up, prob_down) * 100)
        if prob_up > up_thresh:
            signal = "BUY_CE"
        elif prob_down > down_thresh:
            signal = "BUY_PE"
        
        mh, ml, f_low, f_high = self.calculate_fib_zones(df)
        fib_data = {"major_high": mh, "major_low": ml, "fib_low": f_low, "fib_high": f_high}
        return trend, signal, last['close'], last['close'], df, atr, fib_data, strength

# ==========================================
# MACHINE LEARNING ENSEMBLE
# ==========================================
class MLPredictor:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.mlp_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_train_index = 0
        self.feature_cols = None
        self.signal_history = deque(maxlen=10)

    def prepare_features(self, df):
        df = df.copy()
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_3'] = df['close'].pct_change(3)
        df['returns_5'] = df['close'].pct_change(5)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['sma5'] = df['close'].rolling(5).mean()
        df['sma20'] = df['close'].rolling(20).mean()
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        
        if HAS_PTA:
            df['rsi'] = ta.rsi(df['close'], 14)
        else:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        if HAS_PTA:
            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None:
                df['bb_lower'] = bbands.iloc[:, 0]
                df['bb_middle'] = bbands.iloc[:, 1]
                df['bb_upper'] = bbands.iloc[:, 2]
            else:
                df['bb_lower'] = df['bb_middle'] = df['bb_upper'] = df['close']
        else:
            df['bb_lower'] = df['bb_middle'] = df['bb_upper'] = df['close']
        
        future_return = df['close'].shift(-3) / df['close'] - 1
        df['target'] = 0
        df.loc[future_return > 0.005, 'target'] = 1
        df.loc[future_return < -0.005, 'target'] = 2
        
        df = df.dropna()
        self.feature_cols = ['returns_1', 'returns_3', 'returns_5', 'volume_ratio', 
                            'high_low_ratio', 'close_open_ratio', 'sma5', 'sma20',
                            'ema9', 'ema21', 'volatility_10', 'volatility_20',
                            'rsi', 'bb_lower', 'bb_middle', 'bb_upper']
        X = df[self.feature_cols]
        y = df['target']
        return X, y

    def train(self, df, force=False):
        if len(df) < 50:
            return False
        train_df = df.iloc[-500:].copy()
        X, y = self.prepare_features(train_df)
        if len(X) < 30:
            return False
        X_scaled = self.scaler.fit_transform(X)
        
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.rf_model.fit(X_scaled, y)
        
        if HAS_XGB:
            self.xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                                               random_state=42, use_label_encoder=False, eval_metric='mlogloss')
            self.xgb_model.fit(X_scaled, y)
        
        self.mlp_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        self.mlp_model.fit(X_scaled, y)
        
        self.is_trained = True
        self.last_train_index = len(df) - 1
        print(f"✅ ML Ensemble trained on {len(X)} samples")
        return True

    def predict(self, df):
        if not self.is_trained:
            return 0.5, 0.5
        X, _ = self.prepare_features(df.iloc[-100:])
        if X.empty:
            return 0.5, 0.5
        X_scaled = self.scaler.transform(X.iloc[-1:])
        
        prob_up_list = []
        prob_down_list = []
        
        if self.rf_model is not None:
            proba = self.rf_model.predict_proba(X_scaled)[0]
            if len(proba) == 3:
                prob_up_list.append(proba[1])
                prob_down_list.append(proba[2])
            elif len(proba) == 2:
                prob_up_list.append(proba[1])
                prob_down_list.append(1 - proba[1])
            else:
                prob_up_list.append(0.5); prob_down_list.append(0.5)
        
        if HAS_XGB and self.xgb_model is not None:
            proba = self.xgb_model.predict_proba(X_scaled)[0]
            if len(proba) == 3:
                prob_up_list.append(proba[1]); prob_down_list.append(proba[2])
            elif len(proba) == 2:
                prob_up_list.append(proba[1]); prob_down_list.append(1 - proba[1])
            else:
                prob_up_list.append(0.5); prob_down_list.append(0.5)
        
        if self.mlp_model is not None:
            proba = self.mlp_model.predict_proba(X_scaled)[0]
            if len(proba) == 3:
                prob_up_list.append(proba[1]); prob_down_list.append(proba[2])
            elif len(proba) == 2:
                prob_up_list.append(proba[1]); prob_down_list.append(1 - proba[1])
            else:
                prob_up_list.append(0.5); prob_down_list.append(0.5)
        
        prob_up = np.mean(prob_up_list) if prob_up_list else 0.5
        prob_down = np.mean(prob_down_list) if prob_down_list else 0.5
        
        pred_class = 1 if prob_up > prob_down else 2 if prob_down > prob_up else 0
        self.signal_history.append(pred_class)
        
        return prob_up, prob_down

    def should_retrain(self, df):
        if not self.is_trained:
            return True
        if len(df) - self.last_train_index >= 50:
            return True
        return False

ml_predictor = MLPredictor()

# ==========================================
# HIGH PROFIT STRATEGY MODULES
# ==========================================
class ArbitrageModule:
    def __init__(self):
        self.pairs = {
            "NIFTY_BANKNIFTY": ("NIFTY", "BANKNIFTY"),
            "GOLD_SILVER": ("GOLD", "SILVER"),
            "BTC_ETH": ("BTCUSD", "ETHUSD"),
            "EUR_USD": ("EURUSD", "USDX")
        }
        
    def detect_arbitrage(self, prices):
        opportunities = []
        
        if "NIFTY" in prices and "BANKNIFTY" in prices:
            nifty = prices["NIFTY"]
            bank = prices["BANKNIFTY"]
            ratio = bank / nifty
            historical_ratio = 4.2
            
            if ratio > historical_ratio * 1.02:
                opportunities.append({
                    "type": "PAIR_TRADE",
                    "action": "SELL_BANK_BUY_NIFTY",
                    "entry_ratio": ratio,
                    "target_ratio": historical_ratio,
                    "profit_potential": f"{(ratio/historical_ratio - 1)*100:.2f}%"
                })
            elif ratio < historical_ratio * 0.98:
                opportunities.append({
                    "type": "PAIR_TRADE",
                    "action": "BUY_BANK_SELL_NIFTY",
                    "entry_ratio": ratio,
                    "target_ratio": historical_ratio,
                    "profit_potential": f"{(historical_ratio/ratio - 1)*100:.2f}%"
                })
                
        return opportunities

class OptionSellingModule:
    def __init__(self, bot):
        self.bot = bot
        
    def find_premium_opportunities(self, symbol, spot, expiry_days):
        if symbol in ["NIFTY", "BANKNIFTY"]:
            strikes = []
            for i in range(1, 6):
                ce_strike = round(spot / 100) * 100 + (i * 100)
                pe_strike = round(spot / 100) * 100 - (i * 100)
                strikes.extend([ce_strike, pe_strike])
                
            opportunities = []
            for strike in strikes:
                premium = spot * 0.01 * (1 / (expiry_days + 1))
                
                if premium > spot * 0.005:
                    roi = (premium / (spot * 0.1)) * 100
                    
                    if roi > 10:
                        opportunities.append({
                            "strike": strike,
                            "type": "CE" if strike > spot else "PE",
                            "premium": premium,
                            "roi": roi,
                            "days_to_expiry": expiry_days,
                            "theta_decay": premium / expiry_days
                        })
            
            return opportunities
        return []

class MultiLegStrategies:
    def iron_condor(self, spot, iv, days_to_expiry):
        call_buy = round(spot * 1.1 / 100) * 100
        call_sell = round(spot * 1.05 / 100) * 100
        put_sell = round(spot * 0.95 / 100) * 100
        put_buy = round(spot * 0.9 / 100) * 100
        
        premium_received = (iv * 0.3) * spot
        max_loss = (call_buy - call_sell) * 50 - premium_received
        
        return {
            "strategy": "IRON_CONDOR",
            "legs": [
                f"SELL {call_sell} CE",
                f"BUY {call_buy} CE",
                f"SELL {put_sell} PE",
                f"BUY {put_buy} PE"
            ],
            "premium_received": premium_received,
            "max_loss": max_loss,
            "max_profit": premium_received,
            "probability": 0.7,
            "roi": (premium_received / max_loss) * 100 if max_loss > 0 else 0
        }
    
    def straddle_strangle(self, spot, iv, earnings=False):
        if earnings:
            atm_strike = round(spot / 100) * 100
            premium = iv * spot * 0.5
            
            return {
                "strategy": "STRADDLE",
                "legs": [f"BUY {atm_strike} CE", f"BUY {atm_strike} PE"],
                "total_premium": premium * 2,
                "breakeven_up": spot + (premium * 2),
                "breakeven_down": spot - (premium * 2),
                "max_profit": "UNLIMITED",
                "max_loss": premium * 2
            }
        return None

def pin_bar_scanner(df, lookback=50):
    results = []
    for i in range(1, min(lookback, len(df))):
        candle = df.iloc[-i]
        
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        
        is_pin = (upper_wick > body * 1.5 and lower_wick < body * 0.5) or \
                 (lower_wick > body * 1.5 and upper_wick < body * 0.5)
        
        if is_pin:
            direction = "BUY" if lower_wick > body * 1.5 else "SELL"
            entry = candle['close']
            stop = candle['low'] if direction == "BUY" else candle['high']
            target = entry + (abs(entry - stop) * 2) if direction == "BUY" else entry - (abs(entry - stop) * 2)
            
            results.append({
                "Time": candle.name.strftime('%H:%M'),
                "Direction": f"🟢 {direction}" if direction == "BUY" else f"🔴 {direction}",
                "Entry": f"{entry:.2f}",
                "Stop": f"{stop:.2f}",
                "Target": f"{target:.2f}",
                "Risk/Reward": "1:2",
                "Confidence": "High" if body < candle['high'] - candle['low'] * 0.3 else "Medium"
            })
    
    return results[:5]

def compounding_calculator():
    st.subheader("📈 Compounding Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Initial Capital (₹)", 1000, 100000, 10000)
        daily_target = st.slider("Daily Target %", 0.5, 5.0, 2.0, 0.1)
    
    with col2:
        trading_days = st.number_input("Trading Days", 5, 252, 100)
        compound_frequency = st.selectbox("Compound", ["Daily", "Weekly", "Monthly"])
    
    if compound_frequency == "Daily":
        periods = trading_days
        rate = daily_target / 100
    elif compound_frequency == "Weekly":
        periods = trading_days // 5
        rate = (daily_target * 5) / 100
    else:
        periods = trading_days // 22
        rate = (daily_target * 22) / 100
    
    final_capital = initial_capital * ((1 + rate) ** periods)
    total_profit = final_capital - initial_capital
    
    st.metric("Final Capital", f"₹{final_capital:,.0f}", 
              f"{((final_capital/initial_capital-1)*100):.0f}% Return")
    st.metric("Total Profit", f"₹{total_profit:,.0f}")
    st.metric("Daily Average", f"₹{total_profit/trading_days:,.0f}")
    
    growth = []
    for i in range(1, periods + 1):
        growth.append(initial_capital * ((1 + rate) ** i))
    
    chart_data = pd.DataFrame({"Day": range(1, periods + 1), "Capital": growth})
    st.line_chart(chart_data.set_index("Day"))

def add_breakout_scalper():
    st.subheader("🚀 Volume Breakout Scalper")
    
    col1, col2 = st.columns(2)
    with col1:
        volume_threshold = st.slider("Volume Spike Threshold", 1.2, 5.0, 1.5, 0.1)
        breakout_period = st.selectbox("Breakout Period", [5, 10, 15, 20], index=1)
    
    with col2:
        profit_target = st.slider("Profit Target (Points)", 1, 50, 10)
        stop_loss = st.slider("Stop Loss (Points)", 1, 30, 5)
    
    if st.button("🔍 Scan Breakouts Now", use_container_width=True):
        with st.spinner("Scanning for breakouts..."):
            symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "BTCUSD", "XAUUSD"]
            results = []
            
            for sym in symbols:
                try:
                    if "USD" in sym:
                        df = yf.Ticker(YF_TICKERS.get(sym, sym)).history(period="1d", interval="1m")
                    else:
                        continue
                        
                    if not df.empty:
                        df['volume_ma'] = df['Volume'].rolling(20).mean()
                        df['volume_ratio'] = df['Volume'] / df['volume_ma']
                        df['price_range'] = df['High'] - df['Low']
                        df['range_ma'] = df['price_range'].rolling(10).mean()
                        
                        last = df.iloc[-1]
                        
                        if (last['volume_ratio'] > volume_threshold and
                            last['price_range'] > last['range_ma'] * 1.3):
                            
                            direction = "BUY" if last['Close'] > df['Close'].iloc[-2] else "SELL"
                            confidence = min(100, last['volume_ratio'] * 20)
                            
                            results.append({
                                "Symbol": sym,
                                "Direction": f"🟢 {direction}" if direction == "BUY" else f"🔴 {direction}",
                                "Volume Spike": f"{last['volume_ratio']:.1f}x",
                                "Entry": f"₹{last['Close']:.2f}",
                                "SL": f"₹{last['Close'] - stop_loss if direction == 'BUY' else last['Close'] + stop_loss:.2f}",
                                "TP": f"₹{last['Close'] + profit_target if direction == 'BUY' else last['Close'] - profit_target:.2f}",
                                "Confidence": f"{confidence:.0f}%"
                            })
                except:
                    continue
            
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                st.success(f"Found {len(results)} breakout opportunities!")
            else:
                st.info("No breakouts detected at this moment")

def gold_crypto_scalper(symbol="XAUUSD", interval="1m"):
    st.subheader(f"⚡ 1-Min Scalper: {symbol}")
    
    try:
        if symbol in ["XAUUSD", "GOLD"]:
            ticker = "GC=F"
        elif symbol == "BTCUSD":
            ticker = "BTC-USD"
        elif symbol == "ETHUSD":
            ticker = "ETH-USD"
        elif symbol == "SOLUSD":
            ticker = "SOL-USD"
        else:
            ticker = symbol
            
        df = yf.Ticker(ticker).history(period="1d", interval="1m")
        if df.empty:
            st.error("No data available")
            return
            
        df['ema9'] = df['Close'].ewm(span=9).mean()
        df['ema21'] = df['Close'].ewm(span=21).mean()
        
        df['tr'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['vol_ma'] = df['Volume'].rolling(20).mean()
        df['vol_spike'] = df['Volume'] > df['vol_ma'] * 1.2
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        buy_signal = (last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21'] and last['vol_spike'])
        sell_signal = (last['ema9'] < last['ema21'] and prev['ema9'] >= prev['ema21'] and last['vol_spike'])
        
        current_atr = last['atr']
        if "XAU" in symbol or "GOLD" in symbol:
            sl_points = current_atr * 1.5
            tp_points = current_atr * 3.0
        else:
            sl_points = current_atr * 1.2
            tp_points = current_atr * 2.5
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${last['Close']:.2f}", 
                     f"{((last['Close']/prev['Close']-1)*100):.2f}%")
        with col2:
            st.metric("ATR (14)", f"${current_atr:.2f}")
        with col3:
            st.metric("RSI (14)", f"{last['rsi']:.1f}")
        with col4:
            st.metric("Volume", f"{last['Volume']:.0f}")
        
        st.markdown("### 🎯 Entry Signals")
        sig_col1, sig_col2 = st.columns(2)
        
        with sig_col1:
            st.markdown("**🟢 BUY Setup**")
            if buy_signal:
                st.success(f"🚀 **BUY SIGNAL ACTIVE**")
                st.markdown(f"""
                - **Entry:** ${last['Close']:.2f}
                - **Stop Loss:** ${last['Close'] - sl_points:.2f}
                - **Target 1:** ${last['Close'] + tp_points:.2f}
                - **Target 2:** ${last['Close'] + tp_points*2:.2f}
                - **Risk/Reward:** 1:2
                """)
            else:
                st.info("No buy signal")
        
        with sig_col2:
            st.markdown("**🔴 SELL Setup**")
            if sell_signal:
                st.error(f"🩸 **SELL SIGNAL ACTIVE**")
                st.markdown(f"""
                - **Entry:** ${last['Close']:.2f}
                - **Stop Loss:** ${last['Close'] + sl_points:.2f}
                - **Target 1:** ${last['Close'] - tp_points:.2f}
                - **Target 2:** ${last['Close'] - tp_points*2:.2f}
                - **Risk/Reward:** 1:2
                """)
            else:
                st.info("No sell signal")
        
        st.markdown("### 📊 Market Structure")
        struct_col1, struct_col2, struct_col3 = st.columns(3)
        with struct_col1:
            st.markdown(f"**Support:** ${df['Low'].tail(20).min():.2f}")
        with struct_col2:
            st.markdown(f"**Resistance:** ${df['High'].tail(20).max():.2f}")
        with struct_col3:
            trend = "🟢 UPTREND" if last['ema9'] > last['ema21'] else "🔴 DOWNTREND"
            st.markdown(f"**Trend:** {trend}")
        
        chart_data = df[['Close', 'ema9', 'ema21']].tail(60)
        st.line_chart(chart_data)
        
    except Exception as e:
        st.error(f"Error in scalper: {e}")

# ==========================================
# CORE BOT ENGINE (with FOMO integration & trailing stop with min profit)
# ==========================================
class SniperBot:
    def __init__(self, api_key="", client_id="", pwd="", totp_secret="", tg_token="", tg_chat="", wa_phone="", wa_api="", mt5_acc="", mt5_pass="", mt5_server="", mt5_api_url="", zerodha_api="", zerodha_secret="", request_token="", coindcx_api="", coindcx_secret="", delta_api="", delta_secret="", ib_host="", ib_port=7497, ib_client_id=1, is_mock=False):
        self.api_key, self.client_id, self.pwd, self.totp_secret = api_key, client_id, pwd, totp_secret
        self.tg_token, self.tg_chat, self.wa_phone, self.wa_api = tg_token, tg_chat, wa_phone, wa_api
        self.mt5_acc, self.mt5_pass, self.mt5_server, self.mt5_api_url = mt5_acc, mt5_pass, mt5_server, mt5_api_url
        self.zerodha_api, self.zerodha_secret, self.request_token = zerodha_api, zerodha_secret, request_token
        self.coindcx_api, self.coindcx_secret = coindcx_api, coindcx_secret
        self.delta_api, self.delta_secret = delta_api, delta_secret
        self.ib_host, self.ib_port, self.ib_client_id = ib_host, ib_port, ib_client_id
        
        self.api, self.kite, self.token_map, self.is_mock = None, None, None, is_mock
        self.mt5_bridge = None
        self.ib_bridge = None
        self.is_mt5_connected = False
        self.is_ib_connected = False
        self.client_name = "Offline User"
        self.client_ip = get_client_ip()
        self.user_hash = get_user_hash(self.api_key)
        self.analyzer = TechnicalAnalyzer()
        self.scalper = ScalpingModule(self)
        self.arbitrage = ArbitrageModule()
        self.option_seller = OptionSellingModule(self)
        self.multi_leg = MultiLegStrategies()
        
        self.state = {
            "is_running": False, "order_in_flight": False, "active_trade": None, "last_trade": None,
            "logs": deque(maxlen=50), "current_trend": "WAIT", "current_signal": "WAIT",
            "signal_strength": 0, "spot": 0.0, "vwap": 0.0, "ema": 0.0, "atr": 0.0, "fib_data": {}, "latest_data": None,
            "latest_candle": None,
            "ui_popups": deque(maxlen=10), "loop_count": 0, "daily_pnl": 0.0, "trades_today": 0,
            "manual_exit": False,
            "ghost_memory": {},
            "tv_signal": {"action": "WAIT", "symbol": "", "timestamp": 0},
            "scalp_signals": deque(maxlen=20),
            "arbitrage_opps": [],
            "premium_opps": [],
            "news_cache": [],
            "signal_history": deque(maxlen=5),
            "mock_price": None,
            "sound_queue": deque(maxlen=10),
            "fomo_signals": deque(maxlen=20),
            "hz_trades": deque(maxlen=100),
            "hz_pnl": 0.0,
            "hz_wins": 0,
            "hz_losses": 0
        }
        self.settings = {}

    def load_daily_pnl(self):
        """Load today's total PnL from database (for real users)."""
        if self.is_mock or not HAS_DB:
            return 0.0
        today = get_ist().strftime('%Y-%m-%d')
        try:
            res = supabase.table("trade_logs").select("pnl").eq("user_id", self.system_user_id).eq("trade_date", today).execute()
            if res.data:
                total = sum(trade['pnl'] for trade in res.data)
                return total
        except Exception as e:
            self.log(f"⚠️ Could not load daily PnL: {e}")
        return 0.0

    def connect_mt5(self):
        if self.mt5_acc and self.mt5_server:
            self.mt5_bridge = MT5WebBridge(
                account=self.mt5_acc,
                password=self.mt5_pass,
                server=self.mt5_server,
                api_url=self.mt5_api_url
            )
            success, msg = self.mt5_bridge.connect()
            if success:
                self.is_mt5_connected = True
                self.log(f"✅ MT5 Connected via bridge: {msg}")
                return True
            else:
                self.log(f"❌ MT5 Bridge failed: {msg}")
        return False

    def connect_ib(self):
        if self.ib_host:
            self.ib_bridge = IBBridge(self.ib_host, self.ib_port, self.ib_client_id)
            success, msg = self.ib_bridge.connect()
            if success:
                self.is_ib_connected = True
                self.log(f"✅ Interactive Brokers Connected: {msg}")
                return True
            else:
                self.log(f"❌ IB Connection failed: {msg}")
        return False

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
        b_str = []
        
        if self.api:
            try:
                rms = self.api.rms()
                if rms and rms.get('status') and rms.get('data'):
                    data = rms['data']
                    bal = data.get('availablecash', data.get('net', 0))
                    try:
                        bal = float(bal)
                        b_str.append(f"Angel: ₹ {bal:,.2f}")
                    except:
                        pass
            except:
                pass
            
        if self.kite:
            try:
                margins = self.kite.margins()
                eq = margins.get('equity', {})
                bal = eq.get('available', {}).get('live_balance', eq.get('net', 0))
                try:
                    bal = float(bal)
                    b_str.append(f"Zerodha: ₹ {bal:,.2f}")
                except:
                    pass
            except:
                pass
            
        if self.coindcx_api:
            try:
                ts = int(round(time.time() * 1000))
                payload = {"timestamp": ts}
                secret_bytes = bytes(self.coindcx_secret, 'utf-8')
                signature = hmac.new(secret_bytes, json.dumps(payload, separators=(',', ':')).encode('utf-8'), hashlib.sha256).hexdigest()
                res = requests.post("https://api.coindcx.com/exchange/v1/users/balances", headers={'X-AUTH-APIKEY': self.coindcx_api, 'X-AUTH-SIGNATURE': signature}, json=payload, timeout=5)
                if res.status_code == 200:
                    for b in res.json():
                        if b['currency'] == 'USDT':
                            bal = float(b['balance'])
                            if self.settings.get('show_inr_crypto', True):
                                b_str.append(f"DCX: ₹ {bal * get_usdt_inr_rate():,.2f}")
                            else:
                                b_str.append(f"DCX: $ {bal:,.2f}")
            except:
                pass
            
        if self.delta_api:
            try:
                ts, sig = generate_delta_signature('GET', '/v2/wallet/balances', '', self.delta_secret)
                headers = {'api-key': self.delta_api, 'signature': sig, 'timestamp': ts}
                res = requests.get("https://api.delta.exchange/v2/wallet/balances", headers=headers, timeout=5)
                if res.status_code == 200:
                    for b in res.json().get('result', []):
                        if b['asset_symbol'] == 'USDT':
                            bal = float(b['balance'])
                            if self.settings.get('show_inr_crypto', True):
                                b_str.append(f"Delta: ₹ {bal * get_usdt_inr_rate():,.2f}")
                            else:
                                b_str.append(f"Delta: $ {bal:,.2f}")
            except:
                pass
            
        if self.is_mt5_connected and self.mt5_bridge:
            try:
                acc_info = self.mt5_bridge.get_account_info()
                if acc_info:
                    b_str.append(f"MT5: $ {acc_info.get('balance', 0):,.2f}")
            except:
                pass

        if self.is_ib_connected and self.ib_bridge:
            b_str.append("IB: connected")
            
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

        if self.mt5_acc and self.mt5_server:
            if self.connect_mt5():
                success = True

        if self.ib_host:
            if self.connect_ib():
                success = True

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

    def get_master(self):
        if self.token_map is None or self.token_map.empty: self.token_map = get_angel_scrip_master()
        return self.token_map

    def get_token_info(self, index_name):
        if index_name in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "SOLUSD"] and self.is_mt5_connected:
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
            if self.state.get('mock_price') is None:
                base_prices = {"NIFTY": 22000, "BANKNIFTY": 47000, "SENSEX": 73000, "NATURALGAS": 145.0, "CRUDEOIL": 6500.0, "GOLD": 62000.0, "SILVER": 72000.0, "XAUUSD": 2050.0, "EURUSD": 1.0850, "BTCUSD": 65000.0, "ETHUSD": 3500.0, "SOLUSD": 150.0}
                base = base_prices.get(symbol, 500)
                self.state['mock_price'] = float(base)
            change = np.random.normal(0, self.state['mock_price'] * 0.0005)
            self.state['mock_price'] += change
            return float(self.state['mock_price'])
            
        if exchange == "MT5" and self.is_mt5_connected and self.mt5_bridge:
            return self.mt5_bridge.get_live_price(symbol)
            
        if exchange == "COINDCX" and self.coindcx_api:
            try:
                market_symbol = symbol.replace("USD", "USDT") if symbol.endswith("USD") and not symbol.endswith("USDT") else symbol
                res = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5).json()
                for coin in res:
                    if coin['market'] == market_symbol:
                        return float(coin['last_price'])
                for coin in res:
                    if coin['market'].upper() == market_symbol.upper():
                        return float(coin['last_price'])
                for coin in res:
                    if coin['market'].replace('_', '').upper() == market_symbol.upper():
                        return float(coin['last_price'])
            except Exception as e:
                self.log(f"⚠️ CoinDCX price fetch error: {e}")
                return None
            
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

        if self.is_ib_connected and self.ib_bridge:
            try:
                return self.ib_bridge.get_live_price(symbol)
            except:
                pass
        return None

    def get_historical_data(self, exchange, token, symbol="NIFTY", interval="5m"):
        if self.is_mock and token == "12345": return self._fallback_yfinance(symbol, interval)
        if exchange == "MT5" and self.is_mt5_connected and self.mt5_bridge:
            return self.mt5_bridge.get_historical_data(symbol, interval)
            
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
        
        periods = 500 if interval == "1m" else 200
        times = pd.date_range(end=get_ist(), periods=periods, freq=interval)
        trend = np.linspace(0, 1, periods) * 200
        noise = np.random.normal(0, 10, periods).cumsum()
        close_prices = 22000 + trend + noise
        df = pd.DataFrame({
            'timestamp': times,
            'open': close_prices - 2,
            'high': close_prices + 5,
            'low': close_prices - 5,
            'close': close_prices,
            'volume': np.random.randint(1000, 50000, periods)
        })
        df.index = df['timestamp']
        return df

    def place_real_order(self, symbol, token, qty, side="BUY", exchange="NFO"):
        if self.is_mock: return "MOCK_" + uuid.uuid4().hex[:6].upper()
        broker = self.settings.get("primary_broker", "Angel One")
        
        formatted_qty = str(int(float(qty))) if exchange in ["NFO", "NSE", "BFO", "MCX"] else str(qty)
        self.log(f"⚙️ Executing Real API: {symbol} | Qty: {qty} | Side: {side} | Exchange: {exchange}")

        if exchange == "MT5" and self.is_mt5_connected and self.mt5_bridge:
            order_id, msg = self.mt5_bridge.place_order(symbol, qty, side)
            if order_id:
                self.log(f"✅ MT5 Order Success! ID: {order_id}")
                return order_id
            else:
                self.log(f"❌ MT5 Order Failed: {msg}")
                return None

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
                clean_qty = float(round(float(qty), 8))

                # Determine correct market symbol by checking actual ticker data
                ticker_data = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5).json()
                market_symbol = None
                for coin in ticker_data:
                    mkt = coin.get('market', '')
                    # Spot format: BTCUSDT, etc.
                    if mkt.upper() == f"{base_coin}USDT".upper():
                        market_symbol = mkt
                        break
                    # Futures format: B-BTC_USDT, etc.
                    if market_type in ["Futures", "Options"] and mkt.upper() == f"B-{base_coin}_USDT".upper():
                        market_symbol = mkt
                        break

                if not market_symbol:
                    # Fallback: construct based on market type
                    if market_type in ["Futures", "Options"]:
                        market_symbol = f"B-{base_coin}_USDT"
                    else:
                        market_symbol = f"{base_coin}USDT"
                    self.log(f"⚠️ CoinDCX market symbol not found, using constructed: {market_symbol}")

                # Choose endpoint based on market type
                if market_type in ["Futures", "Options"]:
                    payload = {
                        "side": side.lower(),
                        "order_type": "market_order",
                        "pair": market_symbol,
                        "total_quantity": clean_qty,
                        "timestamp": ts
                    }
                    endpoint = "https://api.coindcx.com/exchange/v1/derivatives/futures/orders/create"
                else:
                    payload = {
                        "side": side.lower(),
                        "order_type": "market_order",
                        "market": market_symbol,
                        "total_quantity": clean_qty,
                        "timestamp": ts
                    }
                    endpoint = "https://api.coindcx.com/exchange/v1/orders/create"

                payload_str = json.dumps(payload, separators=(',', ':'))
                secret_bytes = bytes(self.coindcx_secret, 'utf-8')
                signature = hmac.new(secret_bytes, payload_str.encode('utf-8'), hashlib.sha256).hexdigest()

                headers = {
                    'X-AUTH-APIKEY': self.coindcx_api,
                    'X-AUTH-SIGNATURE': signature,
                    'Content-Type': 'application/json'
                }

                self.log(f"📡 CoinDCX payload: {payload_str}")
                res = requests.post(endpoint, headers=headers, data=payload_str, timeout=10)

                if res.status_code == 200:
                    response_data = res.json()
                    if isinstance(response_data, list) and len(response_data) > 0:
                        order_id = response_data[0].get('id', 'DCX_ORDER_OK')
                    elif isinstance(response_data, dict):
                        order_id = response_data.get('id', response_data.get('order_id', 'DCX_ORDER_OK'))
                    else:
                        order_id = 'DCX_ORDER_OK'
                    self.log(f"✅ CoinDCX Order Success! ID: {order_id}")
                    return order_id
                else:
                    self.log(f"❌ CoinDCX API Rejected [{res.status_code}]: {res.text}")
                    return None
            except Exception as e:
                self.log(f"❌ CoinDCX Exception: {e}")
                return None

        if broker == "Zerodha" and self.kite:
            try:
                z_side = self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL
                order_id = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, exchange=exchange, tradingsymbol=symbol, transaction_type=z_side, quantity=int(float(qty)), product=self.kite.PRODUCT_MIS, order_type=self.kite.ORDER_TYPE_MARKET)
                self.log(f"✅ Zerodha Order Pushed! ID: {order_id}")
                return order_id
            except Exception as e: self.log(f"❌ Zerodha Order Error: {str(e)}"); return None

        if self.is_ib_connected and self.ib_bridge:
            order_id, msg = self.ib_bridge.place_order(symbol, int(qty), side)
            if order_id:
                self.log(f"✅ IB Order: {msg}")
                return order_id
            else:
                self.log(f"❌ IB Order Failed: {msg}")
                return None

        try: 
            p_type = "CARRYFORWARD" if exchange in ["NFO", "BFO", "MCX"] else "INTRADAY"
            
            order_type = "MARKET"
            exec_price = 0.0
            
            if exchange in ["NFO", "BFO"]:
                ltp = self.get_live_price(exchange, symbol, token)
                if ltp and ltp > 0:
                    order_type = "LIMIT"
                    if side.upper() == "BUY":
                        safe_price = ltp * 1.05
                    else:
                        safe_price = ltp * 0.95
                    
                    exec_price = round(round(safe_price / 0.05) * 0.05, 2)
                else:
                    self.log(f"⚠️ Could not fetch LTP for {symbol}. Retrying as pure MARKET.")

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
        
        candidates = subset.sort_values('dist_to_spot', ascending=True).head(10)
            
        for _, row in candidates.iterrows():
            ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
            if ltp is None and self.is_mock:
                ltp = max_premium * 0.85
            if ltp and ltp <= max_premium: 
                return row['symbol'], row['token'], row['exch_seg'], ltp

        if self.is_mock: return f"{symbol}28FEB{int(spot)}{opt_type}", "12345", "NFO", min(100.0, max_premium)
        return None, None, None, 0.0

    # Hero/Zero Scanner Methods (stocks only)
    def scan_hero_zero_indian_stocks(self, nifty_stocks=None):
        if nifty_stocks is None:
            nifty_stocks = [
                "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
                "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                "BAJFINANCE", "LT", "WIPRO", "AXISBANK", "TITAN",
                "ASIANPAINT", "MARUTI", "SUNPHARMA", "HCLTECH", "ULTRACEMCO"
            ]
        results = []
        for stock in nifty_stocks:
            try:
                ticker = f"{stock}.NS"
                df = yf.Ticker(ticker).history(period="1d", interval="5m")
                if df.empty or len(df) < 20:
                    continue
                df['ema9'] = df['Close'].ewm(span=9).mean()
                df['ema21'] = df['Close'].ewm(span=21).mean()
                df['volume_ma'] = df['Volume'].rolling(20).mean()
                
                high_low = df['High'] - df['Low']
                high_close = abs(df['High'] - df['Close'].shift())
                low_close = abs(df['Low'] - df['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
                
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                is_hero = (
                    last['Close'] > last['ema9'] and
                    last['ema9'] > last['ema21'] and
                    last['Volume'] > df['volume_ma'].iloc[-1] * 1.2 and
                    last['Close'] > prev['High'] * 0.99 and
                    (last['Close'] - last['Low']) > (last['High'] - last['Close']) * 1.5 and
                    (last['High'] - last['Low']) > atr * 0.4
                )
                
                is_zero = (
                    last['Close'] < last['ema9'] and
                    last['ema9'] < last['ema21'] and
                    last['Volume'] > df['volume_ma'].iloc[-1] * 1.2 and
                    last['Close'] < prev['Low'] * 1.01 and
                    (last['High'] - last['Close']) > (last['Close'] - last['Low']) * 1.5 and
                    (last['High'] - last['Low']) > atr * 0.4
                )
                
                if is_hero or is_zero:
                    direction = "HERO (BUY)" if is_hero else "ZERO (SELL)"
                    if is_hero:
                        entry = last['Close']
                        sl = last['Close'] - atr * 1.5
                        tp1 = last['Close'] + atr * 3
                        tp2 = last['Close'] + atr * 5
                    else:
                        entry = last['Close']
                        sl = last['Close'] + atr * 1.5
                        tp1 = last['Close'] - atr * 3
                        tp2 = last['Close'] - atr * 5
                    
                    results.append({
                        "Stock": stock,
                        "Direction": direction,
                        "Price": round(entry, 2),
                        "Volume Spike": f"{last['Volume'] / df['volume_ma'].iloc[-1]:.1f}x",
                        "ATR": round(atr, 2),
                        "Entry": round(entry, 2),
                        "SL": round(sl, 2),
                        "Target 1": round(tp1, 2),
                        "Target 2": round(tp2, 2),
                        "Risk/Reward": "1:2"
                    })
            except Exception as e:
                continue
        return pd.DataFrame(results)

    def scan_penny_stocks(self):
        penny_list = [
            "IDEA", "YESBANK", "SAIL", "PNB", "IOC", "BHEL", "SUZLON", "JPASSOCIAT",
            "GMRINFRA", "NHPC", "NTPC", "PFC", "RECLTD", "VEDL", "TATAMOTORS"
        ]
        return self.scan_hero_zero_indian_stocks(penny_list)

    def scan_pin_bars(self):
        symbols = {
            "NIFTY": "^NSEI",
            "SENSEX": "^BSESN",
            "BANKNIFTY": "^NSEBANK",
            "GOLD": "GC=F"
        }
        results = []
        for name, ticker in symbols.items():
            try:
                df = yf.Ticker(ticker).history(period="1d", interval="5m")
                if df.empty or len(df) < 20:
                    continue
                df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                pins = pin_bar_scanner(df, lookback=10)
                for pin in pins:
                    pin["Symbol"] = name
                    results.append(pin)
            except:
                continue
        return results

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
                
                lot_size = LOT_SIZES.get(index, 1)
                actual_qty = s['lots'] * lot_size
                
                if df_candles is not None and not df_candles.empty:
                    scalp_signal = self.scalper.scalp_signal(df_candles)
                    if scalp_signal != "WAIT":
                        self.state["scalp_signals"].append({
                            "time": time_str,
                            "signal": scalp_signal,
                            "price": spot,
                            "index": index
                        })
                
                fomo_signal = None
                fomo_signals = fomo_scanner.scan()
                for fs in fomo_signals:
                    asset_key = index.replace(".NS", "").upper()
                    if asset_key in fs['symbol'].upper() or fs['symbol'].upper() in asset_key:
                        fomo_signal = fs
                        break
                if fomo_signal:
                    self.state["fomo_signals"].append(fomo_signal)
                    if self.state["active_trade"] is None:
                        if fomo_signal['signal'].startswith("BUY"):
                            signal = "BUY_CE"
                            trend = f"FOMO BUY on {fomo_signal['symbol']}"
                            signal_strength = 90
                        elif fomo_signal['signal'].startswith("SELL"):
                            signal = "BUY_PE"
                            trend = f"FOMO SELL on {fomo_signal['symbol']}"
                            signal_strength = 90
                        else:
                            signal = "WAIT"
                            trend = "FOMO neutral"
                            signal_strength = 50
                    else:
                        signal = "WAIT"
                        trend = "FOMO ignored (trade active)"
                        signal_strength = 0
                elif spot and df_candles is not None and not df_candles.empty:
                    self.state["spot"] = spot
                    last_candle = df_candles.iloc[-1]
                    self.state["latest_candle"] = last_candle.to_dict()
                    
                    if strategy == "Keyword Rule Builder":
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_keyword_strategy(df_candles, s.get('custom_code', ''), index)
                    elif strategy == "Machine Learning":
                        if ml_predictor.should_retrain(df_candles):
                            ml_predictor.train(df_candles)
                        prob_threshold = s.get('ml_prob_threshold', 0.30)
                        persistence = s.get('signal_persistence', 1)
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_ml_strategy(
                            df_candles, index, prob_threshold, persistence
                        )
                        self.log(f"ML Prob: Up={trend.split()[2]}, Down={trend.split()[4]}, Signal={signal}")
                    elif "VIJAY & RFF" in strategy: 
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_vijay_rff_strategy(df_candles, index)
                    elif "Institutional FVG" in strategy: 
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_institutional_fvg_strategy(df_candles, index)
                    elif "Lux Algo" in strategy:
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_lux_algo_ict_strategy(df_candles, index)
                    else:  # Momentum Breakout + S&R
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_vwap_ema_strategy(df_candles, index)

                    if strategy == "Machine Learning" and signal != "WAIT" and s.get('signal_persistence', 1) > 1:
                        self.state["signal_history"].append(signal)
                        persistence = s.get('signal_persistence', 1)
                        if len(self.state["signal_history"]) >= persistence:
                            recent = list(self.state["signal_history"])[-persistence:]
                            if all(x == signal for x in recent):
                                pass
                            else:
                                signal = "WAIT"
                                trend += " | ⏳ Persistence filter"
                                signal_strength = 0
                    else:
                        self.state["signal_history"].clear()

                    if s.get("mtf_confirm") and signal != "WAIT" and strategy != "TradingView Webhook":
                        df_htf = self.get_historical_data(exch, token, symbol=index, interval="15m") if not self.is_mock else self.get_historical_data("MOCK", "12345", symbol=index, interval="15m")
                        if df_htf is not None and len(df_htf) > 5:
                            htf_ema = df_htf['close'].ewm(span=9).mean().iloc[-1]
                            htf_close = df_htf['close'].iloc[-1]
                            if signal == "BUY_CE" and htf_close < htf_ema: 
                                signal = "WAIT"
                                trend = "MTF Blocked: 15m Bearish"
                                signal_strength = 0
                            elif signal == "BUY_PE" and htf_close > htf_ema: 
                                signal = "WAIT"
                                trend = "MTF Blocked: 15m Bullish"
                                signal_strength = 0

                else:
                    trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = "Waiting for Market Data", "WAIT", 0, 0, df_candles, 0, {}, 0

                if df_chart is not None and hasattr(df_chart, 'columns'):
                    temp_df = self.analyzer.calculate_indicators(df_chart, index in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"])
                    for col in temp_df.columns:
                        if col not in df_chart.columns:
                            df_chart[col] = temp_df[col]

                self.state.update({
                    "current_trend": trend, 
                    "current_signal": signal, 
                    "signal_strength": signal_strength,
                    "vwap": vwap, 
                    "ema": ema, 
                    "atr": current_atr, 
                    "fib_data": fib_data, 
                    "latest_data": df_chart
                })

                if self.state["active_trade"] is None and signal in ["BUY_CE", "BUY_PE"] and current_time < cutoff_time and signal_strength >= s.get('min_signal_strength', 50):
                    
                    qty = actual_qty
                    
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

                        if s.get('three_five_seven', False) and current_atr > 0:
                            if trade_type in ["CE", "BUY"]:
                                dynamic_sl = entry_ltp - current_atr * 1.5
                                tp1 = entry_ltp + current_atr * 3
                                tp2 = entry_ltp + current_atr * 5
                                tp3 = entry_ltp + current_atr * 7
                            else:
                                dynamic_sl = entry_ltp + current_atr * 1.5
                                tp1 = entry_ltp - current_atr * 3
                                tp2 = entry_ltp - current_atr * 5
                                tp3 = entry_ltp - current_atr * 7
                        else:
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
                            "scaled_out": False,
                            "booked_50": False, "booked_80": False
                        }

                        if not is_mock_mode: 
                            exec_side = "SELL" if new_trade['type'] == "SELL" else "BUY"
                            self.place_real_order(strike_sym, strike_token, qty, exec_side, strike_exch)
                            
                        self.push_notify("Trade Entered", f"Entered {qty} {strike_sym} @ {entry_ltp} (Signal Strength: {signal_strength}%)")
                        self.state["sound_queue"].append("entry")
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
                        
                        # Minimum profit threshold for trailing stop (0.5% of entry)
                        min_profit = trade['entry'] * 0.005
                        
                        if trade['type'] == "SELL":
                            lowest = trade.get('lowest_price', trade['entry'])
                            if ltp < lowest:
                                trade['lowest_price'] = ltp
                                if (trade['entry'] - ltp) > min_profit:
                                    tsl_buffer = s['tsl_pts'] * 1.5 if "Trend Rider" in strategy else s['tsl_pts']
                                    new_sl = ltp + tsl_buffer
                                    if new_sl < trade['sl']: trade['sl'] = new_sl
                            
                            hit_tp = False if ("Trend Rider" in strategy) else (ltp <= trade['tgt'])
                            hit_sl = ltp >= trade['sl']
                        else:
                            highest = trade.get('highest_price', trade['entry'])
                            if ltp > highest:
                                trade['highest_price'] = ltp
                                if (ltp - trade['entry']) > min_profit:
                                    tsl_buffer = s['tsl_pts'] * 1.5 if "Trend Rider" in strategy else s['tsl_pts']
                                    new_sl = ltp - tsl_buffer
                                    if new_sl > trade['sl']: trade['sl'] = new_sl
                                
                            hit_tp = False if ("Trend Rider" in strategy) else (ltp >= trade['tgt'])
                            hit_sl = ltp <= trade['sl']

                        if not trade['scaled_out']:
                            reach_tp1 = (ltp <= trade['tp1']) if trade['type'] == "SELL" else (ltp >= trade['tp1'])
                            if reach_tp1:
                                if index in ["NIFTY", "SENSEX", "XAUUSD"] or is_crypto:
                                    lots_held = trade['qty'] / lot_size
                                    half_lots = int(lots_held / 2) if not (is_mt5_asset or is_crypto) else round(trade['qty']/2, 2)
                                    if half_lots > 0:
                                        qty_to_sell = half_lots * lot_size if not (is_mt5_asset or is_crypto) else half_lots
                                        if not is_mock_mode:
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
                        
                        if hit_tp or hit_sl or market_close:
                            if not is_mock_mode: 
                                exec_side = "BUY" if trade['type'] == "SELL" else "SELL"
                                self.place_real_order(trade['symbol'], trade['token'], trade['qty'], exec_side, trade['exch'])
                            
                            if hit_tp:
                                self.state["sound_queue"].append("tp")
                            elif hit_sl:
                                self.state["sound_queue"].append("sl")
                            
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
                            
                            if market_close: win_text += " (Force Exit)"
                            if trade['scaled_out']: win_text += " (Scaled Out)"
                            
                            self.log(f"🛑 EXIT {trade['symbol']} | PnL: {round(pnl, 2)} [{win_text}]")
                            self.push_notify("Trade Closed", f"Closed {trade['symbol']} | PnL: {round(pnl, 2)}")
                            
                            if not self.is_mock: 
                                user_id = getattr(self, "system_user_id", self.api_key)
                                save_trade(user_id, today_date, time_str, trade['symbol'], trade['type'], trade['qty'], trade['entry'], ltp, round(pnl, 2), win_text)
                            else:
                                if "paper_history" not in self.state:
                                    self.state["paper_history"] = []
                                self.state["paper_history"].append({
                                    "Date": today_date,
                                    "Time": time_str,
                                    "Symbol": trade['symbol'],
                                    "Type": trade['type'],
                                    "Qty": trade['qty'],
                                    "Entry Price": trade['entry'],
                                    "Exit Price": ltp,
                                    "PnL": round(pnl, 2),
                                    "Result": win_text
                                })
                            
                            self.state["last_trade"] = trade.copy()
                            self.state["last_trade"].update({"exit_price": ltp, "final_pnl": pnl, "win_text": win_text})
                            self.state["daily_pnl"] += pnl
                            self.state["active_trade"] = None
            except Exception as e:
                self.log(f"⚠️ Loop Error: {str(e)}")
            time.sleep(2)

# ==========================================
# STREAMLIT UI - LOGIN SCREEN
# ==========================================
if getattr(st.session_state, "bot", None) and st.session_state.bot.state.get("ui_popups"):
    while st.session_state.bot.state["ui_popups"]:
        alert = st.session_state.bot.state["ui_popups"].popleft()
        st.toast(alert.get("message", ""), icon="🔔")

# Play sounds from queue
if getattr(st.session_state, "bot", None) and st.session_state.bot.state.get("sound_queue"):
    while st.session_state.bot.state["sound_queue"]:
        sound_type = st.session_state.bot.state["sound_queue"].popleft()
        play_sound_ui(sound_type)

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
                keep_signed = st.checkbox("Keep me signed in (auto‑login using URL)")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("👆 Authenticate & Connect", type="primary", use_container_width=True):
                    creds = load_creds(USER_ID)
                    if creds and (creds.get("client_id") or creds.get("zerodha_api") or creds.get("coindcx_api") or creds.get("delta_api")):
                        temp_bot = SniperBot(
                            api_key=creds.get("angel_api", ""), client_id=creds.get("client_id"), pwd=creds.get("pwd"), 
                            totp_secret=creds.get("totp_secret"), mt5_acc=creds.get("mt5_acc"), 
                            mt5_pass=creds.get("mt5_pass"), mt5_server=creds.get("mt5_server"), mt5_api_url=creds.get("mt5_api_url", ""),
                            zerodha_api=creds.get("zerodha_api"), zerodha_secret=creds.get("zerodha_secret"),
                            coindcx_api=creds.get("coindcx_api"), coindcx_secret=creds.get("coindcx_secret"),
                            delta_api=creds.get("delta_api"), delta_secret=creds.get("delta_secret"),
                            ib_host=creds.get("ib_host", ""), ib_port=creds.get("ib_port", 7497), ib_client_id=creds.get("ib_client_id", 1),
                            is_mock=False
                        )
                        temp_bot.system_user_id = USER_ID
                        with st.spinner("Authenticating via Cloud..."):
                            if temp_bot.login():
                                temp_bot.state["daily_pnl"] = temp_bot.load_daily_pnl()
                                st.session_state.bot = temp_bot
                                st.session_state.audio_enabled = True
                                unlock_audio()
                                play_sound_ui("entry")
                                if keep_signed:
                                    st.query_params["user_id"] = USER_ID
                                    play_sound_ui("entry")
                                    st.rerun()
                            else: st.error("❌ Login Failed! Check API details or TOTP.")
                    else: st.error("❌ Profile not found! Please save it once via the Real Trading menu.")
            elif auth_mode == "🕉️ Real Trading":
                USER_ID = st.text_input("System Login ID (Email or Phone Number)")
                creds = load_creds(USER_ID) if USER_ID else {}
                st.markdown("### 🏦 Select Brokers to Connect")
                ANGEL_API, CLIENT_ID, PIN, TOTP = "", "", "", ""
                Z_API, Z_SEC, Z_REQ = "", "", ""
                MT5_ACC, MT5_PASS, MT5_SERVER, MT5_API_URL = "", "", "", ""
                DCX_API, DCX_SEC = "", ""
                DELTA_API, DELTA_SEC = "", ""
                IB_HOST, IB_PORT, IB_CLIENT_ID = "", 7497, 1
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
                    with col_t: use_mt5 = st.toggle("MetaTrader 5 (Web Bridge - No Desktop Required)", value=bool(creds.get("mt5_acc")))
                    if use_mt5:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1: MT5_ACC = st.text_input("MT5 Account ID", value=creds.get("mt5_acc", ""))
                        with col_m2: MT5_PASS = st.text_input("MT5 Password", type="password", value=creds.get("mt5_pass", ""))
                        MT5_SERVER = st.text_input("Broker Server", value=creds.get("mt5_server", ""))
                        MT5_API_URL = st.text_input("MT5 Web API URL (Optional)", value=creds.get("mt5_api_url", "https://mt5-web-api.mtapi.io/v1"), 
                                                    help="Use a web API service like mtapi.io or your own MT5 gateway")
                with st.container(border=True):
                    col_img, col_t = st.columns([1, 6])
                    with col_img: st.image("https://www.google.com/s2/favicons?domain=interactivebrokers.com&sz=128", width=40)
                    with col_t: use_ib = st.toggle("Interactive Brokers", value=bool(creds.get("ib_host")))
                    if use_ib:
                        col_i1, col_i2 = st.columns(2)
                        with col_i1: IB_HOST = st.text_input("IB Gateway/TWS Host", value=creds.get("ib_host", "127.0.0.1"))
                        with col_i2: IB_PORT = st.number_input("Port", value=creds.get("ib_port", 7497), step=1)
                        IB_CLIENT_ID = st.number_input("Client ID", value=creds.get("ib_client_id", 1), step=1)
                st.divider()
                with st.expander("📱 Notifications (Telegram/WhatsApp)"):
                    TG_TOKEN = st.text_input("Telegram Bot Token", value=creds.get("tg_token", ""))
                    TG_CHAT = st.text_input("Telegram Chat ID", value=creds.get("tg_chat", ""))
                    WA_PHONE = st.text_input("WhatsApp Phone", value=creds.get("wa_phone", ""))
                    WA_API = st.text_input("WhatsApp API Key", value=creds.get("wa_api", ""))
                SAVE_CREDS = st.checkbox("Remember Credentials Securely (Cloud DB)", value=True)
                keep_signed = st.checkbox("Keep me signed in (auto‑login using URL)")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("CONNECT MARKETS 🚀", type="primary", use_container_width=True):
                    if not USER_ID: st.error("Please enter your System Login ID (Email or Phone) to proceed.")
                    else:
                        temp_bot = SniperBot(
                            api_key=ANGEL_API if use_angel else "", client_id=CLIENT_ID if use_angel else "", 
                            pwd=PIN if use_angel else "", totp_secret=TOTP if use_angel else "", 
                            tg_token=TG_TOKEN, tg_chat=TG_CHAT, wa_phone=WA_PHONE, wa_api=WA_API, 
                            mt5_acc=MT5_ACC if use_mt5 else "", mt5_pass=MT5_PASS if use_mt5 else "", 
                            mt5_server=MT5_SERVER if use_mt5 else "", mt5_api_url=MT5_API_URL if use_mt5 else "",
                            zerodha_api=Z_API if use_zerodha else "", zerodha_secret=Z_SEC if use_zerodha else "", 
                            request_token=Z_REQ if use_zerodha else "",
                            coindcx_api=DCX_API if use_coindcx else "", coindcx_secret=DCX_SEC if use_coindcx else "",
                            delta_api=DELTA_API if use_delta else "", delta_secret=DELTA_SEC if use_delta else "",
                            ib_host=IB_HOST if use_ib else "", ib_port=IB_PORT if use_ib else 7497, ib_client_id=IB_CLIENT_ID if use_ib else 1,
                            is_mock=False
                        )
                        temp_bot.system_user_id = USER_ID
                        with st.spinner("Authenticating Secure Connections..."):
                            if temp_bot.login():
                                temp_bot.state["daily_pnl"] = temp_bot.load_daily_pnl()
                                if SAVE_CREDS: save_creds(USER_ID, ANGEL_API, CLIENT_ID, PIN, TOTP, TG_TOKEN, TG_CHAT, WA_PHONE, WA_API, MT5_ACC, MT5_PASS, MT5_SERVER, MT5_API_URL, Z_API, Z_SEC, DCX_API, DCX_SEC, DELTA_API, DELTA_SEC, IB_HOST, IB_PORT, IB_CLIENT_ID)
                                st.session_state.bot = temp_bot
                                st.session_state.audio_enabled = True
                                unlock_audio()
                                play_sound_ui("entry")
                                if keep_signed:
                                    st.query_params["user_id"] = USER_ID
                                    play_sound_ui("entry")
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
                    temp_bot.system_user_id = "paper_user"
                    temp_bot.state["daily_pnl"] = temp_bot.load_daily_pnl()
                    st.session_state.bot = temp_bot
                    st.session_state.audio_enabled = True
                    unlock_audio()
                    play_sound_ui("entry")
                    play_sound_ui("entry")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# --- MAIN TERMINAL ---
else:
    bot = st.session_state.bot
    
    head_c1, head_c2, head_c3 = st.columns([2, 1, 1])
    with head_c1: 
        broker_name = bot.settings.get("primary_broker", "Unknown")
        st.markdown(
            f"**👤 Session:** <span style='color:#0284c7; font-weight:800;'>{bot.client_name}</span> "
            f"<span class='broker-badge'>{broker_name}</span> | **IP:** `{bot.client_ip}`",
            unsafe_allow_html=True
        )
    with head_c2:
        connected = []
        if bot.api: connected.append("Angel")
        if bot.kite: connected.append("Zerodha")
        if bot.coindcx_api: connected.append("CoinDCX")
        if bot.delta_api: connected.append("Delta")
        if bot.is_mt5_connected: connected.append("MT5")
        if bot.is_ib_connected: connected.append("IB")
        if bot.is_mock: connected.append("Paper")
        st.markdown(f"**🔌 Connected:** {', '.join(connected) if connected else 'None'}")
    with head_c3:
        if st.button("🚪 LOGOUT", use_container_width=True):
            STOP_EVENT.set()
            bot.state["is_running"] = False
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()

    st.sidebar.markdown("---")

    with st.sidebar:
        st.header("⚙️ SYSTEM CONFIGURATION")
        
        if not st.session_state.audio_enabled:
            if st.button("🔊 Enable Audio", use_container_width=True):
                st.session_state.audio_enabled = True
                unlock_audio()
                st.success("Audio enabled!")
                st.rerun()
        else:
            st.success("🔊 Audio is ON")
        
        st.markdown("**1. Market Setup**")
        BROKER = st.selectbox("Primary Broker", ["Angel One", "Zerodha", "CoinDCX", "Delta Exchange", "MT5", "Interactive Brokers"], index=0)
        
        st.divider()
        st.markdown("**📈 High‑Profit Strategies**")
        martingale_mode = st.selectbox("Martingale Mode", ["Off", "Martingale", "Anti‑Martingale"], index=0)
        # Scalper always active – no toggle
        
        CUSTOM_STOCK = st.text_input("Add Custom Stock/Coin", value=st.session_state.custom_stock, placeholder="e.g. RELIANCE").upper().strip()
        st.session_state.custom_stock = CUSTOM_STOCK
        all_assets = list(LOT_SIZES.keys()) + ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        if CUSTOM_STOCK and CUSTOM_STOCK not in all_assets:
            all_assets.append(CUSTOM_STOCK)
        if BROKER in ["CoinDCX", "Delta Exchange"]:
            valid_assets = [a for a in all_assets if "USD" in a or "USDT" in a]
        elif BROKER in ["Angel One", "Zerodha"]:
            valid_assets = [a for a in all_assets if a in INDEX_TOKENS or a in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"] or a.isalpha()]
        elif BROKER == "Interactive Brokers":
            valid_assets = all_assets
        else:
            valid_assets = [a for a in all_assets if a in ["XAUUSD", "EURUSD", "BTCUSD", "ETHUSD", "SOLUSD"]]
        if CUSTOM_STOCK and CUSTOM_STOCK not in valid_assets:
            valid_assets.append(CUSTOM_STOCK)
        if not valid_assets:
            valid_assets = ["NIFTY"] if BROKER in ["Angel One", "Zerodha"] else ["BTCUSD"]
        st.session_state.asset_options = valid_assets
        if st.session_state.sb_index_input not in valid_assets:
            st.session_state.sb_index_input = valid_assets[0]
        
        INDEX = st.selectbox("Watchlist Asset", valid_assets, index=valid_assets.index(st.session_state.sb_index_input), key="sb_index_input")
        STRATEGY = st.selectbox("Trading Strategy", STRAT_LIST, index=STRAT_LIST.index(st.session_state.sb_strat_input), key="sb_strat_input")
        TIMEFRAME = st.selectbox("Candle Timeframe", ["1m", "3m", "5m", "15m"], index=2)
        
        CUSTOM_CODE = ""
        TV_PASSPHRASE = "SHREE123"
        if STRATEGY == "Keyword Rule Builder":
            st.divider()
            st.markdown("**🧠 Keyword Logic Builder**")
            selected_rules = st.multiselect(
                "Select Technical Conditions",
                ["EMA Crossover (9 & 21)", "Bollinger Bands Bounce", "RSI Breakout (>60/<40)", "MACD Crossover",
                 "Stochastic RSI", "VWAP Position", "AlphaTrend / Supertrend", "UT Bot", "Next Super Trend",
                 "Bollinger Bands", "Stochastic", "Moving Average", "Exponential", "ICT", "Bull FVG", "Bear FVG"],
                default=["EMA Crossover (9 & 21)"]
            )
            CUSTOM_CODE = ",".join(selected_rules)
            st.session_state.custom_code_input = CUSTOM_CODE
        elif STRATEGY == "TradingView Webhook":
            st.divider()
            st.markdown("**📡 TradingView Integration**")
            if HAS_FLASK:
                st.success(f"Webhook URL: `http://{bot.client_ip}:5000/tv_webhook`")
                TV_PASSPHRASE = st.text_input("Webhook Passphrase", value="SHREE123")
        
        if BROKER in ["CoinDCX", "Delta Exchange"]:
            st.divider()
            st.markdown("**🪙 Crypto Setup**")
            col_c1, col_c2 = st.columns(2)
            with col_c1: CRYPTO_MODE = st.selectbox("Market Type", ["Futures", "Spot", "Options"])
            with col_c2: LEVERAGE = st.number_input("Leverage (x)", 1, 100, 10, 1)
            SHOW_INR_CRYPTO = st.toggle("Convert to ₹ INR", True)
        else:
            CRYPTO_MODE = "Options"
            LEVERAGE = 1
            SHOW_INR_CRYPTO = False

        st.divider()
        st.markdown("**2. Risk Management**")
        lot_size = LOT_SIZES.get(INDEX, 1)
        st.caption(f"1 lot = {lot_size} units for {INDEX}")
        min_val = 1.0 if INDEX in LOT_SIZES and lot_size > 1 else 0.01
        step_val = 1.0 if INDEX in LOT_SIZES and lot_size > 1 else 0.01
        LOTS = st.number_input(
            "Lots",
            min_value=min_val,
            max_value=10000.0,
            value=1.0,
            step=step_val,
            key=f"lots_input_{INDEX}",
            help="Number of lots to trade"
        )
        actual_qty = LOTS * lot_size
        st.caption(f"Total quantity: {actual_qty:.2f} units")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            MAX_TRADES = st.number_input("Max Trades/Day", 1, 50, 5)
            MAX_CAPITAL = st.number_input("Max Cap/Trade (₹/$)", 10.0, 500000.0, 15000.0, step=100.0)
            SL_PTS = st.number_input("SL Points", 5.0, 500.0, 20.0)
            TSL_PTS = st.number_input("Trail SL", 5.0, 500.0, 15.0)
        with col_s2:
            TGT_PTS = st.number_input("Target Steps", 5.0, 1000.0, 15.0)
            CAPITAL_PROTECT = st.number_input("Max Loss", 500.0, 500000.0, 2000.0, step=500.0)
        
        MIN_SIGNAL_STRENGTH = st.slider("Min Signal Strength %", 0, 100, 30, 5,
                                        help="Minimum confidence score required to enter a trade")
        
        st.divider()
        st.markdown("**3. Advanced Triggers**")
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            MTF_CONFIRM = st.toggle("⏱️ Multi-TF Confirmation", False)
            THREE_FIVE_SEVEN = st.toggle("🔢 3-5-7 Rule (ATR based)", False, help="SL=1.5*ATR, TP=3/5/7*ATR")
        with col_adv2:
            if STRATEGY == "Machine Learning":
                ML_PROB_THRESHOLD = st.slider("ML Probability Threshold", 0.1, 0.6, 0.30, 0.05)
                SIGNAL_PERSISTENCE = st.slider("Signal Persistence (bars)", 1, 3, 1, 1)
            else:
                ML_PROB_THRESHOLD = 0.30
                SIGNAL_PERSISTENCE = 1

        st.divider()
        if st.button("🔄 Refresh Balance", use_container_width=True):
            st.rerun()
        
        if not bot.is_mock and st.button("🧪 Ping API Connection", use_container_width=True):
            st.toast("Testing exact API parameters...", icon="🧪")
            bot.log(f"🧪 User executed manual Ping API Connection for {BROKER}.")

        render_signature()

    bot.settings = {
        "primary_broker": BROKER, "strategy": STRATEGY, "index": INDEX, "timeframe": TIMEFRAME, 
        "lots": LOTS,
        "max_trades": MAX_TRADES, "max_capital": MAX_CAPITAL, "capital_protect": CAPITAL_PROTECT, 
        "sl_pts": SL_PTS, "tsl_pts": TSL_PTS, "tgt_pts": TGT_PTS, "paper_mode": bot.is_mock, 
        "mtf_confirm": MTF_CONFIRM, 
        "three_five_seven": THREE_FIVE_SEVEN,
        "crypto_mode": CRYPTO_MODE, "leverage": LEVERAGE, "show_inr_crypto": SHOW_INR_CRYPTO,
        "user_lots": LOT_SIZES.copy(),
        "custom_code": CUSTOM_CODE, "tv_passphrase": TV_PASSPHRASE,
        "martingale_mode": martingale_mode,
        "use_quantity_mode": True,
        "min_signal_strength": MIN_SIGNAL_STRENGTH,
        "ml_prob_threshold": ML_PROB_THRESHOLD,
        "signal_persistence": SIGNAL_PERSISTENCE,
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
                    if STRATEGY == "Keyword Rule Builder": 
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_keyword_strategy(df_preload, CUSTOM_CODE, INDEX)
                    elif STRATEGY == "Machine Learning":
                        if ml_predictor.should_retrain(df_preload):
                            ml_predictor.train(df_preload)
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_ml_strategy(
                            df_preload, INDEX, ML_PROB_THRESHOLD, SIGNAL_PERSISTENCE
                        )
                    elif "VIJAY & RFF" in STRATEGY: 
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_vijay_rff_strategy(df_preload, INDEX)
                    elif "Institutional FVG" in STRATEGY: 
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_institutional_fvg_strategy(df_preload, INDEX)
                    elif "Lux Algo" in STRATEGY:
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_lux_algo_ict_strategy(df_preload, INDEX)
                    elif STRATEGY == "TradingView Webhook": 
                        t, s, v, e, df_c, atr, fib, strength = "Awaiting TradingView Webhook...", "WAIT", df_preload['close'].iloc[-1], df_preload['close'].iloc[-1], df_preload, 0, {}, 50
                    else: 
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_vwap_ema_strategy(df_preload, INDEX)
                    
                    temp_df = bot.analyzer.calculate_indicators(df_preload, INDEX in ["NIFTY", "BANKNIFTY", "SENSEX", "INDIA VIX"])
                    for col in temp_df.columns:
                        if col not in df_preload.columns:
                            df_preload[col] = temp_df[col]
                    
                    bot.state.update({
                        "current_trend": t, "current_signal": s, 
                        "signal_strength": strength,
                        "vwap": v, "ema": e, "atr": atr, 
                        "fib_data": fib, "latest_data": df_preload
                    })

    if not is_mkt_open: 
        st.error(f"🛑 {mkt_status_msg} - Engine will standby until market opens.")
        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🕉️ DASHBOARD", "🔎 SCANNERS", "📜 LOGS", "🚀 CRYPTO/FX", "🎯 STOCK SCANNER"])

    # ========== TAB1 : DASHBOARD ==========
    with tab1:
        kannada_news = st.session_state.kannada_news
        english_news = st.session_state.english_news

        if not kannada_news:
            kannada_news = ["📰 Loading Kannada news... (first fetch in progress)"]
        if not english_news:
            english_news = ["📰 Loading English news... (first fetch in progress)"]

        kannada_news = fetch_kannada_news(INDEX)
        if kannada_news:
            ticker_text = " 🔹 ".join(kannada_news)
            st.markdown(f'<div class="news-ticker"><span>{ticker_text}</span></div>', unsafe_allow_html=True)

        english_news = fetch_english_news(INDEX)
        if english_news:
            ticker_text = " 🔹 ".join(english_news)
            st.markdown(f'<div class="news-ticker" style="background:#0f172a;"><span>{ticker_text}</span></div>', unsafe_allow_html=True)

        daily_pnl = bot.state.get("daily_pnl", 0.0)
        pnl_color = "#22c55e" if daily_pnl >= 0 else "#ef4444"
        pnl_sign = "+" if daily_pnl > 0 else ""

        exch, _ = bot.get_token_info(INDEX)
        if exch == "MT5": term_type = "🌍 MT5 Forex Terminal (Web Bridge)"
        elif exch == "COINDCX": term_type = f"🕉️ CoinDCX {CRYPTO_MODE}"
        elif exch == "DELTA": term_type = f"🔺 Delta Exchange {CRYPTO_MODE}"
        else: term_type = f"🇮🇳 {BROKER} NSE/NFO"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0284c7, #0369a1); padding: 18px; border-radius: 4px; border: 1px solid #e2e8f0; color: white; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h2 style="margin: 0; color: #ffffff; font-weight: 800; letter-spacing: 1px;">🕉️ {INDEX}</h2>
                <p style="margin: 5px 0 0 0; font-size: 0.95rem; color: #e0f2fe; font-weight: 700;">{term_type}</p>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed rgba(255,255,255,0.3);">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <span style="font-size: 0.85rem; color: #f8fafc;">Live Balance:</span><br>
                            <span style="font-size: 1.2rem; font-weight: bold; color: #ffffff;">{bot.get_balance()}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 0.85rem; color: #f8fafc;">Today's P&L:</span><br>
                            <span style="font-size: 2rem; font-weight: bold; color: {pnl_color};">{pnl_sign}₹{abs(round(daily_pnl, 2))}</span>
                        </div>
                    </div>
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
                st.rerun()

        st.markdown("### 🚨 FOMO Scanner (Volume Spike Alerts)")
        fomo_signals = fomo_scanner.scan()
        if fomo_signals:
            df_fomo = pd.DataFrame(fomo_signals)
            st.dataframe(df_fomo, use_container_width=True, hide_index=True)
            if bot.state.get("fomo_signals"):
                st.markdown("**FOMO Signals from current session:**")
                fomo_list = list(bot.state["fomo_signals"])[-5:]
                if fomo_list:
                    st.dataframe(pd.DataFrame(fomo_list), use_container_width=True, hide_index=True)
        else:
            st.info("No volume spike alerts at the moment.")

        ltp_val = round(bot.state['spot'], 4)
        trend_val = bot.state['current_trend']
        signal_strength = bot.state.get('signal_strength', 0)
        
        if bot.state.get('latest_data') is not None and len(bot.state['latest_data']) >= 10:
            mh, ml, f_low, f_high = bot.analyzer.calculate_fib_zones(bot.state['latest_data'])
            gz_display = f"{round(f_low, 2)} - {round(f_high, 2)}" if f_low > 0 else "Calculating..."
        else:
            gz_display = "Calculating..."
        
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
        
        day_high = "N/A"
        day_low = "N/A"
        try:
            yf_ticker = YF_TICKERS.get(INDEX, INDEX)
            df_day = yf.Ticker(yf_ticker).history(period="1d", interval="1m")
            if not df_day.empty:
                day_high = df_day['High'].max()
                day_low = df_day['Low'].min()
        except:
            pass
        
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
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 800; letter-spacing: 1px;">Signal Strength</div>
                    <div style="width: 100%; height: 10px; background: #e2e8f0; border-radius: 5px; margin-top: 5px;">
                        <div style="width: {signal_strength}%; height: 10px; background: #0284c7; border-radius: 5px;"></div>
                    </div>
                    <div style="text-align: right; font-size: 0.8rem; color: #64748b; margin-top: 2px;">{signal_strength}%</div>
                </div>
                <div style="background: #ffffff; padding: 15px; border-radius: 4px; border: 1px solid #e2e8f0; text-align: center; grid-column: span 2; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 800; letter-spacing: 1px;">Live OHLCV</div>
                    <div style="font-size: 1.1rem; color: #0f111a; font-weight: 900; margin-top: 4px;">
                        O: {round(o_val, 4)} &nbsp;|&nbsp; H: {round(h_val, 4)} &nbsp;|&nbsp; L: {round(l_val, 4)} &nbsp;|&nbsp; C: {round(c_val, 4)}<br>
                        <span style="font-size: 0.95rem; color: #0284c7;">V: {v_display}</span>
                    </div>
                </div>
                <div style="background: #ffffff; padding: 15px; border-radius: 4px; border: 1px solid #e2e8f0; text-align: center; grid-column: span 2; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 800; letter-spacing: 1px;">Day's Range</div>
                    <div style="font-size: 1.1rem; color: #0f111a; font-weight: 900; margin-top: 4px;">
                        H: {round(day_high, 4) if day_high != "N/A" else "N/A"} &nbsp;|&nbsp; L: {round(day_low, 4) if day_low != "N/A" else "N/A"}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Live Position Tracker")
        daily_pnl = bot.state.get("daily_pnl", 0.0)
        risk_level = "low" if abs(daily_pnl) < CAPITAL_PROTECT * 0.3 else ("medium" if abs(daily_pnl) < CAPITAL_PROTECT * 0.7 else "high")
        
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span><b>Session Net Yield:</b> {'<span style="color:#22c55e">🟢' if daily_pnl >= 0 else '<span style="color:#ef4444">🔴'} {round(daily_pnl, 2)}</span></span>
                <span><b>Risk Level:</b> <span class="risk-{risk_level}">⚡ {risk_level.upper()}</span></span>
            </div>
        """, unsafe_allow_html=True)
        
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
                            <span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 700;">Qty</span><br>
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
            ema_col = None
            if 'ema_fast' in chart_df.columns:
                ema_col = 'ema_fast'
            elif 'ema_short' in chart_df.columns:
                ema_col = 'ema_short'
            elif 'ema9' in chart_df.columns:
                ema_col = 'ema9'
            if ema_col:
                ema_data = chart_df[['time', ema_col]].dropna().rename(columns={ema_col: 'value'}).to_dict('records')
                if ema_data: chart_series.append({"type": 'Line', "data": ema_data, "options": { "color": '#0ea5e9', "lineWidth": 2, "title": 'EMA' }})

            renderLightweightCharts([{"chart": chartOptions, "series": chart_series}], key="static_tv_chart")

    # ========== TAB2 : SCANNERS ==========
    with tab2:
        tab_a, tab_b, tab_c, tab_us = st.tabs(["📊 52W High/Low", "📡 Multi-Stock + Pin Bar", "📈 Breakout", "🇺🇸 US Stock Scanner"])
        with tab_a:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                if st.button("🔍 Scan 52W High/Low", use_container_width=True):
                    with st.spinner("Scanning..."):
                        watch_list = [
                            "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS",
                            "BHARTIARTL.NS", "ITC.NS", "LT.NS", "WIPRO.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
                            "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "HCLTECH.NS", "ASIANPAINT.NS",
                            "TITAN.NS", "ULTRACEMCO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS",
                            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"
                        ]
                        results = []
                        for ticker in watch_list:
                            try:
                                tk = yf.Ticker(ticker)
                                hist = tk.history(period="1y")
                                if hist.empty: continue
                                ltp = hist['Close'].iloc[-1]
                                high_52 = hist['High'].max()
                                low_52 = hist['Low'].min()
                                atr = (hist['High'] - hist['Low']).rolling(14).mean().iloc[-1]
                                midpoint = (high_52 + low_52) / 2
                                if ltp > midpoint:
                                    direction = "LONG"
                                    entry = ltp
                                    sl = ltp - atr
                                    tp = ltp + atr * 2
                                    signal = "BUY 🟢"
                                else:
                                    direction = "SHORT"
                                    entry = ltp
                                    sl = ltp + atr
                                    tp = ltp - atr * 2
                                    signal = "SELL 🔴"
                                if ltp > high_52 * 0.95:
                                    signal = "STRONG BUY 🚀"
                                elif ltp < low_52 * 1.05:
                                    signal = "STRONG SELL 🔻"
                                results.append({
                                    "Symbol": ticker,
                                    "LTP": round(ltp, 2),
                                    "52W High": round(high_52, 2),
                                    "52W Low": round(low_52, 2),
                                    "Signal": signal,
                                    "Entry": round(entry, 2),
                                    "SL": round(sl, 2),
                                    "TP": round(tp, 2)
                                })
                            except:
                                continue
                        if results:
                            df_res = pd.DataFrame(results)
                            def highlight_signal(val):
                                if "STRONG BUY" in val:
                                    return 'background-color: #22c55e; color: white'
                                elif "STRONG SELL" in val:
                                    return 'background-color: #ef4444; color: white'
                                return ''
                            styled = df_res.style.map(highlight_signal, subset=['Signal'])
                            st.dataframe(styled, use_container_width=True, hide_index=True)
                        else:
                            st.info("No data.")
                st.markdown('</div>', unsafe_allow_html=True)
        with tab_b:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                if st.button("🔄 Scan Multi-Stock", use_container_width=True):
                    with st.spinner("Scanning..."):
                        symbols = ["NIFTY", "SENSEX", "GOLD", "BTC-USD", "ETH-USD", "SOL-USD"]
                        results = []
                        for sym in symbols:
                            yf_sym = sym
                            if sym == "NIFTY": yf_sym = "^NSEI"
                            elif sym == "SENSEX": yf_sym = "^BSESN"
                            elif sym == "GOLD": yf_sym = "GC=F"
                            df = yf.Ticker(yf_sym).history(period="5d", interval="1d")
                            if not df.empty:
                                inside = "Yes" if (df['High'].iloc[-1] <= df['High'].iloc[-2] and df['Low'].iloc[-1] >= df['Low'].iloc[-2]) else "No"
                                last = df.iloc[-1]
                                body = abs(last['Close'] - last['Open'])
                                upper = last['High'] - max(last['Close'], last['Open'])
                                lower = min(last['Close'], last['Open']) - last['Low']
                                pin = "None"
                                if lower > body*2 and last['Close'] > last['Open']:
                                    pin = "Bullish Pin"
                                elif upper > body*2 and last['Close'] < last['Open']:
                                    pin = "Bearish Pin"
                                results.append({
                                    "Symbol": sym,
                                    "LTP": round(last['Close'], 2),
                                    "Inside Bar": inside,
                                    "Pin Bar": pin
                                })
                        if results:
                            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                        else:
                            st.info("No data.")
                st.markdown('</div>', unsafe_allow_html=True)
        with tab_c:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                if st.button("🚀 Scan Breakouts", use_container_width=True):
                    with st.spinner("Scanning..."):
                        watch_list = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]
                        results = []
                        for ticker in watch_list:
                            try:
                                tk = yf.Ticker(ticker)
                                hist = tk.history(period="5d", interval="5m")
                                if len(hist) < 50: continue
                                hist['sma20'] = hist['Close'].rolling(20).mean()
                                hist['volume_sma'] = hist['Volume'].rolling(20).mean()
                                last = hist.iloc[-1]
                                recent_high = hist['High'].tail(20).max()
                                if last['Close'] > recent_high and last['Volume'] > hist['volume_sma'].iloc[-1] * 1.5:
                                    signal = "Breakout UP 🚀"
                                elif last['Close'] < hist['Low'].tail(20).min() and last['Volume'] > hist['volume_sma'].iloc[-1] * 1.5:
                                    signal = "Breakout DOWN 🔻"
                                else:
                                    signal = "No"
                                results.append({
                                    "Stock": ticker.replace(".NS", ""),
                                    "LTP": round(last['Close'], 2),
                                    "Breakout": signal,
                                    "Volume": int(last['Volume'])
                                })
                            except: pass
                        if results:
                            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                        else:
                            st.info("No breakouts.")
                st.markdown('</div>', unsafe_allow_html=True)
        with tab_us:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("🇺🇸 US Stock Scanner")
                us_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT", "JNJ", "PG", "UNH", "HD", "DIS"]
                if st.button("Scan US Stocks", use_container_width=True):
                    with st.spinner("Scanning US stocks..."):
                        results = []
                        for ticker in us_list:
                            try:
                                tk = yf.Ticker(ticker)
                                hist = tk.history(period="1mo", interval="1d")
                                if len(hist) < 20:
                                    continue
                                hist['ema9'] = hist['Close'].ewm(span=9).mean()
                                hist['ema21'] = hist['Close'].ewm(span=21).mean()
                                hist['volume_ma'] = hist['Volume'].rolling(20).mean()
                                last = hist.iloc[-1]
                                if (last['Close'] > last['ema9'] and last['ema9'] > last['ema21'] and last['Volume'] > last['volume_ma'] * 1.1):
                                    signal = "BUY 🟢"
                                    entry = last['Close']
                                    sl = last['Close'] * 0.98
                                    tp = last['Close'] * 1.04
                                elif (last['Close'] < last['ema9'] and last['ema9'] < last['ema21'] and last['Volume'] > last['volume_ma'] * 1.1):
                                    signal = "SELL 🔴"
                                    entry = last['Close']
                                    sl = last['Close'] * 1.02
                                    tp = last['Close'] * 0.96
                                else:
                                    signal = "Neutral"
                                    entry = last['Close']
                                    sl = last['Close'] * 0.98
                                    tp = last['Close'] * 1.02
                                results.append({
                                    "Symbol": ticker,
                                    "LTP": round(last['Close'], 2),
                                    "Signal": signal,
                                    "Entry": round(entry, 2),
                                    "SL": round(sl, 2),
                                    "TP": round(tp, 2)
                                })
                            except:
                                continue
                        if results:
                            df_us = pd.DataFrame(results)
                            def highlight_signal(val):
                                if "BUY" in val:
                                    return 'background-color: #22c55e; color: white'
                                elif "SELL" in val:
                                    return 'background-color: #ef4444; color: white'
                                return ''
                            styled_us = df_us.style.map(highlight_signal, subset=['Signal'])
                            st.dataframe(styled_us, use_container_width=True, hide_index=True)
                        else:
                            st.info("No signals found.")
                st.markdown('</div>', unsafe_allow_html=True)

    # ========== TAB3 : LOGS ==========
    with tab3:
        tab_log, tab_ledger = st.tabs(["📋 Console", "📊 Ledger"])
        with tab_log:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                col_clr, _ = st.columns([1,5])
                with col_clr:
                    if st.button("🗑️ Clear", use_container_width=True):
                        bot.state["logs"].clear()
                        if "paper_history" in bot.state:
                            bot.state["paper_history"] = []
                        bot.state["daily_pnl"] = 0.0
                        bot.state["trades_today"] = 0
                        if not bot.is_mock and HAS_DB:
                            try:
                                uid = getattr(bot,"system_user_id",bot.api_key)
                                supabase.table("trade_logs").delete().eq("user_id", uid).execute()
                            except: pass
                        st.rerun()
                for l in bot.state["logs"]:
                    st.markdown(f"`{l}`")
                st.markdown('</div>', unsafe_allow_html=True)
        with tab_ledger:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                report_period = st.selectbox("Select report period", ["Daily", "Weekly", "All Time"], index=0)
                if bot.is_mock:
                    if bot.state.get("paper_history"):
                        df = pd.DataFrame(bot.state["paper_history"])
                        st.dataframe(df.iloc[::-1], use_container_width=True)
                        if report_period == "Daily":
                            today = get_ist().strftime('%Y-%m-%d')
                            df_report = df[df['Date'] == today]
                        elif report_period == "Weekly":
                            week_ago = (get_ist() - dt.timedelta(days=7)).strftime('%Y-%m-%d')
                            df_report = df[df['Date'] >= week_ago]
                        else:
                            df_report = df
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as w:
                            df_report.to_excel(w, index=False)
                        st.download_button("📥 Export", data=output.getvalue(), file_name="mock_ledger.xlsx")
                    else:
                        st.info("No paper trades yet.")
                else:
                    uid = getattr(bot,"system_user_id",bot.api_key)
                    if HAS_DB:
                        try:
                            res = supabase.table("trade_logs").select("*").eq("user_id", uid).execute()
                            if res.data:
                                df = pd.DataFrame(res.data).drop(columns=["id","user_id"], errors="ignore")
                                st.dataframe(df.iloc[::-1], use_container_width=True)
                                if report_period == "Daily":
                                    today = get_ist().strftime('%Y-%m-%d')
                                    df_report = df[df['trade_date'] == today]
                                elif report_period == "Weekly":
                                    week_ago = (get_ist() - dt.timedelta(days=7)).strftime('%Y-%m-%d')
                                    df_report = df[df['trade_date'] >= week_ago]
                                else:
                                    df_report = df
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as w:
                                    df_report.to_excel(w, index=False)
                                st.download_button("📥 Export", data=output.getvalue(), file_name="live_ledger.xlsx")
                            else:
                                st.info("No live trades.")
                        except Exception as e:
                            st.error(f"DB error: {e}")
                    else:
                        st.error("DB disconnected.")
                st.markdown('</div>', unsafe_allow_html=True)

    # ========== TAB4 : CRYPTO/FX ==========
    with tab4:
        tab_c1, tab_c2, tab_c3 = st.tabs(["🪙 CoinDCX Scanner", "⚡ 1-Min Scalper", "🚀 Breakout Scanner"])
        
        with tab_c1:
            col_dx, col_bias = st.columns(2)
            with col_dx:
                with st.container():
                    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                    st.subheader("🕉️ CoinDCX Momentum")
                    if st.button("Scan CoinDCX 🔥", use_container_width=True):
                        with st.spinner("Fetching..."):
                            try:
                                resp = requests.get("https://api.coindcx.com/exchange/ticker", timeout=10)
                                if resp.status_code == 200:
                                    data = resp.json()
                                    coins = []
                                    for coin in data:
                                        mkt = coin.get('market','')
                                        if mkt.endswith('USDT') or mkt.endswith('INR'):
                                            try:
                                                chg = float(coin.get('change_24_hour',0))
                                                price = float(coin.get('last_price',0))
                                                if price>0:
                                                    coins.append({"Pair":mkt, "LTP":price, "24h %":chg})
                                            except: pass
                                    if coins:
                                        df = pd.DataFrame(coins).sort_values("24h %", ascending=False).head(15)
                                        df['24h %'] = df['24h %'].apply(lambda x: f"+{x}%" if x>0 else f"{x}%")
                                        st.dataframe(df, use_container_width=True, hide_index=True)
                                else:
                                    st.error(f"API error {resp.status_code}")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)
            with col_bias:
                with st.container():
                    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                    st.subheader("📈 Directional Bias")
                    if st.button("Analyze BTC & XAUUSD 🔍", use_container_width=True):
                        with st.spinner("Analyzing..."):
                            assets = {"Bitcoin (BTC-USD)": "BTC-USD", "Gold (XAUUSD)": "GC=F", "Silver (SI=F)": "SI=F"}
                            results = []
                            for name, ticker in assets.items():
                                try:
                                    tk = yf.Ticker(ticker)
                                    df = tk.history(period="5d", interval="15m")
                                    if not df.empty:
                                        df['ema9'] = df['Close'].ewm(span=9).mean()
                                        df['ema21'] = df['Close'].ewm(span=21).mean()
                                        c = df['Close'].iloc[-1]
                                        e9 = df['ema9'].iloc[-1]
                                        e21 = df['ema21'].iloc[-1]
                                        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
                                        if c > e9 > e21:
                                            bias = "UP 🟢"
                                            entry_long = c
                                            sl_long = c - atr
                                            tp_long = c + atr * 2
                                            entry_short = "N/A"
                                            sl_short = "N/A"
                                            tp_short = "N/A"
                                        elif c < e9 < e21:
                                            bias = "DOWN 🔴"
                                            entry_long = "N/A"
                                            sl_long = "N/A"
                                            tp_long = "N/A"
                                            entry_short = c
                                            sl_short = c + atr
                                            tp_short = c - atr * 2
                                        else:
                                            bias = "RANGING 🟡"
                                            entry_long = sl_long = tp_long = entry_short = sl_short = tp_short = "N/A"
                                        results.append({
                                            "Asset": name,
                                            "Price": round(c, 2),
                                            "Bias": bias,
                                            "Entry L": round(entry_long, 2) if entry_long != "N/A" else "N/A",
                                            "SL L": round(sl_long, 2) if sl_long != "N/A" else "N/A",
                                            "TP L": round(tp_long, 2) if tp_long != "N/A" else "N/A",
                                            "Entry S": round(entry_short, 2) if entry_short != "N/A" else "N/A",
                                            "SL S": round(sl_short, 2) if sl_short != "N/A" else "N/A",
                                            "TP S": round(tp_short, 2) if tp_short != "N/A" else "N/A"
                                        })
                                except: pass
                            if results:
                                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                            else:
                                st.info("No data")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab_c2:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                scalper_asset = st.selectbox("Select Asset", ["XAUUSD (Gold)", "BTCUSD", "ETHUSD", "SOLUSD"], index=0)
                asset_map = {"XAUUSD (Gold)": "XAUUSD", "BTCUSD": "BTCUSD", "ETHUSD": "ETHUSD", "SOLUSD": "SOLUSD"}
                symbol = asset_map[scalper_asset]
                gold_crypto_scalper(symbol)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab_c3:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                add_breakout_scalper()
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                compounding_calculator()
                st.markdown('</div>', unsafe_allow_html=True)

    # ========== TAB5 : STOCK SCANNER (Hero/Zero & Pin Bar) ==========
    with tab5:
        st.subheader("🎯 Hero/Zero & Pin Bar Scanner (Stocks only)")
        
        col_hz1, col_hz2 = st.columns(2)
        with col_hz1:
            scan_button = st.button("🔍 Scan Hero/Zero Now", use_container_width=True, type="primary")
        with col_hz2:
            st.session_state.hz_demo_mode = st.checkbox("Use Demo Data", value=False, help="Generate sample signals even in real mode")
        
        if scan_button:
            with st.spinner("Scanning for Hero/Zero patterns..."):
                st.markdown("### Nifty 50 Stocks")
                if st.session_state.hz_demo_mode:
                    nifty_results = bot.scan_hero_zero_indian_stocks()  # will return empty in demo? We'll handle demo inside method
                    # For demo, we'll generate mock data if empty
                    if nifty_results.empty:
                        nifty_results = pd.DataFrame([
                            {"Stock": "RELIANCE", "Direction": "HERO (BUY)", "Price": 2450.50, "Volume Spike": "2.3x", "ATR": 12.5, "Entry": 2450.50, "SL": 2438.00, "Target 1": 2475.50, "Target 2": 2500.50, "Risk/Reward": "1:2"},
                            {"Stock": "TCS", "Direction": "ZERO (SELL)", "Price": 3450.20, "Volume Spike": "1.8x", "ATR": 15.0, "Entry": 3450.20, "SL": 3465.20, "Target 1": 3420.20, "Target 2": 3390.20, "Risk/Reward": "1:2"},
                        ])
                else:
                    nifty_results = bot.scan_hero_zero_indian_stocks()
                if not nifty_results.empty:
                    st.success(f"Found {len(nifty_results)} Hero/Zero opportunities in Nifty 50!")
                    def color_direction(val):
                        if "HERO" in val:
                            return 'background-color: #22c55e; color: white'
                        elif "ZERO" in val:
                            return 'background-color: #ef4444; color: white'
                        return ''
                    styled_results = nifty_results.style.map(color_direction, subset=['Direction'])
                    st.dataframe(styled_results, use_container_width=True, hide_index=True)
                else:
                    st.info("No Hero/Zero opportunities in Nifty 50 at this moment")
                
                st.markdown("### Penny Stocks")
                if st.session_state.hz_demo_mode:
                    penny_results = pd.DataFrame([
                        {"Stock": "IDEA", "Direction": "HERO (BUY)", "Price": 8.75, "Volume Spike": "3.1x", "ATR": 0.35, "Entry": 8.75, "SL": 8.40, "Target 1": 9.80, "Target 2": 10.85, "Risk/Reward": "1:2"},
                        {"Stock": "YESBANK", "Direction": "ZERO (SELL)", "Price": 18.20, "Volume Spike": "2.2x", "ATR": 0.80, "Entry": 18.20, "SL": 19.00, "Target 1": 16.60, "Target 2": 15.00, "Risk/Reward": "1:2"},
                    ])
                else:
                    penny_results = bot.scan_penny_stocks()
                if not penny_results.empty:
                    st.success(f"Found {len(penny_results)} Hero/Zero opportunities in Penny Stocks!")
                    penny_styled = penny_results.style.map(color_direction, subset=['Direction'])
                    st.dataframe(penny_styled, use_container_width=True, hide_index=True)
                else:
                    st.info("No Hero/Zero opportunities in Penny Stocks at this moment")
                
                st.markdown("### Pin Bar Reversals (Indices & Gold) - 1-min signals")
                pin_results = bot.scan_pin_bars()
                if pin_results:
                    pin_df = pd.DataFrame(pin_results)
                    st.dataframe(pin_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No pin bar reversals detected at this moment")
                
                st.markdown("### 📝 Entry Instructions")
                st.info("""
                **For HERO (BUY):**
                - **Entry:** Current price
                - **Stop Loss:** 1.5x ATR below entry
                - **Target 1:** 3x ATR above entry (Book 50%)
                - **Target 2:** 5x ATR above entry (Book remaining)
                - **Risk/Reward:** 1:2
                
                **For ZERO (SELL):**
                - **Entry:** Current price
                - **Stop Loss:** 1.5x ATR above entry
                - **Target 1:** 3x ATR below entry (Book 50%)
                - **Target 2:** 5x ATR below entry (Book remaining)
                - **Risk/Reward:** 1:2
                """)

    # ========== Bottom Dock ==========
    def cycle_asset():
        assets = st.session_state.get('asset_options', list(LOT_SIZES.keys()))
        if st.session_state.sb_index_input in assets:
            st.session_state.sb_index_input = assets[(assets.index(st.session_state.sb_index_input) + 1) % len(assets)]
        else:
            st.session_state.sb_index_input = assets[0]

    def cycle_strat():
        st.session_state.sb_strat_input = STRAT_LIST[(STRAT_LIST.index(st.session_state.sb_strat_input) + 1) % len(STRAT_LIST)]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("◀️", key="dock_back", on_click=cycle_asset, help="Cycle Asset")
    with col2:
        st.button("🏠", key="dock_home", on_click=lambda: st.rerun(), help="Refresh")
    with col3:
        st.button("🔲", key="dock_recent", on_click=cycle_strat, help="Cycle Strategy")

    if bot.state.get("is_running"):
        time.sleep(2)
        st.rerun()
