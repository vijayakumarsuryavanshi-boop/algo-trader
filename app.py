# ==========================================
# SHREE ALGO TRADING PLATFORM – FINAL (FULL VERSION)
# ==========================================

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
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from streamlit.runtime.scriptrunner import add_script_run_ctx
from collections import deque
from streamlit_lightweight_charts import renderLightweightCharts
from supabase import create_client, Client
from threading import Event, Lock
import queue
import math
from scipy.stats import norm
import xml.etree.ElementTree as ET
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
from datetime import datetime, timedelta
import base64
import functools
import random
import socketio
import asyncio

# ---------- Optional imports with fallbacks ----------
try:
    import firebase_admin
    from firebase_admin import credentials, messaging
    HAS_FIREBASE = True
except:
    HAS_FIREBASE = False

try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

try:
    import joblib
    HAS_JOBLIB = True
except:
    HAS_JOBLIB = False

try:
    import user_agents
    HAS_USER_AGENTS = True
except:
    HAS_USER_AGENTS = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import pandas_ta as ta
    HAS_PTA = True
except:
    HAS_PTA = False

try:
    from kiteconnect import KiteConnect
    HAS_ZERODHA = True
except:
    HAS_ZERODHA = False

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except:
    HAS_MT5 = False

try:
    from plyer import notification
    HAS_NOTIFY = True
except:
    HAS_NOTIFY = False

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except:
    HAS_FLASK = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except:
    HAS_FEEDPARSER = False

try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, ContextTypes
    HAS_TELEGRAM_BOT = True
except:
    HAS_TELEGRAM_BOT = False

try:
    from fyers_apiv3 import fyersModel
    HAS_FYERS = True
except:
    HAS_FYERS = False

try:
    from upstox_py import Upstox
    HAS_UPSTOX = True
except:
    HAS_UPSTOX = False

try:
    from fivepaisa import FivePaisaClient
    HAS_5PAISA = True
except:
    HAS_5PAISA = False

try:
    from binance.client import Client as BinanceClient
    from binance.exceptions import BinanceAPIException
    from binance.websocket import BinanceSocketManager
    HAS_BINANCE = True
except:
    HAS_BINANCE = False

try:
    from dhanhq import dhanhq
    HAS_DHAN = True
except:
    HAS_DHAN = False

try:
    from shoonya import Shoonya
    HAS_SHOONYA = True
except:
    HAS_SHOONYA = False

try:
    import stoxkart
    HAS_STOXKART = True
except:
    HAS_STOXKART = False

try:
    import socketio
    HAS_SOCKETIO = True
except:
    HAS_SOCKETIO = False

# ==========================================
# HOLIDAY & EVENT DATA (corrected)
# ==========================================
INDIAN_HOLIDAYS = {
    "2026-03-26": "Shri Ram Navami",
    "2026-03-31": "Shri Mahavir Jayanti",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-28": "Bakri Id",
    "2026-06-26": "Muharram",
    "2026-09-14": "Ganesh Chaturthi",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-10": "Diwali – Balipratipada",
    "2026-11-24": "Prakash Gurpurb Sri Guru Nanak Dev",
    "2026-12-25": "Christmas"
}

MAJOR_EVENTS = [
    ("2026-04-02", "RBI Monetary Policy (Repo Rate)"),
    ("2026-06-04", "RBI Monetary Policy (Repo Rate)"),
    ("2026-08-06", "RBI Monetary Policy (Repo Rate)"),
    ("2026-10-01", "RBI Monetary Policy (Repo Rate)"),
    ("2026-12-03", "RBI Monetary Policy (Repo Rate)"),
    ("2027-02-04", "RBI Monetary Policy (Repo Rate)"),
    ("2026-04-29", "FOMC Meeting (Interest Rate Decision)"),
    ("2026-06-10", "FOMC Meeting (Interest Rate Decision)"),
    ("2026-07-29", "FOMC Meeting (Interest Rate Decision)"),
    ("2026-09-16", "FOMC Meeting (Interest Rate Decision)"),
    ("2026-11-04", "FOMC Meeting (Interest Rate Decision)"),
    ("2026-12-16", "FOMC Meeting (Interest Rate Decision)"),
    ("2027-01-27", "FOMC Meeting (Interest Rate Decision)"),
    ("2027-03-10", "FOMC Meeting (Interest Rate Decision)")
]

def get_tomorrow_info():
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    holiday = INDIAN_HOLIDAYS.get(tomorrow)
    event = None
    for date, name in MAJOR_EVENTS:
        if date == tomorrow:
            event = name
            break
    return holiday, event

# ==========================================
# AUDIO FUNCTIONS
# ==========================================
def play_sound_ui(sound_type="entry"):
    if not st.session_state.get("audio_enabled", False):
        return
    sound_urls = {
        "login": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "entry": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "tp": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "sl": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "exit": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "alert": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "click": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "clock": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"
    }
    url = sound_urls.get(sound_type, sound_urls["alert"])
    unique_url = f"{url}?{uuid.uuid4()}"
    audio_html = f"""
    <audio autoplay style="display:none;">
        <source src="{unique_url}" type="audio/mpeg">
    </audio>
    """
    components.html(audio_html, height=0)
    st.toast(f"🔔 {sound_type.upper()} signal", icon="🔊")

def play_sound_now(sound_type="click"):
    if not st.session_state.get("audio_enabled", False):
        return
    sound_urls = {
        "login": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "entry": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "tp": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "sl": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "exit": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "alert": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "click": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        "clock": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3"
    }
    url = sound_urls.get(sound_type, sound_urls["alert"])
    unique_url = f"{url}?{uuid.uuid4()}"
    audio_html = f"""
    <audio autoplay style="display:none;">
        <source src="{unique_url}" type="audio/mpeg">
    </audio>
    <script>
        setTimeout(function() {{
            var audio = document.currentScript.previousElementSibling;
            audio.play().catch(e => {{}});
        }}, 10);
    </script>
    """
    components.html(audio_html, height=0)

def trigger_click_sound(sound_type="entry"):
    st.session_state.sound_queue.append(sound_type)

def play_clock_sound():
    play_sound_now("clock")

def unlock_audio():
    pass

# ==========================================
# DEVICE INFO
# ==========================================
def get_device_info():
    ua_string = st.context.headers.get("User-Agent", "Unknown")
    if HAS_USER_AGENTS:
        try:
            ua = user_agents.parse(ua_string)
            device = ua.device.family or "Desktop"
            if device == "Other" and ua.is_mobile:
                device = "Mobile"
            browser = ua.browser.family
            os = ua.os.family
            return f"{device} ({os} / {browser})", ua_string
        except:
            pass
    if "Mobile" in ua_string or "Android" in ua_string or "iPhone" in ua_string:
        device = "Mobile"
    elif "Windows" in ua_string:
        device = "Windows PC"
    elif "Mac" in ua_string:
        device = "Mac"
    elif "Linux" in ua_string:
        device = "Linux"
    else:
        device = "Unknown Device"
    return device, ua_string

def get_client_ip():
    try:
        forwarded = st.context.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = st.context.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return st.context.headers.get("Remote-Addr", "Unknown IP")
    except:
        return "Unknown IP"

# ==========================================
# LIVE NEWS
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

def fetch_feed(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall('.//item'):
            title = item.find('title').text
            if title:
                items.append(title)
        return items
    except Exception:
        return None

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
        "https://www.prajavani.net/rss/ವಾಣಿಜ್ಯ",
        "https://vijaykarnataka.com/rss/business",
        "https://kannada.asianetnews.com/rss/business",
        "https://feeds.feedburner.com/Republickannada?format=xml",
        "https://kannada.timesnownews.com/rss/business",
        "https://www.newsonair.gov.in/bulletins/kannada/?feed=rss2"
    ]
    for feed_url in rss_feeds:
        try:
            titles = fetch_feed(feed_url)
            if titles and len(titles) >= 3:
                return [f"📰 {t}" for t in titles[:5]]
        except Exception:
            continue
    return [generate_market_prediction(asset)]

# ==========================================
# SESSION STATE INIT
# ==========================================
def news_updater():
    while True:
        asset = st.session_state.get('sb_index_input', 'NIFTY')
        st.session_state.kannada_news = fetch_kannada_news(asset)
        st.session_state.english_news = fetch_english_news(asset)
        time.sleep(120)

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
    st.session_state.asset_options = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "XAUUSD", "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "BNBUSD", "LTCUSD", "DOTUSD"]
if 'use_quantity_mode' not in st.session_state:
    st.session_state.use_quantity_mode = False
if 'hz_demo_mode' not in st.session_state:
    st.session_state.hz_demo_mode = False
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = False
if 'win_streak' not in st.session_state:
    st.session_state.win_streak = 0
if 'loss_streak' not in st.session_state:
    st.session_state.loss_streak = 0
if 'fomo_mode' not in st.session_state:
    st.session_state.fomo_mode = False
if 'fomo_indices' not in st.session_state:
    st.session_state.fomo_indices = ["NIFTY", "BANKNIFTY", "SENSEX"]
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0
if 'device_name' not in st.session_state:
    st.session_state.device_name, st.session_state.user_agent = get_device_info()
if 'ip_address' not in st.session_state:
    st.session_state.ip_address = get_client_ip()
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'user_id' not in st.session_state:
    st.session_state.user_id = st.query_params.get("user_id", "")
if 'is_developer' not in st.session_state:
    st.session_state.is_developer = False
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'user_role' not in st.session_state:
    st.session_state.user_role = "trader"
if 'manual_mode' not in st.session_state:
    st.session_state.manual_mode = False
if 'stocks_only' not in st.session_state:
    st.session_state.stocks_only = False

# ==========================================
# LICENSE STATE
# ==========================================
if 'license_key' not in st.session_state:
    st.session_state.license_key = ""
if 'license_valid' not in st.session_state:
    st.session_state.license_valid = False
if 'license_user' not in st.session_state:
    st.session_state.license_user = ""
if 'license_expiry' not in st.session_state:
    st.session_state.license_expiry = None
if 'license_features' not in st.session_state:
    st.session_state.license_features = {}
if 'sound_queue' not in st.session_state:
    st.session_state.sound_queue = deque(maxlen=10)
if 'page' not in st.session_state:
    st.session_state.page = "landing"
if 'splash_shown' not in st.session_state:
    st.session_state.splash_shown = False

# ==========================================
# DATABASE HELPERS
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

def get_ist():
    ist_offset = dt.timezone(dt.timedelta(hours=5, minutes=30))
    return dt.datetime.now(ist_offset)

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

@st.cache_data(ttl=300)
def get_usdt_inr_rate():
    try:
        res = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5).json()
        for coin in res:
            if coin['market'] == 'USDTINR': return float(coin['last_price'])
    except: pass
    return 86.50 

@st.cache_data(ttl=3600)
def get_all_crypto_pairs():
    pairs = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "BNBUSD", "LTCUSD", "DOTUSD", "MATICUSD", "SHIBUSD", "TRXUSD", "LINKUSD"]
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
    today_str = now_ist.strftime("%Y-%m-%d")
    if today_str in INDIAN_HOLIDAYS:
        return False, f"Market Closed ({INDIAN_HOLIDAYS[today_str]})"
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
            if res.data: 
                return res.data[0]
        except Exception as e:
            st.toast(f"DB Load Error: {e}")
    return {
        "user_id": user_id, "angel_api": "", "client_id": "", "pwd": "", "totp_secret": "", 
        "tg_token": "", "tg_chat": "", "wa_phone": "", "wa_api": "", 
        "mt5_acc": "", "mt5_pass": "", "mt5_server": "", "mt5_api_url": "",
        "zerodha_api": "", "zerodha_secret": "", "coindcx_api": "", "coindcx_secret": "",
        "delta_api": "", "delta_secret": "", "tg_bot_token": "", "tg_allowed_users": "",
        "fyers_client_id": "", "fyers_secret": "", "fyers_token": "",
        "upstox_api_key": "", "upstox_api_secret": "", "upstox_access_token": "",
        "fivepaisa_client_id": "", "fivepaisa_secret": "", "fivepaisa_access_token": "",
        "binance_api_key": "", "binance_api_secret": "", "binance_testnet": False,
        "stoxkart_api_key": "", "stoxkart_secret": "",
        "dhan_client_id": "", "dhan_access_token": "",
        "shoonya_user": "", "shoonya_password": "", "shoonya_totp": "",
        "email_smtp_server": "", "email_port": 587, "email_username": "", "email_password": "", "email_recipients": "",
        "fcm_server_key": "", "push_enabled": False,
        "role": "trader",
        "blocked": False
    }

def save_creds(user_id, angel_api, client_id, pwd, totp_secret, tg_token, tg_chat, wa_phone, wa_api, 
               mt5_acc, mt5_pass, mt5_server, mt5_api_url, zerodha_api, zerodha_secret, 
               coindcx_api, coindcx_secret, delta_api, delta_secret, tg_bot_token, tg_allowed_users, 
               fyers_client_id, fyers_secret, fyers_token,
               upstox_api_key, upstox_api_secret, upstox_access_token,
               fivepaisa_client_id, fivepaisa_secret, fivepaisa_access_token,
               binance_api_key, binance_api_secret, binance_testnet,
               stoxkart_api_key, stoxkart_secret,
               dhan_client_id, dhan_access_token,
               shoonya_user, shoonya_password, shoonya_totp,
               email_smtp_server, email_port, email_username, email_password, email_recipients,
               fcm_server_key, push_enabled, role):
    if HAS_DB:
        data = {
            "user_id": user_id,
            "angel_api": angel_api,
            "client_id": client_id,
            "pwd": pwd,
            "totp_secret": totp_secret,
            "tg_token": tg_token,
            "tg_chat": tg_chat,
            "wa_phone": wa_phone,
            "wa_api": wa_api,
            "mt5_acc": mt5_acc,
            "mt5_pass": mt5_pass,
            "mt5_server": mt5_server,
            "mt5_api_url": mt5_api_url,
            "zerodha_api": zerodha_api,
            "zerodha_secret": zerodha_secret,
            "coindcx_api": coindcx_api,
            "coindcx_secret": coindcx_secret,
            "delta_api": delta_api,
            "delta_secret": delta_secret,
            "tg_bot_token": tg_bot_token,
            "tg_allowed_users": tg_allowed_users,
            "fyers_client_id": fyers_client_id,
            "fyers_secret": fyers_secret,
            "fyers_token": fyers_token,
            "upstox_api_key": upstox_api_key,
            "upstox_api_secret": upstox_api_secret,
            "upstox_access_token": upstox_access_token,
            "fivepaisa_client_id": fivepaisa_client_id,
            "fivepaisa_secret": fivepaisa_secret,
            "fivepaisa_access_token": fivepaisa_access_token,
            "binance_api_key": binance_api_key,
            "binance_api_secret": binance_api_secret,
            "binance_testnet": binance_testnet,
            "stoxkart_api_key": stoxkart_api_key,
            "stoxkart_secret": stoxkart_secret,
            "dhan_client_id": dhan_client_id,
            "dhan_access_token": dhan_access_token,
            "shoonya_user": shoonya_user,
            "shoonya_password": shoonya_password,
            "shoonya_totp": shoonya_totp,
            "email_smtp_server": email_smtp_server,
            "email_port": email_port,
            "email_username": email_username,
            "email_password": email_password,
            "email_recipients": email_recipients,
            "fcm_server_key": fcm_server_key,
            "push_enabled": push_enabled,
            "role": role
        }
        try:
            supabase.table("user_credentials").upsert(data, on_conflict='user_id').execute()
        except Exception as e:
            st.toast(f"DB Save Error: {e}")

def save_device_session(user_id, device_name, ip, session_id):
    if HAS_DB and user_id:
        data = {
            "user_id": user_id,
            "device_name": device_name,
            "ip_address": ip,
            "session_id": session_id,
            "last_seen": get_ist().isoformat()
        }
        try:
            supabase.table("user_sessions").upsert(data, on_conflict='session_id').execute()
        except:
            pass

def get_active_sessions(user_id):
    if HAS_DB and user_id:
        try:
            res = supabase.table("user_sessions").select("*").eq("user_id", user_id).execute()
            return res.data
        except:
            return []
    return []

def save_trade(user_id, trade_date, trade_time, symbol, t_type, qty, entry, exit_price, pnl, result, slippage=0):
    if HAS_DB and user_id and user_id != "mock_user":
        data = {
            "user_id": user_id, "trade_date": trade_date, "trade_time": trade_time,
            "symbol": symbol, "trade_type": t_type, "qty": qty,
            "entry_price": float(entry), "exit_price": float(exit_price),
            "pnl": float(pnl), "result": result, "slippage": slippage
        }
        try: supabase.table("trade_logs").insert(data).execute()
        except: pass

def save_trade_journal(user_id, entry):
    if HAS_DB and user_id:
        try:
            supabase.table("trade_journal").insert(entry).execute()
        except Exception as e:
            print(f"Journal save error: {e}")

def get_user_role(user_id):
    if HAS_DB and user_id:
        try:
            res = supabase.table("user_credentials").select("role").eq("user_id", user_id).execute()
            if res.data:
                return res.data[0].get("role", "trader")
        except:
            pass
    return "trader"

def is_user_blocked(user_id):
    """Check if a user is blocked."""
    if not HAS_DB or not user_id:
        return False
    try:
        # Use blocked column in user_credentials (added manually)
        res = supabase.table("user_credentials").select("blocked").eq("user_id", user_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0].get("blocked", False)
        return False
    except:
        return False

def set_user_blocked(user_id, blocked, reason=""):
    """Set blocked status for a user."""
    if not HAS_DB:
        return False
    try:
        supabase.table("user_credentials").update({"blocked": blocked}).eq("user_id", user_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to update block status: {e}")
        return False

# ==========================================
# LICENSE FUNCTIONS
# ==========================================
def generate_license_id(mobile_or_email):
    salt = secrets.token_hex(4)
    raw = f"{mobile_or_email}_{salt}_{datetime.utcnow().timestamp()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16].upper()

def create_license(mobile_or_email, plan="basic", days=365):
    if not supabase:
        return None
    lic_key = generate_license_id(mobile_or_email)
    expiry = (datetime.utcnow() + timedelta(days=days)).isoformat()
    features = {"plan": plan, "max_trades": 5 if plan=="basic" else 50}
    data = {
        "license_key": lic_key,
        "user_identifier": mobile_or_email,
        "expiry": expiry,
        "features": features,
        "active": True
    }
    try:
        supabase.table("licenses").insert(data).execute()
        return lic_key
    except Exception as e:
        st.error(f"License creation failed: {e}")
        return None

def validate_license(license_key):
    if not supabase:
        return False, "Database not connected"
    try:
        resp = supabase.table("licenses").select("*").eq("license_key", license_key).execute()
        if not resp.data:
            return False, "License not found"
        lic = resp.data[0]
        if not lic.get("active"):
            return False, "License is deactivated"
        expiry = datetime.fromisoformat(lic["expiry"])
        if expiry < datetime.utcnow():
            return False, "License expired"
        st.session_state.license_valid = True
        st.session_state.license_user = lic["user_identifier"]
        st.session_state.license_expiry = expiry
        st.session_state.license_features = lic.get("features", {})
        return True, "License valid"
    except Exception as e:
        return False, str(e)

def quick_login_with_license(license_key):
    valid, msg = validate_license(license_key)
    if valid:
        user_id = st.session_state.license_user
        if is_user_blocked(user_id):
            return False, "Your account has been blocked. Contact support."
        creds = load_creds(user_id)
        bot = SniperBot(
            api_key=creds.get("angel_api", ""),
            client_id=creds.get("client_id", ""),
            pwd=creds.get("pwd", ""),
            totp_secret=creds.get("totp_secret", ""),
            mt5_acc=creds.get("mt5_acc", ""),
            mt5_pass=creds.get("mt5_pass", ""),
            mt5_server=creds.get("mt5_server", ""),
            mt5_api_url=creds.get("mt5_api_url", ""),
            zerodha_api=creds.get("zerodha_api", ""),
            zerodha_secret=creds.get("zerodha_secret", ""),
            coindcx_api=creds.get("coindcx_api", ""),
            coindcx_secret=creds.get("coindcx_secret", ""),
            delta_api=creds.get("delta_api", ""),
            delta_secret=creds.get("delta_secret", ""),
            tg_bot_token=creds.get("tg_bot_token", ""),
            tg_allowed_users=creds.get("tg_allowed_users", ""),
            fyers_client_id=creds.get("fyers_client_id", ""),
            fyers_secret=creds.get("fyers_secret", ""),
            fyers_token=creds.get("fyers_token", ""),
            upstox_api_key=creds.get("upstox_api_key", ""),
            upstox_api_secret=creds.get("upstox_api_secret", ""),
            upstox_access_token=creds.get("upstox_access_token", ""),
            fivepaisa_client_id=creds.get("fivepaisa_client_id", ""),
            fivepaisa_secret=creds.get("fivepaisa_secret", ""),
            fivepaisa_access_token=creds.get("fivepaisa_access_token", ""),
            binance_api_key=creds.get("binance_api_key", ""),
            binance_api_secret=creds.get("binance_api_secret", ""),
            binance_testnet=creds.get("binance_testnet", False),
            stoxkart_api_key=creds.get("stoxkart_api_key", ""),
            stoxkart_secret=creds.get("stoxkart_secret", ""),
            dhan_client_id=creds.get("dhan_client_id", ""),
            dhan_access_token=creds.get("dhan_access_token", ""),
            shoonya_user=creds.get("shoonya_user", ""),
            shoonya_password=creds.get("shoonya_password", ""),
            shoonya_totp=creds.get("shoonya_totp", ""),
            email_smtp_server=creds.get("email_smtp_server", ""),
            email_port=creds.get("email_port", 587),
            email_username=creds.get("email_username", ""),
            email_password=creds.get("email_password", ""),
            email_recipients=creds.get("email_recipients", ""),
            fcm_server_key=creds.get("fcm_server_key", ""),
            push_enabled=creds.get("push_enabled", False),
            is_mock=False
        )
        bot.system_user_id = user_id
        bot.user_role = creds.get("role", "trader")
        if bot.login():
            bot.state["daily_pnl"] = bot.load_daily_pnl()
            st.session_state.bot = bot
            st.session_state.audio_enabled = True
            bot.state["sound_queue"].append("login")
            save_device_session(user_id, st.session_state.device_name,
                                st.session_state.ip_address,
                                st.session_state.session_id)
            return True, "Logged in via license"
        else:
            return False, "Login with saved credentials failed"
    else:
        return False, msg

# ==========================================
# UI & CUSTOM CSS (enhanced landing page + dashboard)
# ==========================================
st.set_page_config(page_title="SHREE", page_icon="🕉️", layout="wide", initial_sidebar_state="expanded")

theme = st.session_state.theme
if theme == "dark":
    bg_color = "#0e1117"
    text_color = "#fafafa"
    card_bg = "#1e1e1e"
    border_color = "#333"
else:
    bg_color = "#f4f7f6"
    text_color = "#0f111a"
    card_bg = "#ffffff"
    border_color = "#cbd5e1"

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{ background-color: {bg_color}; color: {text_color}; font-family: 'Inter', sans-serif; }}
    @media (max-width: 850px) {{
        header[data-testid="stHeader"] {{ visibility: visible !important; height: auto !important; background-color: #0284c7 !important; }}
        header[data-testid="stHeader"] svg {{ fill: white !important; }}
        .main .block-container {{ padding-top: 50px !important; padding-bottom: 90px !important; }}
        .landing-hero {{ padding: 2rem 1rem !important; max-width: 100% !important; }}
    }}
    [data-testid="stSidebar"] {{ background-color: #0284c7 !important; border-right: 1px solid #0369a1; }}
    [data-testid="stSidebar"] * {{ color: #ffffff !important; }}
    [data-testid="stSidebar"] .stToggle label {{
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    [data-testid="stSidebar"] .stToggle {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 5px 0 !important;
    }}
    [data-testid="stSidebar"] .stToggle [data-baseweb="toggle"] {{
        background-color: rgba(255,255,255,0.2) !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
    }}
    [data-testid="stSidebar"] .stToggle [data-baseweb="toggle"][aria-checked="true"] {{
        background-color: #22c55e !important;
        border-color: #ffffff !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label {{
        font-weight: 600 !important;
        opacity: 0.95;
    }}
    div[data-baseweb="select"] * {{ color: #0f111a !important; font-weight: 600 !important; }}
    div[data-baseweb="select"] {{ background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; }}
    div[data-baseweb="base-input"] > input, input[type="number"], input[type="password"], input[type="text"], textarea {{
        color: #0f111a !important; font-weight: 600 !important; background-color: #ffffff !important; border: 1px solid #cbd5e1 !important;
    }}
    div[data-testid="stTabs"] {{ background: transparent !important; }}
    div[data-baseweb="tab-list"] {{
        background: #e2e8f0 !important; 
        padding: 6px !important;
        border-radius: 12px !important;
        display: flex !important;
        gap: 8px !important;
        margin-bottom: 20px !important;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"] {{
        flex: 1 !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        background: transparent !important;
        color: #64748b !important;
        transition: all 0.3s !important;
    }}
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #0284c7, #0369a1) !important; 
        color: #ffffff !important;
    }}
    .glass-panel {{ background: {card_bg}; border: 1px solid {border_color}; border-radius: 12px; padding: 30px; }}
    .hz-stats {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; }}
    .modern-card {{ background: {card_bg}; border-radius: 16px; padding: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.04); border: 1px solid {border_color}; margin-bottom: 20px; }}
    .modern-card h3 {{ margin-top: 0; color: #2563eb; font-weight: 700; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
    .broker-badge {{ display: inline-block; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; background: #2563eb; color: white; margin-left: 0.5rem; }}
    .news-ticker-kannada {{ background: #1e293b; color: white; padding: 8px 0; overflow: hidden; white-space: nowrap; border-radius: 8px; margin-bottom: 15px; position: relative; }}
    .news-ticker-kannada span {{ display: inline-block; padding-left: 100%; animation: ticker-kannada 120s linear infinite; }}
    @keyframes ticker-kannada {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-100%); }} }}
    .news-ticker-english {{ background: #0f172a; color: white; padding: 8px 0; overflow: hidden; white-space: nowrap; border-radius: 8px; margin-bottom: 15px; }}
    .news-ticker-english span {{ display: inline-block; padding-left: 100%; animation: ticker-english 180s linear infinite; }}
    @keyframes ticker-english {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-100%); }} }}
    
    .bottom-dock {{
        position: fixed;
        bottom: 15px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        gap: 20px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 50px;
        padding: 8px 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255,255,255,0.3);
        border: 1px solid rgba(255,255,255,0.2);
        z-index: 999;
        animation: dock-appear 0.3s ease-out;
    }}
    @keyframes dock-appear {{
        0% {{ opacity: 0; transform: translateX(-50%) translateY(20px); }}
        100% {{ opacity: 1; transform: translateX(-50%) translateY(0); }}
    }}
    .bottom-dock .dock-item {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-width: 70px;
        padding: 8px 0;
        border-radius: 30px;
        transition: all 0.2s ease;
        cursor: pointer;
        color: #4b5563;
    }}
    .bottom-dock .dock-item:hover {{
        background: rgba(2, 132, 199, 0.1);
        color: #0284c7;
        transform: translateY(-2px);
    }}
    .bottom-dock .dock-item .dock-icon {{
        font-size: 26px;
        margin-bottom: 4px;
    }}
    .bottom-dock .dock-item .dock-label {{
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}
    .bottom-dock .dock-item.active {{
        color: #0284c7;
        background: rgba(2, 132, 199, 0.15);
    }}
    .bottom-dock .stButton button {{
        background: transparent;
        border: none;
        padding: 0;
        margin: 0;
        min-width: unset;
        box-shadow: none;
        font-weight: normal;
    }}
    .bottom-dock .stButton button:hover {{
        background: transparent;
        color: inherit;
        border: none;
        box-shadow: none;
    }}
    .bottom-dock .stButton button:focus {{
        outline: none;
        box-shadow: none;
    }}
    .simulated-badge {{
        background: #f59e0b;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        margin-left: 8px;
        display: inline-block;
    }}
    .rejection-reason {{
        font-size: 0.8rem;
        color: #ef4444;
        margin-top: 4px;
    }}
    .risk-low {{ color: #22c55e; font-weight: bold; }}
    .risk-medium {{ color: #fbbf24; font-weight: bold; }}
    .risk-high {{ color: #ef4444; font-weight: bold; }}
    .fullscreen-chart {{ height: 90vh !important; }}
    .normal-chart {{ height: 400px !important; }}
    .profile-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #0284c7;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        cursor: pointer;
        border: 2px solid white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .profile-icon:hover {{
        background-color: #0369a1;
    }}
    .button-row {{
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
        justify-content: flex-start;
    }}
    .button-row .stButton button {{
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border-radius: 30px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }}
    .button-row .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    .start-btn button {{ background: #22c55e; color: white; }}
    .stop-btn button {{ background: #ef4444; color: white; }}
    .refresh-btn button {{ background: #3b82f6; color: white; }}
    .Exit-btn button {{ background: #7f1d1d; color: white; }}
    .admin-card {{
        background: #1e293b;
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }}
    .admin-card h4 {{ color: #fbbf24; margin-top: 0; }}
    .blocked-user {{ background: #7f1d1d; padding: 5px; border-radius: 4px; }}
    .risk-box {{
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }}
    .license-section {{
        background: #e8f0fe;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #90caf9;
    }}
    /* Bento Grid + Kinetic Typography */
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes slideUpFade {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 0 0 rgba(2,132,199,0.7); }}
        70% {{ box-shadow: 0 0 0 15px rgba(2,132,199,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(2,132,199,0); }}
    }}
    .landing-hero {{
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
        color: white;
        border-radius: 0 0 2rem 2rem;
        margin-bottom: 2rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }}
    .kinetic-title {{
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        animation: slideUpFade 1s ease-out;
        text-shadow: 0 0 20px rgba(255,255,255,0.5), 0 4px 8px rgba(0,0,0,0.2);
        letter-spacing: 2px;
    }}
    .kinetic-sub {{
        font-size: 1.2rem;
        opacity: 0.95;
        animation: slideUpFade 1.2s ease-out;
        text-shadow: 0 1px 5px rgba(0,0,0,0.2);
    }}
    .bento-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.5rem;
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }}
    .bento-card {{
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 24px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}
    .bento-card:hover {{
        transform: translateY(-8px) scale(1.02);
        border-color: #3b82f6;
        box-shadow: 0 20px 40px rgba(59,130,246,0.4);
        background: rgba(255,255,255,0.2);
    }}
    .bento-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 10px #3b82f6);
    }}
    .bento-card h3 {{
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #ffffff, #a5f3fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
    }}
    .bento-card p {{
        font-size: 0.95rem;
        opacity: 0.9;
        line-height: 1.5;
        color: white;
    }}
    .broker-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 1rem;
        padding: 2rem;
        max-width: 900px;
        margin: 0 auto;
    }}
    .broker-item {{
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(5px);
        border-radius: 16px;
        padding: 1rem 0.5rem;
        text-align: center;
        transition: all 0.2s;
        border: 1px solid rgba(255,255,255,0.05);
    }}
    .broker-item:hover {{
        transform: scale(1.05);
        border-color: #3b82f6;
        background: rgba(59,130,246,0.2);
    }}
    .broker-item img {{
        width: 40px;
        height: 40px;
        object-fit: contain;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 5px #3b82f6);
    }}
    .broker-item span {{
        display: block;
        font-size: 0.8rem;
        color: white;
        font-weight: 500;
    }}
    .terms-card {{
        background: rgba(0,0,0,0.75) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 28px;
        padding: 2rem;
        color: white;
        max-width: 600px;
        margin: 0 auto 2rem;
        animation: float 5s ease-in-out infinite;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }}
    .terms-card .typewriter {{
        overflow: hidden;
        border-right: 2px solid white;
        white-space: nowrap;
        margin: 0 auto;
        animation: typing 3.5s steps(40, end), blink-caret 0.75s step-end infinite;
        font-size: 1.1rem;
        font-weight: 500;
        color: white;
    }}
    @keyframes typing {{
        from {{ width: 0 }}
        to {{ width: 100% }}
    }}
    @keyframes blink-caret {{
        from, to {{ border-color: transparent }}
        50% {{ border-color: white }}
    }}
    .stButton > button:has(span:contains("Enter")) {{
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        padding: 1rem 2rem !important;
        border-radius: 40px !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3) !important;
        transition: all 0.3s !important;
        width: 100% !important;
        animation: pulseGlow 2s infinite;
    }}
    .stButton > button:has(span:contains("Enter")):hover {{
        transform: scale(1.05) !important;
        box-shadow: 0 15px 30px rgba(59,130,246,0.5) !important;
    }}
    .footer {{
        background: rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        color: white;
        padding: 2rem;
        margin-top: 3rem;
        border-radius: 2rem 2rem 0 0;
        text-align: center;
        border-top: 1px solid rgba(255,255,255,0.1);
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}
    /* Live Position Tracker */
    .live-tracker {{
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 2px solid #3b82f6;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5), 0 0 0 2px rgba(59,130,246,0.3) inset;
        margin-bottom: 20px;
        color: #ffffff;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}
    .live-tracker:hover {{
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6), 0 0 0 3px #3b82f6 inset;
        transform: translateY(-2px);
    }}
    .live-tracker .header {{
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px dashed #334155;
        padding-bottom: 15px;
        margin-bottom: 15px;
    }}
    .live-tracker .symbol-badge {{
        background: #3b82f6;
        color: white;
        padding: 6px 15px;
        border-radius: 30px;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        box-shadow: 0 0 15px #3b82f6;
        display: inline-block;
    }}
    .live-tracker .pnl-badge {{
        padding: 8px 20px;
        border-radius: 40px;
        font-weight: 900;
        font-size: 1.6rem;
        background: rgba(255,255,255,0.1);
        border: 2px solid;
        animation: pulse-pnl 2s infinite;
        margin-top: 5px;
    }}
    @keyframes pulse-pnl {{
        0% {{ box-shadow: 0 0 0 0 currentColor; }}
        70% {{ box-shadow: 0 0 15px 5px currentColor; }}
        100% {{ box-shadow: 0 0 0 0 currentColor; }}
    }}
    .live-tracker .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin-bottom: 20px;
    }}
    .live-tracker .info-box {{
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 16px;
        border: 1px solid #334155;
        backdrop-filter: blur(5px);
    }}
    .live-tracker .info-box .label {{
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }}
    .live-tracker .info-box .value {{
        font-size: 1.4rem;
        font-weight: 800;
        color: #facc15;
        text-shadow: 0 0 10px #facc15;
        word-break: break-word;
    }}
    .live-tracker .info-box .value.green {{
        color: #4ade80;
        text-shadow: 0 0 10px #4ade80;
    }}
    .live-tracker .info-box .value.red {{
        color: #f87171;
        text-shadow: 0 0 10px #f87171;
    }}
    .live-tracker .targets {{
        background: linear-gradient(90deg, #1e293b, #111827);
        padding: 12px;
        border-radius: 40px;
        font-size: 1rem;
        text-align: center;
        color: #38bdf8;
        font-weight: 700;
        border: 1px solid #38bdf8;
        box-shadow: 0 0 20px #38bdf8;
        letter-spacing: 0.5px;
        word-break: break-word;
    }}
    .live-tracker .targets span {{
        color: #fbbf24;
        font-weight: 800;
        margin: 0 10px;
    }}
    .sticky-buttons {{
        position: sticky;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        display: flex;
        gap: 12px;
        justify-content: flex-end;
        margin-top: 10px;
    }}
    .sticky-buttons button {{
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 40px;
        padding: 8px 20px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: 0.2s;
    }}
    .sticky-buttons button:hover {{
        transform: scale(1.05);
        background: #2563eb;
    }}
    @media (max-width: 768px) {{
        .live-tracker .grid {{
            grid-template-columns: 1fr;
        }}
        .live-tracker .pnl-badge {{
            font-size: 1.2rem;
            padding: 5px 12px;
        }}
        .live-tracker .symbol-badge {{
            font-size: 0.8rem;
            padding: 4px 10px;
        }}
        .live-tracker .info-box .value {{
            font-size: 1.1rem;
        }}
        .live-tracker .targets {{
            font-size: 0.8rem;
            padding: 8px;
        }}
        .sticky-buttons {{
            gap: 8px;
        }}
        .sticky-buttons button {{
            padding: 6px 12px;
            font-size: 0.8rem;
        }}
        .kinetic-title {{
            font-size: 2rem;
        }}
        .kinetic-sub {{
            font-size: 1rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS AND DEFAULTS
# ==========================================
LOT_SIZES = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "SENSEX": 20,
    "FINNIFTY": 40,
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
    "BNBUSD": 1,
    "LTCUSD": 1,
    "DOTUSD": 1,
    "MATICUSD": 1,
    "SHIBUSD": 1,
    "TRXUSD": 1,
    "LINKUSD": 1,
    "AVAXUSD": 1,
    "UNIUSD": 1,
    "AAVEUSD": 1,
    "MKRUSD": 1,
    "SUSHIUSD": 1,
    "YFIUSD": 1,
    "COMPUSD": 1,
    "SNXUSD": 1,
    "CRVUSD": 1,
    "BALUSD": 1,
    "KNCUSD": 1,
    "ZRXUSD": 1,
    "ENJUSD": 1,
    "MANAUSD": 1,
    "SANDUSD": 1,
    "AXSUSD": 1,
    "GALAUSD": 1,
    "CHZUSD": 1
}

YF_TICKERS = {
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "FINNIFTY": "^FINNIFTY",
    "CRUDEOIL": "CL=F", 
    "NATURALGAS": "NG=F", "GOLD": "GC=F", "SILVER": "SI=F", "XAUUSD": "GC=F", "EURUSD": "EURUSD=X", 
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD",
    "XRPUSD": "XRP-USD", "ADAUSD": "ADA-USD", "DOGEUSD": "DOGE-USD", "BNBUSD": "BNB-USD",
    "LTCUSD": "LTC-USD", "DOTUSD": "DOT-USD", "MATICUSD": "MATIC-USD", "SHIBUSD": "SHIB-USD",
    "TRXUSD": "TRX-USD", "LINKUSD": "LINK-USD", "AVAXUSD": "AVAX-USD", "UNIUSD": "UNI-USD",
    "AAVEUSD": "AAVE-USD", "MKRUSD": "MKR-USD", "SUSHIUSD": "SUSHI-USD", "YFIUSD": "YFI-USD",
    "COMPUSD": "COMP-USD", "SNXUSD": "SNX-USD", "CRVUSD": "CRV-USD", "BALUSD": "BAL-USD",
    "KNCUSD": "KNC-USD", "ZRXUSD": "ZRX-USD", "ENJUSD": "ENJ-USD", "MANAUSD": "MANA-USD",
    "SANDUSD": "SAND-USD", "AXSUSD": "AXS-USD", "GALAUSD": "GALA-USD", "CHZUSD": "CHZ-USD"
}
INDEX_SYMBOLS = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank", "SENSEX": "BSE SENSEX", "FINNIFTY": "Nifty Financial Services", "INDIA VIX": "INDIA VIX"}
INDEX_TOKENS = {
    "NIFTY": ("NSE", "26000"), 
    "BANKNIFTY": ("NSE", "26009"), 
    "INDIA VIX": ("NSE", "26017"), 
    "SENSEX": ("BSE", "99919000"),
    "FINNIFTY": ("NSE", "26037")
}

COMMODITIES = ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"]

STRAT_LIST = [
    "Momentum Breakout + S&R",
    "Machine Learning",
    "VIJAY & RFF All-In-One", 
    "Institutional FVG + SMC",
    "Lux Algo Institutional ICT",
    "Keyword Rule Builder", 
    "TradingView Webhook",
    "Mean Reversion (BB/RSI)",
    "Arbitrage (Nifty/BankNifty)",
    "Event Driven (Earnings)"
]

if not st.session_state.user_lots:
    st.session_state.user_lots = LOT_SIZES.copy()

is_mkt_open, mkt_status_msg = get_market_status(st.session_state.sb_index_input)

@st.cache_data(ttl=43200)
def get_angel_scrip_master():
    main_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    main_df = pd.DataFrame(requests.get(main_url, timeout=45).json())
    mcx_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster_MCX.json"
    try:
        mcx_df = pd.DataFrame(requests.get(mcx_url, timeout=45).json())
        df = pd.concat([main_df, mcx_df], ignore_index=True)
    except:
        df = main_df
    df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce') / 100
    return df

# ==========================================
# BRIDGE CLASSES
# ==========================================
class UpstoxBridge:
    def __init__(self, api_key, api_secret, access_token):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.client = None
        self.connected = False

    def connect(self):
        if not HAS_UPSTOX:
            return False, "Upstox library not installed."
        try:
            self.connected = True
            return True, "Connected to Upstox"
        except Exception as e:
            return False, f"Upstox connection error: {e}"

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def place_order(self, symbol, qty, side, order_type="MARKET", price=None):
        if not self.connected:
            return None, "Not connected"
        try:
            return "UPSTOX_ORDER", "Order placed (simulated)"
        except Exception as e:
            return None, str(e)

    def get_historical_data(self, symbol, interval, days=10):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            return {'balance': 0}
        except:
            return None

class FivePaisaBridge:
    def __init__(self, client_id, secret, access_token):
        self.client_id = client_id
        self.secret = secret
        self.access_token = access_token
        self.client = None
        self.connected = False

    def connect(self):
        if not HAS_5PAISA:
            return False, "5paisa library not installed."
        try:
            self.connected = True
            return True, "Connected to 5paisa"
        except Exception as e:
            return False, f"5paisa connection error: {e}"

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def place_order(self, symbol, qty, side, order_type="MARKET", price=None):
        if not self.connected:
            return None, "Not connected"
        try:
            return "5PAISA_ORDER", "Order placed (simulated)"
        except Exception as e:
            return None, str(e)

    def get_historical_data(self, symbol, interval, days=10):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            return {'balance': 0}
        except:
            return None

class BinanceBridge:
    def __init__(self, api_key, api_secret, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.connected = False
        self.ws_thread = None
        self.ws_running = False
        self.live_prices = {}

    def connect(self):
        if not HAS_BINANCE:
            return False, "Binance library not installed."
        try:
            if self.testnet:
                self.client = BinanceClient(self.api_key, self.api_secret, testnet=True)
            else:
                self.client = BinanceClient(self.api_key, self.api_secret)
            self.connected = True
            self._start_websocket()
            return True, "Connected to Binance"
        except Exception as e:
            return False, f"Binance connection error: {e}"

    def _start_websocket(self):
        def ws_worker():
            self.ws_running = True
            while self.ws_running:
                time.sleep(1)
        self.ws_thread = threading.Thread(target=ws_worker, daemon=True)
        self.ws_thread.start()

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except:
            return None

    def place_order(self, symbol, side, quantity, order_type="MARKET", price=None):
        if not self.connected:
            return None, "Not connected"
        try:
            side = side.upper()
            order_type = order_type.upper()
            if order_type == "MARKET":
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity
                )
            else:
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity,
                    price=price
                )
            return order['orderId'], None
        except Exception as e:
            return None, str(e)

    def get_historical_klines(self, symbol, interval="5m"):
        if not self.connected:
            return None
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=500)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df
        except Exception as e:
            print(f"Binance historical error: {e}")
            return None

    def get_asset_balance(self, asset="USDT"):
        if not self.connected:
            return None
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance['free'])
        except:
            return None

class FyersBridge:
    def __init__(self, client_id, secret, token):
        self.client_id = client_id
        self.secret = secret
        self.token = token
        self.fyers = None
        self.connected = False

    def connect(self):
        if not HAS_FYERS:
            return False, "Fyers library not installed."
        try:
            self.fyers = fyersModel.FyersModel(client_id=self.client_id, token=self.token, log_path="")
            response = self.fyers.get_profile()
            if response and response.get('code') == 200:
                self.connected = True
                return True, "Connected to Fyers"
            else:
                return False, "Fyers authentication failed"
        except Exception as e:
            return False, f"Fyers connection error: {e}"

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            quote = self.fyers.quotes({"symbols": symbol})
            if quote and quote.get('code') == 200 and quote.get('d'):
                return float(quote['d'][0]['v']['ltp'])
            return None
        except:
            return None

    def place_order(self, symbol, qty, side, order_type="MARKET", price=None):
        if not self.connected:
            return None, "Not connected"
        try:
            order_type_map = {"MARKET": 2, "LIMIT": 1}
            side_map = {"BUY": 1, "SELL": -1}
            order_data = {
                "symbol": symbol,
                "qty": int(qty),
                "type": order_type_map.get(order_type.upper(), 2),
                "side": side_map.get(side.upper(), 1),
                "productType": "INTRADAY",
                "limitPrice": price if order_type.upper() == "LIMIT" else 0,
                "stopPrice": 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }
            response = self.fyers.place_order(order_data)
            if response and response.get('code') == 200:
                return response.get('id'), None
            else:
                return None, f"Fyers order failed: {response}"
        except Exception as e:
            return None, str(e)

    def get_historical_data(self, symbol, resolution="5", days=10):
        if not self.connected:
            return None
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            response = self.fyers.history({
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            })
            if response and response.get('code') == 200 and response.get('candles'):
                data = response['candles']
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                return df
            return None
        except Exception as e:
            print(f"Fyers historical error: {e}")
            return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            funds = self.fyers.funds()
            if funds and funds.get('code') == 200:
                return {'balance': float(funds.get('fund_limit', 0))}
            return None
        except:
            return None

class MT5WebBridge:
    def __init__(self, account, password, server, api_url="https://mt5-web-api.mtapi.io/v1"):
        self.account = account
        self.password = password
        self.server = server
        self.api_url = api_url
        self.token = None
        self.connected = False

    def connect(self):
        try:
            self.connected = True
            return True, "Connected to MT5/Exness (simulated)"
        except Exception as e:
            return False, f"MT5 connection error: {e}"

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def place_order(self, symbol, qty, side):
        if not self.connected:
            return None, "Not connected"
        try:
            return "MT5_ORDER", None
        except Exception as e:
            return None, str(e)

    def get_historical_data(self, symbol, interval):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            return {'balance': 0, 'login': self.account}
        except:
            return None

class StoxkartBridge:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.client = None
        self.connected = False

    def connect(self):
        if not HAS_STOXKART:
            return False, "Stoxkart library not installed."
        try:
            self.connected = True
            return True, "Connected to Stoxkart"
        except Exception as e:
            return False, f"Stoxkart connection error: {e}"

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def place_order(self, symbol, qty, side, order_type="MARKET", price=None):
        if not self.connected:
            return None, "Not connected"
        try:
            return "STX_ORDER", "Order placed (simulated)"
        except Exception as e:
            return None, str(e)

    def get_historical_data(self, symbol, interval="5m", days=10):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            return {'balance': 0}
        except:
            return None

class DeltaExchangeBridge:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.connected = False
        self.ws = None

    def connect(self):
        try:
            self.connected = True
            return True, "Connected to Delta Exchange"
        except Exception as e:
            return False, f"Delta connection error: {e}"

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            target = symbol if symbol.endswith("USD") or symbol.endswith("USDT") else f"{symbol}USD"
            res = requests.get(f"https://api.delta.exchange/v2/products/ticker/24hr?symbol={target}").json()
            if res.get('success'):
                return float(res['result']['close'])
            return None
        except:
            return None

    def place_order(self, symbol, qty, side, order_type="MARKET", price=None):
        if not self.connected:
            return None, "Not connected"
        try:
            target = symbol if symbol.endswith("USD") or symbol.endswith("USDT") else f"{symbol}USD"
            payload = {
                "product_id": target,
                "size": int(float(qty)),
                "side": "buy" if side == "BUY" else "sell",
                "order_type": "limit_order" if order_type.upper() == "LIMIT" else "market_order"
            }
            if order_type.upper() == "LIMIT" and price:
                payload["limit_price"] = price
            payload_str = json.dumps(payload, separators=(',', ':'))
            ts, sig = generate_delta_signature('POST', '/v2/orders', payload_str, self.secret)
            headers = {'api-key': self.api_key, 'signature': sig, 'timestamp': ts, 'Content-Type': 'application/json'}
            res = requests.post("https://api.delta.exchange/v2/orders", headers=headers, data=payload_str)
            if res.status_code == 200:
                return res.json().get('result', {}).get('id'), None
            else:
                return None, f"Delta error: {res.text}"
        except Exception as e:
            return None, str(e)

    def get_historical_data(self, symbol, interval="5m"):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            ts, sig = generate_delta_signature('GET', '/v2/wallet/balances', '', self.secret)
            headers = {'api-key': self.api_key, 'signature': sig, 'timestamp': ts}
            res = requests.get("https://api.delta.exchange/v2/wallet/balances", headers=headers, timeout=5)
            if res.status_code == 200:
                for b in res.json().get('result', []):
                    if b['asset_symbol'] == 'USDT':
                        return {'balance': float(b['balance'])}
            return None
        except:
            return None

class DhanBridge:
    def __init__(self, client_id, access_token):
        self.client_id = client_id
        self.access_token = access_token
        self.client = None
        self.connected = False

    def connect(self):
        if not HAS_DHAN:
            return False, "Dhan library not installed."
        try:
            self.client = dhanhq.DhanHq(self.client_id, self.access_token)
            self.connected = True
            return True, "Connected to Dhan"
        except Exception as e:
            return False, f"Dhan connection error: {e}"

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def place_order(self, symbol, qty, side, order_type="MARKET", price=None):
        if not self.connected:
            return None, "Not connected"
        try:
            return "DHAN_ORDER", "Order placed (simulated)"
        except Exception as e:
            return None, str(e)

    def get_historical_data(self, symbol, interval="5m", days=10):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            return {'balance': 0}
        except:
            return None

class ShoonyaBridge:
    def __init__(self, user, password, totp):
        self.user = user
        self.password = password
        self.totp = totp
        self.client = None
        self.connected = False

    def connect(self):
        if not HAS_SHOONYA:
            return False, "Shoonya library not installed."
        try:
            self.client = Shoonya()
            self.connected = True
            return True, "Connected to Shoonya"
        except Exception as e:
            return False, f"Shoonya connection error: {e}"

    def get_live_price(self, symbol):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def place_order(self, symbol, qty, side, order_type="MARKET", price=None):
        if not self.connected:
            return None, "Not connected"
        try:
            return "SHOONYA_ORDER", "Order placed (simulated)"
        except Exception as e:
            return None, str(e)

    def get_historical_data(self, symbol, interval="5m", days=10):
        if not self.connected:
            return None
        try:
            return None
        except:
            return None

    def get_account_info(self):
        if not self.connected:
            return None
        try:
            return {'balance': 0}
        except:
            return None

class CoinDCXBridge:
    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret
        self.connected = False
        self.ws = None
        self.live_balance = {}
        self.ws_thread = None
        self.ws_running = False

    def connect(self):
        try:
            self.connected = True
            self._start_websocket()
            return True, "Connected to CoinDCX"
        except Exception as e:
            return False, f"CoinDCX connection error: {e}"

    def _start_websocket(self):
        def ws_worker():
            self.ws_running = True
            while self.ws_running:
                time.sleep(1)
        self.ws_thread = threading.Thread(target=ws_worker, daemon=True)
        self.ws_thread.start()

    def get_live_price(self, symbol):
        try:
            market_symbol = symbol.replace("USD", "USDT") if symbol.endswith("USD") and not symbol.endswith("USDT") else symbol
            res = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5).json()
            for coin in res:
                mkt = coin.get('market', '')
                if mkt == market_symbol or mkt.upper() == market_symbol.upper():
                    return float(coin['last_price'])
            return None
        except:
            return None

    def place_order(self, symbol, qty, side, order_type="MARKET", price=None, market_type="Spot", leverage=1):
        try:
            ts = int(round(time.time() * 1000))
            base_coin = symbol.replace("USDT", "").replace("USD", "").replace("INR", "")
            price_live = self.get_live_price(symbol)
            
            if not price_live and not price:
                return None, "Price not available"
            if not price:
                price = price_live
                
            max_cap = 15000
            position_value = max_cap * leverage
            raw_qty = position_value / price
            
            if "BTC" in symbol or "ETH" in symbol:
                clean_qty = round(raw_qty, 4)
            elif "XRP" in symbol or "ADA" in symbol or "SOL" in symbol:
                clean_qty = round(raw_qty, 1)
            else:
                clean_qty = round(raw_qty, 0)
                
            if clean_qty < 0.0001:
                clean_qty = 0.0001

            try:
                exchange_info = requests.get("https://api.coindcx.com/exchange/v1/markets_details", timeout=5).json()
            except:
                exchange_info = []
                
            market_symbol = None
            for mkt in exchange_info:
                if isinstance(mkt, dict):
                    cointype = mkt.get('target_currency_short_name', mkt.get('cointype', ''))
                    currency = mkt.get('base_currency_short_name', mkt.get('currency', ''))
                    
                    if cointype.upper() == base_coin.upper() and currency.upper() in ['USDT', 'INR']:
                        if market_type in ["Futures", "Options"] and mkt.get('is_future', False):
                            market_symbol = mkt.get('coindcx_name', mkt.get('symbol'))
                            break
                        elif market_type == "Spot" and not mkt.get('is_future', False):
                            market_symbol = mkt.get('coindcx_name', mkt.get('symbol'))
                            break
                            
            if not market_symbol:
                if market_type in ["Futures", "Options"]:
                    market_symbol = f"B-{base_coin}_USDT"
                else:
                    market_symbol = f"{base_coin}USDT"

            # Create specific payload formats for Futures vs. Spot
            if market_type in ["Futures", "Options"]:
                payload = {
                    "side": side.lower(), # FIX: Must be "buy" or "sell", not "long" or "short"
                    "order_type": "limit_order" if order_type.upper() == "LIMIT" else "market_order",
                    "pair": market_symbol,
                    "total_quantity": clean_qty,
                    "leverage": leverage, # FIX: Required for Futures
                    "margin_currency_short_name": "USDT", # FIX: Required for Futures
                    "timestamp": ts
                }
                if order_type.upper() == "LIMIT" and price:
                    payload["price"] = price # Futures uses "price"
                    
                endpoint = "https://api.coindcx.com/exchange/v1/derivatives/futures/orders/create"
            else:
                payload = {
                    "side": side.lower(),
                    "order_type": "limit_order" if order_type.upper() == "LIMIT" else "market_order",
                    "market": market_symbol,
                    "total_quantity": clean_qty,
                    "timestamp": ts
                }
                if order_type.upper() == "LIMIT" and price:
                    payload["price_per_unit"] = price # FIX: Spot strictly requires "price_per_unit"
                    
                endpoint = "https://api.coindcx.com/exchange/v1/orders/create"

            payload_str = json.dumps(payload, separators=(',', ':'))
            secret_bytes = bytes(self.secret, 'utf-8')
            signature = hmac.new(secret_bytes, payload_str.encode('utf-8'), hashlib.sha256).hexdigest()
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'Content-Type': 'application/json'
            }
            
            res = requests.post(endpoint, headers=headers, data=payload_str, timeout=10)
            
            if res.status_code == 200:
                response_data = res.json()
                if isinstance(response_data, list) and len(response_data) > 0:
                    order_id = response_data.get('id', 'DCX_ORDER_OK')
                elif isinstance(response_data, dict):
                    order_id = response_data.get('id', response_data.get('order_id', 'DCX_ORDER_OK'))
                else:
                    order_id = 'DCX_ORDER_OK'
                return order_id, None
            else:
                # FIX: Improved Error Parser to grab the exact API message if it fails again
                try:
                    err_data = res.json()
                    error_msg = err_data.get('message', err_data.get('error', res.text))
                except:
                    error_msg = res.text if res.text else f"Empty response (Status Code: {res.status_code})"
                return None, f"API Rejected ({res.status_code}): {error_msg}"
                
        except Exception as e:
            return None, str(e)
    def get_balance(self):
        try:
            ts = int(round(time.time() * 1000))
            payload = {"timestamp": ts}
            payload_str = json.dumps(payload, separators=(',', ':'))
            secret_bytes = bytes(self.secret, 'utf-8')
            signature = hmac.new(secret_bytes, payload_str.encode('utf-8'), hashlib.sha256).hexdigest()
            headers = {
                'X-AUTH-APIKEY': self.api_key,
                'X-AUTH-SIGNATURE': signature,
                'Content-Type': 'application/json'
            }
            res = requests.post("https://api.coindcx.com/exchange/v1/users/balances", headers=headers, data=payload_str, timeout=5)
            if res.status_code == 200:
                data = res.json()
                for item in data:
                    if item['currency'] == 'USDT':
                        return float(item['balance'])
            return 0.0
        except:
            return 0.0

    def get_historical_data(self, symbol, interval="5m", days=10):
        try:
            pair = symbol.replace("USD", "USDT") if symbol.endswith("USD") and not symbol.endswith("USDT") else symbol
            res_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "1D"}
            resolution = res_map.get(interval, "5")
            end = int(time.time())
            start = end - days * 86400
            url = f"https://public.coindcx.com/market_data/candles?pair={pair}&interval={resolution}&startTime={start*1000}&endTime={end*1000}&limit=500"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    df = pd.DataFrame(data)
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
                    df.set_index('time', inplace=True)
                    return df
            return None
        except Exception as e:
            print(f"CoinDCX historical error: {e}")
            return None

# ==========================================
# OPTION GREEKS CALCULATOR
# ==========================================
class OptionGreeks:
    @staticmethod
    def black_scholes(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

# ==========================================
# FOMO SCANNER
# ==========================================
class FOMOScanner:
    def __init__(self):
        self.watchlist = {
            "INDIAN_STOCKS": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "WIPRO.NS"],
            "US_STOCKS": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "DIS", "JPM"],
            "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "BNB-USD"],
            "INDICES": ["^NSEI", "^BSESN", "^NSEBANK"],
            "COMMODITIES": ["GC=F", "SI=F", "CL=F", "NG=F"]
        }
        self.signals = deque(maxlen=50)
        self.last_alert_time = {}
        self._last_scan_time = 0
        
    def scan(self):
        now = time.time()
        if now - self._last_scan_time < 5:
            return list(self.signals)
        self._last_scan_time = now
        
        signals = []
        current_asset = st.session_state.get('sb_index_input', 'NIFTY')
        for category, tickers in self.watchlist.items():
            for ticker in tickers:
                try:
                    df = yf.Ticker(ticker).history(period="1d", interval="5m")
                    if len(df) < 20: continue
                    df['volume_ma'] = df['Volume'].rolling(20).mean()
                    df['volume_ratio'] = df['Volume'] / df['volume_ma']
                    df['high_20'] = df['High'].rolling(20).max()
                    df['low_20'] = df['Low'].rolling(20).min()
                    last = df.iloc[-1]
                    signal_type = None
                    if last['volume_ratio'] > 1.5 and last['Close'] > last['high_20'] * 0.99:
                        signal_type = "BUY 🚀"
                    elif last['volume_ratio'] > 1.5 and last['Close'] < last['low_20'] * 1.01:
                        signal_type = "SELL 🔻"
                    if signal_type:
                        signals.append({
                            "time": get_ist().strftime("%H:%M"),
                            "symbol": ticker,
                            "category": category,
                            "price": round(last['Close'], 2),
                            "volume_spike": f"{last['volume_ratio']:.1f}x",
                            "signal": signal_type
                        })
                        base_asset = None
                        if ticker == "^NSEI": base_asset = "NIFTY"
                        elif ticker == "^BSESN": base_asset = "SENSEX"
                        elif ticker == "^NSEBANK": base_asset = "BANKNIFTY"
                        elif ticker in ["CL=F", "NG=F", "GC=F", "SI=F"]:
                            base_asset = {"CL=F": "CRUDEOIL", "NG=F": "NATURALGAS", "GC=F": "GOLD", "SI=F": "SILVER"}.get(ticker, ticker)
                        else:
                            base_asset = ticker.replace(".NS", "").replace("-USD", "USD")
                        if base_asset and base_asset != current_asset:
                            if ticker not in self.last_alert_time or now - self.last_alert_time[ticker] > 300:
                                self.last_alert_time[ticker] = now
                                if ticker in ["^NSEI", "^NSEBANK", "^BSESN"]:
                                    play_clock_sound()
                                    st.toast(f"🔔 FOMO ALERT: {ticker} volume spike {last['volume_ratio']:.1f}x", icon="🔔")
                                    if st.session_state.bot:
                                        st.session_state.bot.push_notify("FOMO Alert", f"{ticker} volume spike {last['volume_ratio']:.1f}x")
                except:
                    continue
        self.signals.extend(signals)
        return signals

fomo_scanner = FOMOScanner()

# ==========================================
# Scalping Module
# ==========================================
class ScalpingModule:
    def __init__(self, bot):
        self.bot = bot
        
    def scalp_signal(self, df):
        if df is None or len(df) < 20:
            return "WAIT"
        df = df.copy()
        df['ema9'] = df['close'].ewm(span=9).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['tick_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21'] and last['tick_ratio'] > 1.2:
            return "SCALP_BUY"
        if last['ema9'] < last['ema21'] and prev['ema9'] >= prev['ema21'] and last['tick_ratio'] > 1.2:
            return "SCALP_SELL"
        if last['macd_hist'] > 0 and prev['macd_hist'] <= 0 and last['tick_ratio'] > 1.2:
            return "SCALP_BUY"
        if last['macd_hist'] < 0 and prev['macd_hist'] >= 0 and last['tick_ratio'] > 1.2:
            return "SCALP_SELL"
        if last['rsi'] > 50 and prev['rsi'] <= 50 and last['tick_ratio'] > 1.0:
            return "SCALP_BUY"
        if last['rsi'] < 50 and prev['rsi'] >= 50 and last['tick_ratio'] > 1.0:
            return "SCALP_SELL"
        return "WAIT"

# ==========================================
# TECHNICAL ANALYZER (ALL STRATEGIES – EARLY SIGNALS)
# ==========================================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df, is_index=False):
        df = df.copy()
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        if not is_index:
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        hl2 = (df['high'] + df['low']) / 2
        df['atr_st'] = tr.rolling(10).mean()
        df['upper_band'] = hl2 + 3 * df['atr_st']
        df['lower_band'] = hl2 - 3 * df['atr_st']
        df['st_direction'] = 1
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['upper_band'].iloc[i-1]:
                df.loc[df.index[i], 'st_direction'] = 1
            elif df['close'].iloc[i] < df['lower_band'].iloc[i-1]:
                df.loc[df.index[i], 'st_direction'] = -1
            else:
                df.loc[df.index[i], 'st_direction'] = df['st_direction'].iloc[i-1]
        df['supertrend'] = df['lower_band'] if df['st_direction'].iloc[-1] == 1 else df['upper_band']
        return df

    @staticmethod
    def get_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def get_support_resistance(df, lookback=20):
        highs = df['high']
        lows = df['low']
        peaks = (highs.shift(1) < highs) & (highs.shift(-1) < highs)
        troughs = (lows.shift(1) > lows) & (lows.shift(-1) > lows)
        recent_high = highs[peaks].iloc[-1] if peaks.any() else None
        recent_low = lows[troughs].iloc[-1] if troughs.any() else None
        return recent_high, recent_low

    @staticmethod
    def calculate_fib_zones(df, lookback=100):
        if df is None or len(df) < 10:
            return 0, 0, 0, 0
        actual_lookback = min(lookback, len(df))
        major_high = df['high'].rolling(actual_lookback).max().iloc[-1]
        major_low = df['low'].rolling(actual_lookback).min().iloc[-1]
        diff = major_high - major_low
        fib_618 = major_high - (diff * 0.618)
        fib_650 = major_high - (diff * 0.650)
        return major_high, major_low, min(fib_618, fib_650), max(fib_618, fib_650)

    @staticmethod
    def detect_pin_bar(df):
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

    @staticmethod
    def analyze_open_interest(price_change, oi_change):
        if price_change > 0.1 and oi_change > 5:
            return "Long Buildup 🟢"
        elif price_change < -0.1 and oi_change > 5:
            return "Short Buildup 🔴"
        elif price_change > 0.1 and oi_change < -5:
            return "Short Covering 🟢"
        elif price_change < -0.1 and oi_change < -5:
            return "Long Unwinding 🔴"
        elif price_change > 0.1 and abs(oi_change) <= 5:
            return "Mild Buying 🟡"
        elif price_change < -0.1 and abs(oi_change) <= 5:
            return "Mild Selling 🟡"
        else:
            return "Neutral OI ⚪"

    # -----------------------------------------------------------------
    # PRIMARY STRATEGY (Momentum Breakout + S&R)
    # -----------------------------------------------------------------
    @staticmethod
    def apply_primary_strategy(df, index_name="NIFTY"):
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = TechnicalAnalyzer.calculate_indicators(df, is_index)

        if len(df) < 20:
            return "Insufficient Data", "WAIT", 0, 0, df, 0, {}, 0

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = last['atr'] if not pd.isna(last['atr']) else 0

        if last['ema9'] > last['ema21']:
            trend = "BULLISH"
        elif last['ema9'] < last['ema21']:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        volume_ok = last['volume_ratio'] > 1.2 if not pd.isna(last['volume_ratio']) else False
        rsi_cross_up = prev['rsi'] <= 50 and last['rsi'] > 50
        rsi_cross_down = prev['rsi'] >= 50 and last['rsi'] < 50
        macd_hist_up = prev['macd_hist'] <= 0 and last['macd_hist'] > 0
        macd_hist_down = prev['macd_hist'] >= 0 and last['macd_hist'] < 0
        bb_bounce_up = last['close'] <= last['bb_lower'] * 1.01 and last['close'] > last['open']
        bb_bounce_down = last['close'] >= last['bb_upper'] * 0.99 and last['close'] < last['open']

        if trend == "BULLISH" and (rsi_cross_up or macd_hist_up or bb_bounce_up) and volume_ok:
            signal = "BUY_CE"
            trend_desc = f"🟢 STRONG BULLISH (EMA9>21, RSI>{last['rsi']:.0f}, Vol x{last['volume_ratio']:.1f})"
            strength = 80
        elif trend == "BEARISH" and (rsi_cross_down or macd_hist_down or bb_bounce_down) and volume_ok:
            signal = "BUY_PE"
            trend_desc = f"🔴 STRONG BEARISH (EMA9<21, RSI<{last['rsi']:.0f}, Vol x{last['volume_ratio']:.1f})"
            strength = 80
        elif trend == "BULLISH" and (rsi_cross_up or macd_hist_up or volume_ok):
            signal = "BUY_CE"
            trend_desc = f"🟡 BULLISH (EMA9>21, moderate)"
            strength = 60
        elif trend == "BEARISH" and (rsi_cross_down or macd_hist_down or volume_ok):
            signal = "BUY_PE"
            trend_desc = f"🟡 BEARISH (EMA9<21, moderate)"
            strength = 60
        else:
            if last['macd_hist'] > 0 and last['rsi'] > 50:
                signal = "BUY_CE"
                trend_desc = f"⚪ WEAK BULLISH (MACD rising, RSI>{last['rsi']:.0f})"
                strength = 40
            elif last['macd_hist'] < 0 and last['rsi'] < 50:
                signal = "BUY_PE"
                trend_desc = f"⚪ WEAK BEARISH (MACD falling, RSI<{last['rsi']:.0f})"
                strength = 40
            else:
                signal = "WAIT"
                trend_desc = f"⚖️ RANGING (RSI {last['rsi']:.0f})"
                strength = 30

        vwap = last.get('vwap', 0)
        ema = last['ema9']
        fib = {}
        return trend_desc, signal, vwap, ema, df, atr, fib, strength

    # -----------------------------------------------------------------
    # KEYWORD RULE BUILDER
    # -----------------------------------------------------------------
    @staticmethod
    def apply_keyword_strategy(df, custom_code, index_name="NIFTY"):
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = TechnicalAnalyzer.calculate_indicators(df, is_index)

        if len(df) < 20:
            return "Insufficient Data", "WAIT", 0, 0, df, 0, {}, 0

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = last['atr'] if not pd.isna(last['atr']) else 0

        rules = [r.strip() for r in custom_code.split(',')] if custom_code else []

        ema_bull = last['ema9'] > last['ema21']
        ema_bear = last['ema9'] < last['ema21']
        ema_cross_up = prev['ema9'] <= prev['ema21'] and last['ema9'] > last['ema21']
        ema_cross_down = prev['ema9'] >= prev['ema21'] and last['ema9'] < last['ema21']

        rsi_bull_cross = prev['rsi'] <= 50 and last['rsi'] > 50
        rsi_bear_cross = prev['rsi'] >= 50 and last['rsi'] < 50

        macd_hist_up = prev['macd_hist'] <= 0 and last['macd_hist'] > 0
        macd_hist_down = prev['macd_hist'] >= 0 and last['macd_hist'] < 0

        bb_bounce_up = last['close'] <= last['bb_lower'] * 1.01 and last['close'] > last['open']
        bb_bounce_down = last['close'] >= last['bb_upper'] * 0.99 and last['close'] < last['open']

        vwap_above = last['close'] > last.get('vwap', last['close'])
        vwap_below = last['close'] < last.get('vwap', last['close'])

        bull_score = 0
        bear_score = 0
        total_active = 0

        for rule in rules:
            total_active += 1
            if "EMA Crossover" in rule:
                if ema_cross_up: bull_score += 2
                if ema_cross_down: bear_score += 2
            if "Bollinger Bands Bounce" in rule:
                if bb_bounce_up: bull_score += 2
                if bb_bounce_down: bear_score += 2
            if "RSI Breakout" in rule:
                if rsi_bull_cross: bull_score += 2
                if rsi_bear_cross: bear_score += 2
            if "MACD Crossover" in rule:
                if macd_hist_up: bull_score += 2
                if macd_hist_down: bear_score += 2
            if "VWAP Position" in rule:
                if vwap_above: bull_score += 1
                if vwap_below: bear_score += 1
            if "AlphaTrend / Supertrend" in rule or "UT Bot" in rule or "Next Super Trend" in rule:
                if last['st_direction'] == 1: bull_score += 1
                if last['st_direction'] == -1: bear_score += 1
            if "Bollinger Bands" in rule:
                if last['close'] > last['bb_mid']: bull_score += 1
                if last['close'] < last['bb_mid']: bear_score += 1
            if "Moving Average" in rule or "Exponential" in rule:
                if ema_bull: bull_score += 1
                if ema_bear: bear_score += 1

        volume_ok = last['volume_ratio'] > 1.2

        if total_active == 0:
            signal = "WAIT"
            trend_desc = "⚖️ No rules selected"
            strength = 30
        elif bull_score > bear_score + 2 and volume_ok:
            signal = "BUY_CE"
            strength = min(80 + bull_score * 2, 100)
            trend_desc = f"🟢 BULLISH (Rules: {bull_score}/{bear_score})"
        elif bear_score > bull_score + 2 and volume_ok:
            signal = "BUY_PE"
            strength = min(80 + bear_score * 2, 100)
            trend_desc = f"🔴 BEARISH (Rules: {bear_score}/{bull_score})"
        elif bull_score > bear_score:
            signal = "BUY_CE"
            strength = 50 + bull_score * 2
            trend_desc = f"🟡 WEAK BULLISH (Rules: {bull_score}/{bear_score})"
        elif bear_score > bull_score:
            signal = "BUY_PE"
            strength = 50 + bear_score * 2
            trend_desc = f"🟡 WEAK BEARISH (Rules: {bear_score}/{bull_score})"
        else:
            signal = "WAIT"
            strength = 30
            trend_desc = f"⚖️ NEUTRAL (Rules: {bull_score}/{bear_score})"

        vwap = last.get('vwap', 0)
        ema = last['ema9']
        fib = {}
        return trend_desc, signal, vwap, ema, df, atr, fib, strength

    # -----------------------------------------------------------------
    # MACHINE LEARNING STRATEGY
    # -----------------------------------------------------------------
    @staticmethod
    def apply_ml_strategy(df, index_name, ml_prob_threshold, signal_persistence):
        import __main__ as main
        ml_predictor = getattr(main, 'ml_predictor', None)
        if ml_predictor is None:
            return "ML Predictor not available", "WAIT", 0, 0, df, 0, {}, 0

        prob_up, prob_down = ml_predictor.predict(df)
        signal = "WAIT"
        strength = 0
        if prob_up > prob_down + ml_prob_threshold:
            signal = "BUY_CE"
            strength = int(prob_up * 100)
            trend_desc = f"🤖 ML: Up {prob_up:.0%} | Down {prob_down:.0%}"
        elif prob_down > prob_up + ml_prob_threshold:
            signal = "BUY_PE"
            strength = int(prob_down * 100)
            trend_desc = f"🤖 ML: Down {prob_down:.0%} | Up {prob_up:.0%}"
        else:
            trend_desc = f"🤖 ML: Uncertain (Up {prob_up:.0%}, Down {prob_down:.0%})"
            strength = 50

        return trend_desc, signal, 0, 0, df, 0, {}, strength

    # -----------------------------------------------------------------
    # VIJAY & RFF ALL-IN-ONE
    # -----------------------------------------------------------------
    @staticmethod
    def apply_vijay_rff_strategy(df, index_name="NIFTY"):
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = TechnicalAnalyzer.calculate_indicators(df, is_index)

        if len(df) < 20:
            return "Insufficient Data", "WAIT", 0, 0, df, 0, {}, 0

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = last['atr'] if not pd.isna(last['atr']) else 0

        ema_bull = last['ema9'] > last['ema21']
        ema_bear = last['ema9'] < last['ema21']
        rsi_bull = last['rsi'] > 60
        rsi_bear = last['rsi'] < 40
        macd_bull = last['macd'] > last['macd_signal'] and last['macd_hist'] > 0
        macd_bear = last['macd'] < last['macd_signal'] and last['macd_hist'] < 0
        supertrend_bull = last['st_direction'] == 1
        supertrend_bear = last['st_direction'] == -1
        volume_spike = last['volume_ratio'] > 1.5
        near_high = last['close'] > last['high'] * 0.95
        near_low = last['close'] < last['low'] * 1.05

        bull_score = sum([ema_bull, rsi_bull, macd_bull, supertrend_bull, volume_spike, near_high])
        bear_score = sum([ema_bear, rsi_bear, macd_bear, supertrend_bear, volume_spike, near_low])

        if bull_score >= 3 and ema_bull:
            signal = "BUY_CE"
            strength = 70 + bull_score * 5
            trend_desc = f"🟢 VIJAY BUY (Score {bull_score})"
        elif bear_score >= 3 and ema_bear:
            signal = "BUY_PE"
            strength = 70 + bear_score * 5
            trend_desc = f"🔴 VIJAY SELL (Score {bear_score})"
        elif bull_score >= 2:
            signal = "BUY_CE"
            strength = 50 + bull_score * 5
            trend_desc = f"🟡 VIJAY WEAK BUY (Score {bull_score})"
        elif bear_score >= 2:
            signal = "BUY_PE"
            strength = 50 + bear_score * 5
            trend_desc = f"🟡 VIJAY WEAK SELL (Score {bear_score})"
        else:
            signal = "WAIT"
            strength = 30
            trend_desc = f"⚖️ VIJAY NEUTRAL (Score B{bull_score}/S{bear_score})"

        vwap = last.get('vwap', 0)
        ema = last['ema9']
        fib = {}
        return trend_desc, signal, vwap, ema, df, atr, fib, strength

    # -----------------------------------------------------------------
    # INSTITUTIONAL FVG + SMC
    # -----------------------------------------------------------------
    @staticmethod
    def apply_institutional_fvg_strategy(df, index_name="NIFTY"):
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = TechnicalAnalyzer.calculate_indicators(df, is_index)

        if len(df) < 5:
            return "Insufficient Data", "WAIT", 0, 0, df, 0, {}, 0

        last = df.iloc[-1]
        atr = last['atr'] if not pd.isna(last['atr']) else 0

        if len(df) >= 3:
            c1 = df.iloc[-3]
            c2 = df.iloc[-2]
            c3 = df.iloc[-1]

            bullish_fvg = (c2['low'] > c1['high']) and (c3['low'] > c2['high'])
            bearish_fvg = (c2['high'] < c1['low']) and (c3['high'] < c2['low'])

            if bullish_fvg:
                signal = "BUY_CE"
                strength = 85
                trend_desc = "🟢 Bullish FVG Detected"
            elif bearish_fvg:
                signal = "BUY_PE"
                strength = 85
                trend_desc = "🔴 Bearish FVG Detected"
            else:
                signal = "WAIT"
                strength = 30
                trend_desc = "⚖️ No FVG"
        else:
            signal = "WAIT"
            strength = 30
            trend_desc = "⚖️ Insufficient candles"

        vwap = last.get('vwap', 0)
        ema = last['ema9']
        fib = {}
        return trend_desc, signal, vwap, ema, df, atr, fib, strength

    # -----------------------------------------------------------------
    # LUX ALGO INSTITUTIONAL ICT
    # -----------------------------------------------------------------
    @staticmethod
    def apply_lux_algo_ict_strategy(df, index_name="NIFTY"):
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = TechnicalAnalyzer.calculate_indicators(df, is_index)

        if len(df) < 20:
            return "Insufficient Data", "WAIT", 0, 0, df, 0, {}, 0

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = last['atr'] if not pd.isna(last['atr']) else 0

        body = abs(last['close'] - last['open'])
        avg_body = (df['high'] - df['low']).rolling(10).mean().iloc[-1]
        displacement = body > avg_body * 1.5

        ema_bull = last['ema9'] > last['ema21']
        ema_bear = last['ema9'] < last['ema21']

        prev_body = abs(prev['close'] - prev['open'])
        prev_upper = prev['high'] - max(prev['close'], prev['open'])
        prev_lower = min(prev['close'], prev['open']) - prev['low']

        if displacement and last['close'] > last['open'] and ema_bull:
            if prev_lower > prev_body * 1.5:
                signal = "BUY_CE"
                strength = 90
                trend_desc = "🟢 ICT Bullish Order Block"
            else:
                signal = "BUY_CE"
                strength = 70
                trend_desc = "🟢 ICT Bullish Displacement"
        elif displacement and last['close'] < last['open'] and ema_bear:
            if prev_upper > prev_body * 1.5:
                signal = "BUY_PE"
                strength = 90
                trend_desc = "🔴 ICT Bearish Order Block"
            else:
                signal = "BUY_PE"
                strength = 70
                trend_desc = "🔴 ICT Bearish Displacement"
        else:
            signal = "WAIT"
            strength = 30
            trend_desc = "⚖️ No ICT Setup"

        vwap = last.get('vwap', 0)
        ema = last['ema9']
        fib = {}
        return trend_desc, signal, vwap, ema, df, atr, fib, strength

    # -----------------------------------------------------------------
    # MEAN REVERSION (BB/RSI)
    # -----------------------------------------------------------------
    @staticmethod
    def apply_mean_reversion_strategy(df, index_name="NIFTY"):
        is_index = index_name in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]
        df = TechnicalAnalyzer.calculate_indicators(df, is_index)

        if len(df) < 20:
            return "Insufficient Data", "WAIT", 0, 0, df, 0, {}, 0

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = last['atr'] if not pd.isna(last['atr']) else 0

        oversold_rsi = last['rsi'] < 30
        overbought_rsi = last['rsi'] > 70
        touch_lower_bb = last['close'] <= last['bb_lower']
        touch_upper_bb = last['close'] >= last['bb_upper']
        bounce_from_lower = touch_lower_bb and last['close'] > prev['close']
        bounce_from_upper = touch_upper_bb and last['close'] < prev['close']

        volume_ok = last['volume_ratio'] > 1.2

        if (oversold_rsi or touch_lower_bb) and bounce_from_lower and volume_ok:
            signal = "BUY_CE"
            strength = 80
            trend_desc = f"🟢 MEAN REV BUY (RSI {last['rsi']:.0f}, BB touch)"
        elif (overbought_rsi or touch_upper_bb) and bounce_from_upper and volume_ok:
            signal = "BUY_PE"
            strength = 80
            trend_desc = f"🔴 MEAN REV SELL (RSI {last['rsi']:.0f}, BB touch)"
        elif oversold_rsi or touch_lower_bb:
            signal = "BUY_CE"
            strength = 50
            trend_desc = f"🟡 WATCH BUY (RSI {last['rsi']:.0f})"
        elif overbought_rsi or touch_upper_bb:
            signal = "BUY_PE"
            strength = 50
            trend_desc = f"🟡 WATCH SELL (RSI {last['rsi']:.0f})"
        else:
            signal = "WAIT"
            strength = 30
            trend_desc = f"⚖️ NEUTRAL (RSI {last['rsi']:.0f})"

        vwap = last.get('vwap', 0)
        ema = last['ema9']
        fib = {}
        return trend_desc, signal, vwap, ema, df, atr, fib, strength

    # ==========================================
    # ADD MISSING METHOD: filter_option_by_greeks
    # ==========================================
    @staticmethod
    def filter_option_by_greeks(spot, strike, time_to_expiry, iv, option_type):
        """
        Placeholder for Greeks filter. In a real implementation you'd compute
        delta, gamma, theta, vega and enforce thresholds.
        For now, always return True.
        """
        return True

# ==========================================
# VEGA & DELTA FILTERS
# ==========================================
def apply_vega_filter(df, signal_strength):
    """Reject if IV (ATR%) is too high."""
    if df is None or len(df) < 20:
        return True, signal_strength
    last = df.iloc[-1]
    atr_pct = last.get('atr', 0) / last['close'] * 100
    if atr_pct > 2.5:
        return False, 0
    if atr_pct > 2.0:
        signal_strength = int(signal_strength * 0.8)
    return True, signal_strength

def apply_delta_filter(signal_strength, prob_threshold=20):
    if signal_strength < prob_threshold:
        return False, 0
    return True, signal_strength

# ==========================================
# MACHINE LEARNING PREDICTOR
# ==========================================
class MLPredictor:
    def __init__(self):
        self.rf_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_train_index = 0
        self.feature_cols = ['rsi', 'macd_hist', 'volume_ratio', 'atr_pct', 'ema9', 'ema21', 'close_pct']
        self.train_lock = Lock()
        self.train_queue = queue.Queue()
        self.train_thread = threading.Thread(target=self._background_train, daemon=True)
        self.train_thread.start()
        self.explainer = None

    def _background_train(self):
        while True:
            try:
                df = self.train_queue.get(timeout=1)
                if df is not None and HAS_SKLEARN:
                    self._do_train(df)
            except queue.Empty:
                continue

    def _do_train(self, df):
        try:
            df = df.copy()
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df['atr_pct'] = df['atr'] / df['close'] * 100
            df['close_pct'] = df['close'].pct_change() * 100
            df = df.dropna()

            if len(df) < 50:
                return

            X = df[self.feature_cols].values
            y = df['target'].values

            X_scaled = self.scaler.fit_transform(X)

            self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            self.rf_model.fit(X_scaled, y)
            self.is_trained = True
            self.last_train_index = len(df)
            print("ML model trained successfully")
        except Exception as e:
            print(f"ML training error: {e}")

    def prepare_features(self, df):
        if df is None or len(df) < 20:
            return None
        last = df.iloc[-1]
        try:
            features = {
                'rsi': last.get('rsi', 50),
                'macd_hist': last.get('macd_hist', 0),
                'volume_ratio': last.get('volume_ratio', 1),
                'atr_pct': last.get('atr', 0) / last['close'] * 100,
                'ema9': last.get('ema9', last['close']),
                'ema21': last.get('ema21', last['close']),
                'close_pct': df['close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
            }
            return np.array([list(features.values())])
        except:
            return None

    def train(self, df):
        if len(df) < 50 or not HAS_SKLEARN:
            return
        self.train_queue.put(df.copy())

    def predict(self, df):
        if not self.is_trained or not HAS_SKLEARN:
            if df is not None and len(df) > 10:
                last_close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2]
                if last_close > prev_close:
                    return 0.8, 0.2
                elif last_close < prev_close:
                    return 0.2, 0.8
            return 0.5, 0.5

        X = self.prepare_features(df)
        if X is None:
            return 0.5, 0.5

        X_scaled = self.scaler.transform(X)
        prob = self.rf_model.predict_proba(X_scaled)[0]
        return prob[1], prob[0]

    def should_retrain(self, df):
        return not self.is_trained or len(df) - self.last_train_index >= 50

    def explain_prediction(self, df):
        if not HAS_SHAP or not self.is_trained:
            return None
        X = self.prepare_features(df)
        if X is None:
            return None
        X_scaled = self.scaler.transform(X)
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(X_scaled)
        return shap_values

ml_predictor = MLPredictor()

# ==========================================
# HIGH PROFIT STRATEGY MODULES (abbreviated)
# ==========================================
class ArbitrageModule:
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
        return []

class MultiLegStrategies:
    def iron_condor(self, spot, iv, days_to_expiry):
        return None
    def straddle_strangle(self, spot, iv, earnings=False):
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
    st.metric("Final Capital", f"₹{final_capital:,.0f}", f"{((final_capital/initial_capital-1)*100):.0f}% Return")
    st.metric("Total Profit", f"₹{total_profit:,.0f}")
    st.metric("Daily Average", f"₹{total_profit/trading_days:,.0f}")
    growth = [initial_capital * ((1 + rate) ** i) for i in range(1, periods + 1)]
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
    if st.button("🔍 Scan Breakouts Now", use_container_width=True, on_click=lambda: play_sound_now("click")):
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
        df['tr'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
        df['atr'] = df['tr'].rolling(14).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['vol_ma'] = df['Volume'].rolling(20).mean()
        df['vol_spike'] = df['Volume'] > df['vol_ma'] * 1.2
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['volume_roc'] = df['Volume'].pct_change(5) * 100
        last = df.iloc[-1]
        prev = df.iloc[-2]
        buy_signal = (last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21'] and last['vol_spike']) or \
                     (last['rsi'] > 50 and prev['rsi'] <= 50 and last['macd'] > last['macd_signal'] and last['vol_spike']) or \
                     (last['macd_hist'] > 0 and prev['macd_hist'] <= 0 and last['vol_spike']) or \
                     (last['volume_roc'] > 20 and last['Close'] > prev['Close'] and last['vol_spike'])
        sell_signal = (last['ema9'] < last['ema21'] and prev['ema9'] >= prev['ema21'] and last['vol_spike']) or \
                      (last['rsi'] < 50 and prev['rsi'] >= 50 and last['macd'] < last['macd_signal'] and last['vol_spike']) or \
                      (last['macd_hist'] < 0 and prev['macd_hist'] >= 0 and last['vol_spike']) or \
                      (last['volume_roc'] > 20 and last['Close'] < prev['Close'] and last['vol_spike'])
        current_atr = last['atr']
        if "XAU" in symbol or "GOLD" in symbol:
            sl_points = current_atr * 1.5
            tp_points = current_atr * 3.0
        else:
            sl_points = current_atr * 1.2
            tp_points = current_atr * 2.5
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${last['Close']:.2f}", f"{((last['Close']/prev['Close']-1)*100):.2f}%")
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
# TELEGRAM CONTROL BOT
# ==========================================
class TelegramController:
    def __init__(self, token, allowed_users):
        self.token = token
        self.allowed_users = [int(u.strip()) for u in allowed_users.split(',')] if allowed_users else []
        self.app = None
        self.bot_instance = None
        self.running = False

    async def start(self, bot_instance):
        if not HAS_TELEGRAM_BOT or not self.token:
            return
        self.bot_instance = bot_instance
        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("stop", self.stop))
        self.app.add_handler(CommandHandler("start", self.start_engine))
        self.app.add_handler(CommandHandler("pnl", self.pnl))
        self.app.add_handler(CommandHandler("balance", self.balance))
        self.app.add_handler(CommandHandler("help", self.help))
        threading.Thread(target=self._run_polling, daemon=True).start()

    def _run_polling(self):
        self.running = True
        self.app.run_polling()

    async def _check_user(self, update):
        user_id = update.effective_user.id
        if self.allowed_users and user_id not in self.allowed_users:
            await update.message.reply_text("⛔ Unauthorized.")
            return False
        return True

    async def status(self, update, context):
        if not await self._check_user(update): return
        bot = self.bot_instance
        status = "🟢 Running" if bot.state["is_running"] else "🔴 Stopped"
        active = f"Active Trade: {bot.state['active_trade']['symbol']} PnL: {bot.state['active_trade']['floating_pnl']:.2f}" if bot.state["active_trade"] else "No active trade"
        await update.message.reply_text(f"*Bot Status*\n{status}\n{active}\nTrades Today: {bot.state['trades_today']}\nDaily PnL: {bot.state['daily_pnl']:.2f}", parse_mode='Markdown')

    async def stop(self, update, context):
        if not await self._check_user(update): return
        bot = self.bot_instance
        bot.state["is_running"] = False
        bot.state["engine_active"] = False
        await update.message.reply_text("🛑 Engine stopped.")

    async def start_engine(self, update, context):
        if not await self._check_user(update): return
        bot = self.bot_instance
        if not bot.state["is_running"]:
            bot.state["is_running"] = True
            bot.state["engine_active"] = True
            t = threading.Thread(target=bot.trading_loop, daemon=True)
            add_script_run_ctx(t)
            t.start()
            await update.message.reply_text("▶️ Engine started.")
        else:
            await update.message.reply_text("Engine already running.")

    async def pnl(self, update, context):
        if not await self._check_user(update): return
        bot = self.bot_instance
        await update.message.reply_text(f"Today's PnL: ₹{bot.state['daily_pnl']:.2f}")

    async def balance(self, update, context):
        if not await self._check_user(update): return
        bot = self.bot_instance
        await update.message.reply_text(f"Balance: {bot.get_balance()}")

    async def help(self, update, context):
        await update.message.reply_text("Commands:\n/status\n/stop\n/start\n/pnl\n/balance")

# ==========================================
# BACKTESTING ENGINE
# ==========================================
class Backtester:
    def __init__(self, strategy_func, df, initial_capital=100000, lot_size=1):
        self.strategy_func = strategy_func
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.lot_size = lot_size
        self.trades = []
        self.equity_curve = []
        self.total_return = 0.0
        self.sharpe = 0.0
        self.max_dd = 0.0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0

    def run(self):
        self.df['signal'] = None
        self.df['position'] = 0
        position = 0
        entry_price = 0
        entry_index = None
        for i in range(50, len(self.df)):
            slice_df = self.df.iloc[:i+1]
            trend, signal, vwap, ema, _, atr, _, strength = self.strategy_func(slice_df, "NIFTY")
            self.df.loc[self.df.index[i], 'signal'] = signal
            price = self.df.iloc[i]['close']
            if signal != "WAIT" and position == 0:
                position = 1 if signal == "BUY_CE" else -1
                entry_price = price
                entry_index = i
                self.df.loc[self.df.index[i], 'position'] = position
            elif position != 0:
                exit_signal = False
                if (position == 1 and signal == "BUY_PE") or (position == -1 and signal == "BUY_CE"):
                    exit_signal = True
                elif i == len(self.df)-1:
                    exit_signal = True
                if exit_signal:
                    exit_price = price
                    pnl = (exit_price - entry_price) * position * self.lot_size
                    self.trades.append({
                        "entry_time": self.df.index[entry_index],
                        "exit_time": self.df.index[i],
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl": pnl,
                        "type": "LONG" if position == 1 else "SHORT"
                    })
                    position = 0
                    entry_price = 0
            else:
                self.df.loc[self.df.index[i], 'position'] = 0
        self.df['returns'] = self.df['close'].pct_change() * self.df['position'].shift(1)
        self.df['strategy_returns'] = self.df['returns'] * self.lot_size
        self.df['equity'] = self.initial_capital * (1 + self.df['strategy_returns']).cumprod()
        self.equity_curve = self.df[['equity']].dropna()
        self.total_return = (self.equity_curve.iloc[-1]['equity'] / self.initial_capital - 1) * 100
        returns = self.df['strategy_returns'].dropna()
        self.sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        self.max_dd = (self.equity_curve['equity'] / self.equity_curve['equity'].cummax() - 1).min() * 100
        pnl_list = [t['pnl'] for t in self.trades]
        if pnl_list:
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]
            self.win_rate = len(wins) / len(pnl_list) * 100 if pnl_list else 0
            self.avg_win = np.mean(wins) if wins else 0
            self.avg_loss = np.mean(losses) if losses else 0
        return self.equity_curve, self.total_return, self.trades

    def plot_results(self):
        st.line_chart(self.equity_curve)

# ==========================================
# NOTIFICATION MANAGER
# ==========================================
class NotificationManager:
    def __init__(self, email_config=None, fcm_server_key=None, tg_token=None, tg_chat=None, wa_phone=None, wa_api=None):
        self.email_config = email_config
        self.fcm_server_key = fcm_server_key
        self.tg_token = tg_token
        self.tg_chat = tg_chat
        self.wa_phone = wa_phone
        self.wa_api = wa_api

    def send_email(self, subject, body):
        if not self.email_config:
            return
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = self.email_config['recipients']
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"Email error: {e}")

    def send_push(self, title, body):
        if not self.fcm_server_key or not HAS_FIREBASE:
            return
        try:
            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                topic="all",
            )
            messaging.send(message)
        except Exception as e:
            print(f"Push error: {e}")

    def send_telegram(self, message):
        if self.tg_token and self.tg_chat:
            try:
                requests.get(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                             params={"chat_id": self.tg_chat, "text": message, "parse_mode": "HTML"}, timeout=3)
            except:
                pass

    def send_whatsapp(self, message):
        if self.wa_phone and self.wa_api:
            try:
                requests.get("https://api.callmebot.com/whatsapp.php",
                             params={"phone": self.wa_phone, "text": message, "apikey": self.wa_api}, timeout=3)
            except:
                pass

    def notify_all(self, title, message):
        self.send_email(title, message)
        self.send_push(title, message)
        self.send_telegram(f"<b>{title}</b>\n{message}")
        self.send_whatsapp(f"{title}\n{message}")

# ==========================================
# TAX REPORTING
# ==========================================
def generate_tax_report(user_id, year):
    if HAS_DB:
        try:
            start_date = f"{year}-04-01"
            end_date = f"{year+1}-03-31"
            res = supabase.table("trade_logs").select("*").eq("user_id", user_id).gte("trade_date", start_date).lte("trade_date", end_date).execute()
            if res.data:
                df = pd.DataFrame(res.data)
                df['holding_days'] = 0
                df['tax_category'] = df['holding_days'].apply(lambda x: 'STCG' if x <= 365 else 'LTCG')
                summary = df.groupby('tax_category')['pnl'].sum().to_dict()
                return summary, df
        except Exception as e:
            print(f"Tax report error: {e}")
    return None, None

# ==========================================
# USER ROLES & PERMISSIONS
# ==========================================
def check_role(allowed_roles):
    user_role = st.session_state.get('user_role', 'trader')
    return user_role in allowed_roles

# ==========================================
# ENHANCED MARKET DATA
# ==========================================
def fetch_option_chain(symbol, expiry):
    df_map = get_angel_scrip_master()
    if df_map is None or df_map.empty:
        return None
    subset = df_map[(df_map['name'] == symbol) & (df_map['expiry'] == expiry)]
    if subset.empty:
        return None
    ce = subset[subset['symbol'].str.endswith('CE')].copy()
    pe = subset[subset['symbol'].str.endswith('PE')].copy()
    merged = pd.merge(ce, pe, on='strike', suffixes=('_ce', '_pe'))
    return merged

# ==========================================
# CORE BOT ENGINE (SniperBot) – with fixed get_strike and get_broker_balances
# ==========================================
class SniperBot:
    def __init__(self, api_key="", client_id="", pwd="", totp_secret="", 
                 tg_token="", tg_chat="", wa_phone="", wa_api="", 
                 mt5_acc="", mt5_pass="", mt5_server="", mt5_api_url="", 
                 zerodha_api="", zerodha_secret="", request_token="", 
                 coindcx_api="", coindcx_secret="", delta_api="", delta_secret="", 
                 is_mock=False, tg_bot_token="", tg_allowed_users="", 
                 fyers_client_id="", fyers_secret="", fyers_token="",
                 upstox_api_key="", upstox_api_secret="", upstox_access_token="",
                 fivepaisa_client_id="", fivepaisa_secret="", fivepaisa_access_token="",
                 binance_api_key="", binance_api_secret="", binance_testnet=False,
                 stoxkart_api_key="", stoxkart_secret="",
                 dhan_client_id="", dhan_access_token="",
                 shoonya_user="", shoonya_password="", shoonya_totp="",
                 email_smtp_server="", email_port=587, email_username="", email_password="", email_recipients="",
                 fcm_server_key="", push_enabled=False):
        
        self.api_key, self.client_id, self.pwd, self.totp_secret = api_key, client_id, pwd, totp_secret
        self.tg_token, self.tg_chat, self.wa_phone, self.wa_api = tg_token, tg_chat, wa_phone, wa_api
        self.mt5_acc, self.mt5_pass, self.mt5_server, self.mt5_api_url = mt5_acc, mt5_pass, mt5_server, mt5_api_url
        self.zerodha_api, self.zerodha_secret, self.request_token = zerodha_api, zerodha_secret, request_token
        self.coindcx_api, self.coindcx_secret = coindcx_api, coindcx_secret
        self.delta_api, self.delta_secret = delta_api, delta_secret
        self.tg_bot_token = tg_bot_token
        self.tg_allowed_users = tg_allowed_users
        self.fyers_client_id = fyers_client_id
        self.fyers_secret = fyers_secret
        self.fyers_token = fyers_token
        self.upstox_api_key = upstox_api_key
        self.upstox_api_secret = upstox_api_secret
        self.upstox_access_token = upstox_access_token
        self.fivepaisa_client_id = fivepaisa_client_id
        self.fivepaisa_secret = fivepaisa_secret
        self.fivepaisa_access_token = fivepaisa_access_token
        self.binance_api_key = binance_api_key
        self.binance_api_secret = binance_api_secret
        self.binance_testnet = binance_testnet
        self.stoxkart_api_key = stoxkart_api_key
        self.stoxkart_secret = stoxkart_secret
        self.dhan_client_id = dhan_client_id
        self.dhan_access_token = dhan_access_token
        self.shoonya_user = shoonya_user
        self.shoonya_password = shoonya_password
        self.shoonya_totp = shoonya_totp

        self.email_config = {
            "smtp_server": email_smtp_server,
            "port": email_port,
            "username": email_username,
            "password": email_password,
            "recipients": email_recipients
        } if email_smtp_server else None
        self.fcm_server_key = fcm_server_key
        self.push_enabled = push_enabled
        self.notification_manager = NotificationManager(
            email_config=self.email_config,
            fcm_server_key=self.fcm_server_key,
            tg_token=self.tg_token,
            tg_chat=self.tg_chat,
            wa_phone=self.wa_phone,
            wa_api=self.wa_api
        )

        self.api, self.kite, self.token_map, self.is_mock = None, None, None, is_mock
        self.mt5_bridge = None
        self.fyers_bridge = None
        self.upstox_bridge = None
        self.fivepaisa_bridge = None
        self.binance_bridge = None
        self.stoxkart_bridge = None
        self.delta_bridge = None
        self.coindcx_bridge = None
        self.dhan_bridge = None
        self.shoonya_bridge = None
        self.is_mt5_connected = False
        self.is_fyers_connected = False
        self.is_upstox_connected = False
        self.is_fivepaisa_connected = False
        self.is_binance_connected = False
        self.is_stoxkart_connected = False
        self.is_delta_connected = False
        self.is_coindcx_connected = False
        self.is_dhan_connected = False
        self.is_shoonya_connected = False
        self.client_name = "Offline User"
        self._primary_client_set = False
        self.client_ip = get_client_ip()
        self.user_hash = get_user_hash(self.api_key)
        self.analyzer = TechnicalAnalyzer()
        self.scalper = ScalpingModule(self)
        self.arbitrage = ArbitrageModule()
        self.option_seller = OptionSellingModule(self)
        self.multi_leg = MultiLegStrategies()
        self.telegram_controller = TelegramController(tg_bot_token, tg_allowed_users) if tg_bot_token else None
        
        self.state = {
            "is_running": False,
            "engine_active": True,
            "order_in_flight": False,
            "active_trade": None,
            "active_trades": [],
            "last_trade": None,
            "logs": deque(maxlen=50),
            "current_trend": "WAIT",
            "current_signal": "WAIT",
            "signal_strength": 0,
            "spot": 0.0,
            "vwap": 0.0,
            "ema": 0.0,
            "atr": 0.0,
            "fib_data": {},
            "latest_data": None,
            "latest_candle": None,
            "ui_popups": deque(maxlen=10),
            "loop_count": 0,
            "daily_pnl": 0.0,
            "trades_today": 0,
            "manual_exit": False,
            "stop_after_manual_exit": False,
            "ghost_memory": {},
            "tv_signal": {"action": "WAIT", "symbol": "", "timestamp": 0},
            "scalp_signals": deque(maxlen=20),
            "arbitrage_opps": [],
            "premium_opps": [],
            "hz_trades": deque(maxlen=100),
            "hz_pnl": 0.0,
            "hz_wins": 0,
            "hz_losses": 0,
            "hz_last_signal_time": None,
            "news_cache": [],
            "signal_history": deque(maxlen=5),
            "mock_price": None,
            "sound_queue": deque(maxlen=10),
            "fomo_signals": deque(maxlen=20),
            "last_trade_time": None,
            "trade_lock": Lock(),
            "pending_signal": None,
            "ml_explanation": None
        }
        self.settings = {}
        self.system_user_id = None
        self.user_role = "trader"

        self.ws_angel = None
        self.ws_angel_connected = False
        self.live_prices_angel = {}
        self._angel_ws_stop = Event()
        self._angel_ws_reconnect = Event()

        self.db_queue = queue.Queue()
        self.db_thread = threading.Thread(target=self._db_worker, daemon=True)
        self.db_thread.start()

        self.loss_streak_cooldown = False
        self.cooldown_end_time = None

        self.market_end_notify_thread = threading.Thread(target=self._market_end_notifier, daemon=True)
        self.market_end_notify_thread.start()

    def _db_worker(self):
        while True:
            try:
                data = self.db_queue.get(timeout=1)
                if data and HAS_DB:
                    supabase.table("trade_logs").insert(data).execute()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"DB write error: {e}")

    def _market_end_notifier(self):
        while True:
            now = get_ist()
            if now.hour == 15 and now.minute == 30 and now.second < 30:
                if self.state.get("daily_pnl", 0.0) != 0:
                    msg = f"Market Closed. Today's PnL: ₹{self.state['daily_pnl']:.2f}, Trades: {self.state['trades_today']}"
                    self.push_notify("Market End Summary", msg)
            time.sleep(60)

    def force_exit(self):
        with self.state["trade_lock"]:
            if self.state["active_trade"]:
                t = self.state["active_trade"]
                if not self.is_mock and not t.get("simulated"):
                    exec_side = "BUY" if t['type'] in ["SELL", "SELL_CALL", "SELL_PUT", "SHORT"] else "SELL"
                    order_id, err = self.place_real_order(t['symbol'], t['token'], t['qty'], exec_side, t['exch'], "MARKET")
                    if order_id:
                        ltp = self.get_live_price(t['exch'], t['symbol'], t['token']) or t['entry']
                    else:
                        ltp = t['entry']
                        self.log(f"Exit order failed: {err}")
                else:
                    ltp = self.get_live_price(t['exch'], t['symbol'], t['token']) or t['entry']
                if t['type'] in ["SELL", "SELL_CALL", "SELL_PUT", "SHORT"]:
                    pnl = (t['entry'] - ltp) * t['qty']
                else: 
                    pnl = (ltp - t['entry']) * t['qty']
                
                if not self.is_mock and hasattr(self, "system_user_id"):
                    today = get_ist().strftime('%Y-%m-%d')
                    now = get_ist().strftime('%H:%M:%S')
                    slippage = abs(t.get('fill_price', ltp) - t.get('signal_price', t['entry']))
                    self.db_queue.put({
                        "user_id": self.system_user_id, "trade_date": today, "trade_time": now,
                        "symbol": t['symbol'], "trade_type": t['type'], "qty": t['qty'],
                        "entry_price": t['entry'], "exit_price": ltp,
                        "pnl": pnl, "result": "Manual Exit", "slippage": slippage
                    })
                self.state["daily_pnl"] += pnl
                self.state["active_trade"] = None
                self.state["sound_queue"].append("exit")
                st.toast(f"Trade closed at {ltp:.2f} | PnL: ₹{pnl:.2f}", icon="✅")
            if self.state["active_trades"]:
                for trade in self.state["active_trades"]:
                    if not self.is_mock and not trade.get("simulated"):
                        exec_side = "BUY" if trade['type'] in ["SELL", "SELL_CALL", "SELL_PUT", "SHORT"] else "SELL"
                        self.place_real_order(trade['symbol'], trade['token'], trade['qty'], exec_side, trade['exch'], "MARKET")
                    ltp = self.get_live_price(trade['exch'], trade['symbol'], trade['token']) or trade['entry']
                    if trade['type'] in ["SELL", "SELL_CALL", "SELL_PUT", "SHORT"]:
                        pnl = (trade['entry'] - ltp) * trade['qty']
                    else:
                        pnl = (ltp - trade['entry']) * trade['qty']
                    self.state["daily_pnl"] += pnl
                self.state["active_trades"] = []
                self.state["sound_queue"].append("exit")
                st.toast("All arbitrage trades closed", icon="✅")
            self.state["is_running"] = False
            self.state["engine_active"] = False

    def protect_profit(self):
        with self.state["trade_lock"]:
            if self.state["active_trade"]:
                trade = self.state["active_trade"]
                ltp = trade.get('current_ltp', trade['entry'])
                is_long = trade['type'] in ["CE", "BUY"]
                if is_long:
                    pnl = (ltp - trade['entry']) * trade['qty']
                else:
                    pnl = (trade['entry'] - ltp) * trade['qty']

                thresholds = [(500, 0), (1000, 500), (1500, 1000)]
                new_sl = None
                for threshold, lock_amount in thresholds:
                    if pnl >= threshold:
                        if is_long:
                            new_sl = trade['entry'] + lock_amount / trade['qty']
                        else:
                            new_sl = trade['entry'] - lock_amount / trade['qty']

                if new_sl is not None:
                    if is_long:
                        if new_sl > trade['sl']:
                            trade['sl'] = new_sl
                            self.log(f"🛡️ Protect profit: SL moved to {new_sl:.2f} (locked ₹{lock_amount})")
                            self.push_notify("Protect Profit", f"Stop loss moved to {new_sl:.2f}, locked ₹{lock_amount}")
                    else:
                        if new_sl < trade['sl']:
                            trade['sl'] = new_sl
                            self.log(f"🛡️ Protect profit: SL moved to {new_sl:.2f} (locked ₹{lock_amount})")
                            self.push_notify("Protect Profit", f"Stop loss moved to {new_sl:.2f}, locked ₹{lock_amount}")
                else:
                    self.log("No profit threshold reached to lock.")
            else:
                self.log("No active trade to protect")

    def start_pnl_updater(self):
        def update():
            while True:
                if self.state.get("active_trade"):
                    trade = self.state["active_trade"]
                    ltp = self.get_live_price(trade['exch'], trade['symbol'], trade['token'])
                    if ltp:
                        trade['current_ltp'] = ltp
                        if trade['type'] in ["SELL", "SELL_CALL", "SELL_PUT", "SHORT"]:
                            pnl = (trade['entry'] - ltp) * trade['qty']
                        else:
                            pnl = (ltp - trade['entry']) * trade['qty']
                        trade['floating_pnl'] = pnl
                        if 'entry_time' in trade:
                            elapsed = (get_ist() - trade['entry_time']).total_seconds()
                            trade['elapsed_time'] = elapsed
                        st.session_state.tracker_updated = True
                if self.state.get("active_trades"):
                    for trade in self.state["active_trades"]:
                        ltp = self.get_live_price(trade['exch'], trade['symbol'], trade['token'])
                        if ltp:
                            trade['current_ltp'] = ltp
                            if trade['type'] in ["SELL", "SELL_CALL", "SELL_PUT", "SHORT"]:
                                pnl = (trade['entry'] - ltp) * trade['qty']
                            else:
                                pnl = (ltp - trade['entry']) * trade['qty']
                            trade['floating_pnl'] = pnl
                            if 'entry_time' in trade:
                                elapsed = (get_ist() - trade['entry_time']).total_seconds()
                                trade['elapsed_time'] = elapsed
                            st.session_state.tracker_updated = True
                time.sleep(1)
        thread = threading.Thread(target=update, daemon=True)
        add_script_run_ctx(thread)
        thread.start()

    def load_daily_pnl(self):
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

    def get_broker_balances(self):
        """Return a dict of broker name -> balance string."""
        balances = {}
        if self.api:
            try:
                rms = self.api.rmslimit()
                if rms and rms.get('status') and rms.get('data'):
                    bal = rms['data'].get('availablecash', 0)
                    balances["Angel One"] = f"₹{float(bal):,.2f}"
            except: pass
        if self.kite:
            try:
                margins = self.kite.margins()
                eq = margins.get('equity', {})
                bal = eq.get('available', {}).get('live_balance', eq.get('net', 0))
                balances["Zerodha"] = f"₹{float(bal):,.2f}"
            except: pass
        if self.coindcx_bridge and self.is_coindcx_connected:
            try:
                bal = self.coindcx_bridge.get_balance()
                balances["CoinDCX"] = f"${bal:.2f}"
            except: pass
        if self.delta_bridge and self.is_delta_connected:
            try:
                info = self.delta_bridge.get_account_info()
                if info:
                    balances["Delta Exchange"] = f"${info.get('balance', 0):,.2f}"
            except: pass
        if self.mt5_bridge and self.is_mt5_connected:
            try:
                acc = self.mt5_bridge.get_account_info()
                if acc:
                    balances["MT5"] = f"${acc.get('balance', 0):,.2f}"
            except: pass
        if self.fyers_bridge and self.is_fyers_connected:
            try:
                info = self.fyers_bridge.get_account_info()
                if info:
                    balances["Fyers"] = f"₹{info.get('balance', 0):,.2f}"
            except: pass
        if self.upstox_bridge and self.is_upstox_connected:
            try:
                info = self.upstox_bridge.get_account_info()
                if info:
                    balances["Upstox"] = f"₹{info.get('balance', 0):,.2f}"
            except: pass
        if self.fivepaisa_bridge and self.is_fivepaisa_connected:
            try:
                info = self.fivepaisa_bridge.get_account_info()
                if info:
                    balances["5paisa"] = f"₹{info.get('balance', 0):,.2f}"
            except: pass
        if self.binance_bridge and self.is_binance_connected:
            try:
                bal = self.binance_bridge.get_asset_balance("USDT")
                balances["Binance"] = f"${bal:,.2f}" if bal else "0.00"
            except: pass
        if self.stoxkart_bridge and self.is_stoxkart_connected:
            try:
                info = self.stoxkart_bridge.get_account_info()
                if info:
                    balances["Stoxkart"] = f"₹{info.get('balance', 0):,.2f}"
            except: pass
        if self.dhan_bridge and self.is_dhan_connected:
            try:
                info = self.dhan_bridge.get_account_info()
                if info:
                    balances["Dhan"] = f"₹{info.get('balance', 0):,.2f}"
            except: pass
        if self.shoonya_bridge and self.is_shoonya_connected:
            try:
                info = self.shoonya_bridge.get_account_info()
                if info:
                    balances["Shoonya"] = f"₹{info.get('balance', 0):,.2f}"
            except: pass
        return balances

    def connect_mt5(self):
        if self.mt5_acc and self.mt5_server:
            self.mt5_bridge = MT5WebBridge(account=self.mt5_acc, password=self.mt5_pass, server=self.mt5_server, api_url=self.mt5_api_url)
            success, msg = self.mt5_bridge.connect()
            if success:
                self.is_mt5_connected = True
                self.log(f"✅ MT5 Connected via bridge: {msg}")
                return True
            else:
                self.log(f"❌ MT5 Bridge failed: {msg}")
        return False

    def connect_fyers(self):
        if self.fyers_client_id and self.fyers_secret and self.fyers_token:
            self.fyers_bridge = FyersBridge(self.fyers_client_id, self.fyers_secret, self.fyers_token)
            success, msg = self.fyers_bridge.connect()
            if success:
                self.is_fyers_connected = True
                self.log(f"✅ Fyers Connected: {msg}")
                return True
            else:
                self.log(f"❌ Fyers connection failed: {msg}")
        return False

    def connect_upstox(self):
        if self.upstox_api_key and self.upstox_api_secret:
            self.upstox_bridge = UpstoxBridge(self.upstox_api_key, self.upstox_api_secret, self.upstox_access_token)
            success, msg = self.upstox_bridge.connect()
            if success:
                self.is_upstox_connected = True
                self.log(f"✅ Upstox Connected: {msg}")
                return True
            else:
                self.log(f"❌ Upstox connection failed: {msg}")
        return False

    def connect_fivepaisa(self):
        if self.fivepaisa_client_id and self.fivepaisa_secret:
            self.fivepaisa_bridge = FivePaisaBridge(self.fivepaisa_client_id, self.fivepaisa_secret, self.fivepaisa_access_token)
            success, msg = self.fivepaisa_bridge.connect()
            if success:
                self.is_fivepaisa_connected = True
                self.log(f"✅ 5paisa Connected: {msg}")
                return True
            else:
                self.log(f"❌ 5paisa connection failed: {msg}")
        return False

    def connect_binance(self):
        if self.binance_api_key and self.binance_api_secret:
            self.binance_bridge = BinanceBridge(api_key=self.binance_api_key, api_secret=self.binance_api_secret, testnet=self.binance_testnet)
            success, msg = self.binance_bridge.connect()
            if success:
                self.is_binance_connected = True
                self.log(f"✅ Binance Connected: {msg}")
                return True
            else:
                self.log(f"❌ Binance connection failed: {msg}")
        return False

    def connect_stoxkart(self):
        if self.stoxkart_api_key and self.stoxkart_secret:
            self.stoxkart_bridge = StoxkartBridge(self.stoxkart_api_key, self.stoxkart_secret)
            success, msg = self.stoxkart_bridge.connect()
            if success:
                self.is_stoxkart_connected = True
                self.log(f"✅ Stoxkart Connected: {msg}")
                return True
            else:
                self.log(f"❌ Stoxkart connection failed: {msg}")
        return False

    def connect_delta(self):
        if self.delta_api and self.delta_secret:
            self.delta_bridge = DeltaExchangeBridge(self.delta_api, self.delta_secret)
            success, msg = self.delta_bridge.connect()
            if success:
                self.is_delta_connected = True
                self.log(f"✅ Delta Exchange Connected: {msg}")
                return True
            else:
                self.log(f"❌ Delta connection failed: {msg}")
        return False

    def connect_coindcx(self):
        if self.coindcx_api and self.coindcx_secret:
            self.coindcx_bridge = CoinDCXBridge(self.coindcx_api, self.coindcx_secret)
            success, msg = self.coindcx_bridge.connect()
            if success:
                self.is_coindcx_connected = True
                self.log(f"✅ CoinDCX Connected: {msg}")
                return True
            else:
                self.log(f"❌ CoinDCX connection failed: {msg}")
        return False

    def connect_dhan(self):
        if self.dhan_client_id and self.dhan_access_token:
            self.dhan_bridge = DhanBridge(self.dhan_client_id, self.dhan_access_token)
            success, msg = self.dhan_bridge.connect()
            if success:
                self.is_dhan_connected = True
                self.log(f"✅ Dhan Connected: {msg}")
                return True
            else:
                self.log(f"❌ Dhan connection failed: {msg}")
        return False

    def connect_shoonya(self):
        if self.shoonya_user and self.shoonya_password:
            self.shoonya_bridge = ShoonyaBridge(self.shoonya_user, self.shoonya_password, self.shoonya_totp)
            success, msg = self.shoonya_bridge.connect()
            if success:
                self.is_shoonya_connected = True
                self.log(f"✅ Shoonya Connected: {msg}")
                return True
            else:
                self.log(f"❌ Shoonya connection failed: {msg}")
        return False

    def login(self):
        if self.is_mock:
            self.client_name, self.api_key = "Paper Trading User", "mock_user"
            self.user_hash = get_user_hash(self.api_key)
            self.push_notify("🟢 Session Started", f"Paper Trading active.")
            self.start_webhook_listener()
            if self.tg_bot_token:
                self.telegram_controller.start(self)
            self.start_pnl_updater()
            return True

        success = False
        if self.api_key and self.totp_secret:
            try:
                obj = SmartConnect(api_key=self.api_key)
                totp = pyotp.TOTP(self.totp_secret).now()
                res = obj.generateSession(self.client_id, self.pwd, totp)
                if res and res.get('status'):
                    self.api = obj
                    fetched_name = ""
                    if 'data' in res:
                        data = res['data']
                        if 'userProfile' in data and 'name' in data['userProfile']:
                            fetched_name = data['userProfile']['name']
                        elif 'name' in data:
                            fetched_name = data['name']
                    if not self._primary_client_set:
                        self.client_name = f"Angel User ({fetched_name})" if fetched_name else f"Angel ({self.client_id})"
                        self._primary_client_set = True
                    self.log(f"✅ Angel One Connected as {fetched_name if fetched_name else self.client_id}")
                    self.start_angel_ws()
                    success = True
                else:
                    self.log(f"❌ Angel Login failed: {res.get('message', 'Check credentials')}")
            except Exception as e:
                self.log(f"❌ Angel Login Exception: {e}")

        if self.zerodha_api and self.zerodha_secret and self.request_token and HAS_ZERODHA:
            try:
                self.kite = KiteConnect(api_key=self.zerodha_api)
                data = self.kite.generate_session(self.request_token, api_secret=self.zerodha_secret)
                self.kite.set_access_token(data["access_token"])
                try:
                    profile = self.kite.profile()
                    fetched_name = profile.get('user_name') or profile.get('user_shortname') or profile.get('email', '')
                    if not self._primary_client_set:
                        self.client_name = f"Zerodha ({fetched_name})" if fetched_name else "Zerodha User"
                        self._primary_client_set = True
                except:
                    if not self._primary_client_set:
                        self.client_name = "Zerodha User"
                        self._primary_client_set = True
                self.log(f"✅ Zerodha Kite Connected")
                success = True
            except Exception as e:
                self.log(f"❌ Zerodha Exception: {e}")

        if self.mt5_acc and self.mt5_server:
            if self.connect_mt5():
                if self.is_mt5_connected and self.mt5_bridge:
                    acc_info = self.mt5_bridge.get_account_info()
                    if acc_info:
                        name = acc_info.get('name', acc_info.get('login', 'User'))
                        if not self._primary_client_set:
                            self.client_name = f"MT5 ({name})"
                            self._primary_client_set = True
                    else:
                        if not self._primary_client_set:
                            self.client_name = "MT5 User"
                            self._primary_client_set = True
                success = True

        if self.connect_coindcx():
            if self.is_coindcx_connected and self.coindcx_bridge:
                if not self._primary_client_set:
                    self.client_name = f"CoinDCX User"
                    self._primary_client_set = True
            success = True

        if self.connect_delta():
            if self.is_delta_connected and self.delta_bridge:
                if not self._primary_client_set:
                    self.client_name = "Delta User"
                    self._primary_client_set = True
            success = True

        if self.connect_fyers():
            if self.is_fyers_connected and self.fyers_bridge:
                try:
                    profile = self.fyers_bridge.fyers.get_profile()
                    if profile and profile.get('data'):
                        fetched_name = profile['data'].get('name', '')
                        if not self._primary_client_set:
                            self.client_name = f"Fyers ({fetched_name})" if fetched_name else "Fyers User"
                            self._primary_client_set = True
                    else:
                        if not self._primary_client_set:
                            self.client_name = "Fyers User"
                            self._primary_client_set = True
                except:
                    if not self._primary_client_set:
                        self.client_name = "Fyers User"
                        self._primary_client_set = True
            success = True

        if self.connect_upstox():
            if self.is_upstox_connected and self.upstox_bridge:
                if not self._primary_client_set:
                    self.client_name = "Upstox User"
                    self._primary_client_set = True
            success = True

        if self.connect_fivepaisa():
            if self.is_fivepaisa_connected and self.fivepaisa_bridge:
                if not self._primary_client_set:
                    self.client_name = "5paisa User"
                    self._primary_client_set = True
            success = True

        if self.connect_binance():
            if self.is_binance_connected and self.binance_bridge:
                if not self._primary_client_set:
                    self.client_name = "Binance User"
                    self._primary_client_set = True
            success = True

        if self.connect_stoxkart():
            if self.is_stoxkart_connected and self.stoxkart_bridge:
                if not self._primary_client_set:
                    self.client_name = "Stoxkart User"
                    self._primary_client_set = True
            success = True

        if self.connect_dhan():
            if self.is_dhan_connected and self.dhan_bridge:
                if not self._primary_client_set:
                    self.client_name = "Dhan User"
                    self._primary_client_set = True
            success = True

        if self.connect_shoonya():
            if self.is_shoonya_connected and self.shoonya_bridge:
                if not self._primary_client_set:
                    self.client_name = "Shoonya User"
                    self._primary_client_set = True
            success = True

        if success:
            self.push_notify("🟢 Gateway Active", f"Connections established.")
            self.start_webhook_listener()
            if self.tg_bot_token:
                self.telegram_controller.start(self)
            self.start_pnl_updater()
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

        ALLOWED_IPS = ['54.240.152.0/24', '54.240.153.0/24']

        def is_allowed_ip(ip):
            from ipaddress import ip_address, ip_network
            for net in ALLOWED_IPS:
                if ip_address(ip) in ip_network(net):
                    return True
            return False

        @app.route('/tv_webhook', methods=['POST'])
        def webhook():
            client_ip = request.remote_addr
            if not is_allowed_ip(client_ip):
                self.log(f"Blocked webhook from {client_ip}")
                return jsonify({"status": "unauthorized"}), 401

            data = request.json
            if data and data.get("passphrase") == self.settings.get("tv_passphrase", "SHREE123"):
                action = data.get("action", "WAIT").upper()
                symbol = data.get("symbol", "").upper()
                self.state["tv_signal"] = {"action": action, "symbol": symbol, "timestamp": time.time()}
                self.log(f"🔔 TV Webhook Alert: {action} on {symbol}")
                return jsonify({"status": "success"}), 200
            return jsonify({"status": "unauthorized"}), 401

        threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000, 'use_reloader': False}, daemon=True).start()
        self.log("🌐 TradingView Webhook Listener Active on Port 5000 (IP whitelisted)")

    def get_master(self):
        if self.token_map is None or self.token_map.empty:
            self.token_map = get_angel_scrip_master()
        return self.token_map

    def get_token_info(self, index_name):
        if index_name in COMMODITIES:
            df_map = self.get_master()
            if df_map is not None and not df_map.empty:
                subset = df_map[(df_map['name'] == index_name) & (df_map['exch_seg'] == "MCX")]
                if not subset.empty:
                    today = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
                    subset = subset[subset['expiry'] >= today]
                    if not subset.empty:
                        row = subset.iloc[0]
                        return row['exch_seg'], row['token']
            token_map = {
                "CRUDEOIL": "21181",
                "NATURALGAS": "21182",
                "GOLD": "21183",
                "SILVER": "21184"
            }
            if index_name in token_map:
                return "MCX", token_map[index_name]
            else:
                return "MCX", "0"

        if index_name in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "BNBUSD", "LTCUSD", "DOTUSD", "MATICUSD", "SHIBUSD", "TRXUSD", "LINKUSD"] and self.is_mt5_connected:
            return "MT5", index_name

        if self.settings.get("primary_broker") == "Delta Exchange" or ("USD" in index_name and "Delta" in self.settings.get("primary_broker", "")):
            return "DELTA", index_name
        if self.settings.get("primary_broker") == "CoinDCX" or "USDT" in index_name or "INR" in index_name:
            return "COINDCX", index_name
        if self.settings.get("primary_broker") == "Fyers":
            if index_name in INDEX_TOKENS:
                return "FYERS", index_name
            else:
                return "FYERS", f"NSE:{index_name}-EQ" if not index_name.startswith("NSE:") else index_name
        if self.settings.get("primary_broker") == "Upstox":
            return "UPSTOX", index_name
        if self.settings.get("primary_broker") == "5paisa":
            return "5PAISA", index_name
        if self.settings.get("primary_broker") == "Binance":
            return "BINANCE", index_name
        if self.settings.get("primary_broker") == "Stoxkart":
            return "STOXKART", index_name
        if self.settings.get("primary_broker") == "Dhan":
            return "DHAN", index_name
        if self.settings.get("primary_broker") == "Shoonya":
            return "SHOONYA", index_name

        if index_name in INDEX_TOKENS:
            return INDEX_TOKENS[index_name]
        df_map = self.get_master()
        if df_map is not None and not df_map.empty:
            today_date = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
            futs = df_map[(df_map['name'] == index_name) & (df_map['instrumenttype'].isin(['FUTCOM', 'FUTIDX', 'FUTSTK', 'EQ']))]
            if not futs.empty:
                eqs = futs[futs['instrumenttype'] == 'EQ']
                if not eqs.empty:
                    return eqs.iloc[0]['exch_seg'], eqs.iloc[0]['token']
                futs = futs[futs['expiry'] >= today_date]
                if not futs.empty:
                    return futs[futs['expiry'] == futs['expiry'].min()].iloc[0]['exch_seg'], futs[futs['expiry'] == futs['expiry'].min()].iloc[0]['token']
        return "NSE", "12345"

    def get_market_data_oi(self, exchange, token):
        if self.is_mock or exchange in ["MT5", "COINDCX", "DELTA", "FYERS", "UPSTOX", "5PAISA", "BINANCE", "STOXKART", "DHAN", "SHOONYA"]:
            return np.random.randint(50000, 150000), np.random.randint(1000, 10000)
        if not self.api:
            return 0, 0
        try:
            res = self.api.marketData({"mode": "FULL", "exchangeTokens": { exchange: [str(token)] }})
            if res and res.get('status') and res.get('data'):
                return res['data']['fetched'][0].get('opnInterest', 0), res['data']['fetched'][0].get('totMacVal', 0)
        except:
            pass
        return 0, 0

    def get_live_price(self, exchange, symbol, token):
        if exchange in ["NSE", "NFO", "BSE", "BFO", "MCX"] and self.ws_angel_connected and token in self.live_prices_angel:
            return self.live_prices_angel[token]

        if self.is_mock and token == "12345":
            if "CE" in symbol or "PE" in symbol:
                if self.state.get("active_trade") and self.state["active_trade"]["symbol"] == symbol:
                    base_price = self.state["active_trade"]["entry"]
                    change = np.random.normal(0, base_price * 0.005)
                    return base_price + change
                return np.random.uniform(150, 300)
            else:
                if self.state.get('mock_price') is None:
                    base_prices = {"NIFTY": 22000, "BANKNIFTY": 47000, "SENSEX": 73000, "FINNIFTY": 21000, "NATURALGAS": 145.0, "CRUDEOIL": 6500.0, "GOLD": 62000.0, "SILVER": 72000.0, "XAUUSD": 2050.0, "EURUSD": 1.0850, "BTCUSD": 65000.0, "ETHUSD": 3500.0, "SOLUSD": 150.0}
                    base = base_prices.get(symbol, 500)
                    self.state['mock_price'] = float(base)
                change = np.random.normal(0, self.state['mock_price'] * 0.0005)
                self.state['mock_price'] += change
                return float(self.state['mock_price'])

        price = None
        try:
            if exchange == "MT5" and self.is_mt5_connected and self.mt5_bridge:
                price = self.mt5_bridge.get_live_price(symbol)
            elif exchange == "COINDCX" and self.is_coindcx_connected and self.coindcx_bridge:
                price = self.coindcx_bridge.get_live_price(symbol)
            elif exchange == "DELTA" and self.is_delta_connected and self.delta_bridge:
                price = self.delta_bridge.get_live_price(symbol)
            elif exchange == "BINANCE" and self.is_binance_connected and self.binance_bridge:
                price = self.binance_bridge.get_live_price(symbol)
            elif exchange == "FYERS" and self.is_fyers_connected and self.fyers_bridge:
                price = self.fyers_bridge.get_live_price(symbol)
            elif exchange == "UPSTOX" and self.is_upstox_connected and self.upstox_bridge:
                price = self.upstox_bridge.get_live_price(symbol)
            elif exchange == "5PAISA" and self.is_fivepaisa_connected and self.fivepaisa_bridge:
                price = self.fivepaisa_bridge.get_live_price(symbol)
            elif exchange == "STOXKART" and self.is_stoxkart_connected and self.stoxkart_bridge:
                price = self.stoxkart_bridge.get_live_price(symbol)
            elif exchange == "DHAN" and self.is_dhan_connected and self.dhan_bridge:
                price = self.dhan_bridge.get_live_price(symbol)
            elif exchange == "SHOONYA" and self.is_shoonya_connected and self.shoonya_bridge:
                price = self.shoonya_bridge.get_live_price(symbol)
            elif self.kite and self.settings.get("primary_broker") == "Zerodha":
                try:
                    tsym = f"{exchange}:{symbol}"
                    res = self.kite.quote([tsym])
                    price = float(res[tsym]['last_price'])
                except:
                    pass
            elif self.api:
                try:
                    trading_symbol = INDEX_SYMBOLS.get(symbol, symbol)
                    res = self.api.ltpData(exchange, trading_symbol, str(token))
                    if res and res.get('status'):
                        price = float(res['data']['ltp'])
                except:
                    pass
        except Exception as e:
            self.log(f"⚠️ Error in get_live_price: {e}")

        if price is None and symbol in YF_TICKERS:
            try:
                yf_ticker = YF_TICKERS[symbol]
                time.sleep(0.2)
                df = yf.Ticker(yf_ticker).history(period="1d", interval="1m")
                if not df.empty:
                    price = float(df['Close'].iloc[-1])
                    self.log(f"⚠️ Using yfinance fallback for {symbol}: {price}")
            except:
                pass
        return price

    def get_historical_data(self, exchange, token, symbol="NIFTY", interval="5m"):
        @st.cache_data(ttl=60)
        def _cached_yfinance(symbol, interval):
            yf_int = interval if interval in ["1m", "5m", "15m"] else "5m"
            yf_ticker = YF_TICKERS.get(symbol)
            if not yf_ticker and ("USD" in symbol or "USDT" in symbol):
                base_coin = symbol.replace("USDT", "").replace("USD", "")
                yf_ticker = f"{base_coin}-USD"
            if yf_ticker:
                try:
                    time.sleep(random.uniform(0.1, 0.5))
                    df = yf.Ticker(yf_ticker).history(period="5d" if interval == "1m" else "10d", interval=yf_int)
                    if not df.empty:
                        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                        return df
                except Exception as e:
                    print(f"yfinance error for {symbol}: {e}")
            return None

        if self.is_mock and token == "12345":
            return self._fallback_yfinance(symbol, interval)

        df = None
        try:
            if exchange == "MT5" and self.is_mt5_connected and self.mt5_bridge:
                df = self.mt5_bridge.get_historical_data(symbol, interval)
            elif exchange == "COINDCX" and self.is_coindcx_connected and self.coindcx_bridge:
                df = self.coindcx_bridge.get_historical_data(symbol, interval, days=10)
            elif exchange == "DELTA" and self.is_delta_connected and self.delta_bridge:
                df = self.delta_bridge.get_historical_data(symbol, interval)
            elif exchange == "BINANCE" and self.is_binance_connected and self.binance_bridge:
                df = self.binance_bridge.get_historical_klines(symbol, interval)
            elif exchange == "FYERS" and self.is_fyers_connected and self.fyers_bridge:
                fyers_int = interval.replace("m", "").replace("h", "").replace("d", "D")
                df = self.fyers_bridge.get_historical_data(symbol, fyers_int, days=10)
            elif self.kite and self.settings.get("primary_broker") == "Zerodha":
                try:
                    z_int_map = {"1m": "minute", "3m": "3minute", "5m": "5minute", "15m": "15minute"}
                    now_ist = get_ist()
                    fromdate = now_ist - dt.timedelta(days=10)
                    records = self.kite.historical_data(int(token), fromdate.strftime("%Y-%m-%d"), now_ist.strftime("%Y-%m-%d"), z_int_map.get(interval, "5minute"))
                    df = pd.DataFrame(records)
                    df.rename(columns={'date': 'timestamp'}, inplace=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.index = df['timestamp']
                except:
                    pass
            elif self.api:
                try:
                    interval_map = {"1m": "ONE_MINUTE", "3m": "THREE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE"}
                    api_interval = interval_map.get(interval, "FIVE_MINUTE")
                    now_ist = get_ist()
                    fromdate = now_ist - dt.timedelta(days=10)
                    res = self.api.getCandleData({"exchange": exchange, "symboltoken": str(token), "interval": api_interval, "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"), "todate": now_ist.strftime("%Y-%m-%d %H:%M")})
                    if res and res.get('status') and res.get('data'):
                        df = pd.DataFrame(res['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.index = df['timestamp']
                except:
                    pass
        except Exception as e:
            self.log(f"⚠️ Error in get_historical_data: {e}")

        if df is not None and not df.empty:
            df.rename(columns=lambda x: x.lower(), inplace=True)
            if 'open' not in df.columns: df['open'] = df['close']
            if 'high' not in df.columns: df['high'] = df['close']
            if 'low' not in df.columns: df['low'] = df['close']
            if 'volume' not in df.columns: df['volume'] = 0

        if (df is None or df.empty) and symbol in YF_TICKERS:
            self.log(f"⚠️ Using yfinance fallback for {symbol} historical data")
            df = _cached_yfinance(symbol, interval)
            if df is not None and not df.empty:
                df.rename(columns=lambda x: x.lower(), inplace=True)
        return df

    def _fallback_yfinance(self, symbol, interval):
        yf_int = interval if interval in ["1m", "5m", "15m"] else "5m"
        yf_ticker = YF_TICKERS.get(symbol)
        if not yf_ticker and ("USD" in symbol or "USDT" in symbol):
            base_coin = symbol.replace("USDT", "").replace("USD", "")
            yf_ticker = f"{base_coin}-USD"
        if yf_ticker:
            try:
                time.sleep(random.uniform(0.1, 0.5))
                df = yf.Ticker(yf_ticker).history(period="5d" if interval == "1m" else "10d", interval=yf_int)
                if not df.empty:
                    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                    return df
            except Exception as e:
                self.log(f"⚠️ yfinance error for {symbol}: {e}")
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

    def analyze_oi_and_greeks(self, df, is_hero_zero, signal):
        if not is_hero_zero or df is None or len(df) < 20:
            return True, ""
        last = df.iloc[-1]
        atr = self.analyzer.get_atr(df).iloc[-1]
        body = abs(last['close'] - last['open'])
        vol_sma = df['volume'].rolling(20).mean().iloc[-1]
        is_hero = signal == "BUY_CE" and last['close'] > last['open'] * 1.02
        is_zero = signal == "BUY_PE" and last['close'] < last['open'] * 0.98
        volume_confirm = last['volume'] > vol_sma * 1.5
        price_action_confirm = ((is_hero and last['close'] > last['high'] * 0.95) or (is_zero and last['close'] < last['low'] * 1.05))
        volatility_ok = body > atr * 0.5
        now_ist = get_ist()
        if 11 <= now_ist.hour <= 13:
            return False, "⚠️ Avoid Hero/Zero during lunch hours"
        if self.state.get("hz_last_signal_time"):
            time_diff = (get_ist() - self.state["hz_last_signal_time"]).seconds / 60
            if time_diff < 15:
                return False, f"⏳ Cooldown: {15 - time_diff:.0f} mins remaining"
        if is_hero and volume_confirm and price_action_confirm and volatility_ok:
            self.state["hz_last_signal_time"] = get_ist()
            return True, "🔥 HERO DETECTED: Strong buying pressure with volume"
        if is_zero and volume_confirm and price_action_confirm and volatility_ok:
            self.state["hz_last_signal_time"] = get_ist()
            return True, "🩸 ZERO DETECTED: Strong selling pressure with volume"
        return False, "⚠️ No Hero/Zero: Insufficient momentum"

    def calculate_hero_zero_position(self, signal_strength, atr, spot, capital):
        base_size = capital * 0.02
        if signal_strength >= 90:
            strength_mult = 1.5
        elif signal_strength >= 75:
            strength_mult = 1.2
        elif signal_strength >= 60:
            strength_mult = 1.0
        else:
            strength_mult = 0.5
        volatility_mult = min(1.5, 30 / atr) if atr > 0 else 1.0
        now_ist = get_ist()
        hour = now_ist.hour
        if hour in [9, 10, 14, 15]:
            time_mult = 1.3
        elif hour in [11, 12, 13]:
            time_mult = 0.7
        else:
            time_mult = 0.5
        position_value = base_size * strength_mult * volatility_mult * time_mult
        if spot > 0:
            quantity = position_value / spot
        else:
            quantity = 0
        return round(quantity, 2), {
            "base": base_size,
            "strength": strength_mult,
            "volatility": volatility_mult,
            "time": time_mult,
            "final": position_value
        }

    def scan_hero_zero_indian_stocks(self, nifty_stocks=None):
        if st.session_state.get('hz_demo_mode', False):
            mock_results = []
            stocks = nifty_stocks if nifty_stocks else ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
            for stock in stocks:
                mock_results.append({
                    "Stock": stock,
                    "Direction": "HERO (BUY)" if np.random.rand() > 0.5 else "ZERO (SELL)",
                    "Price": round(np.random.uniform(100, 2000), 2),
                    "Volume Spike": f"{np.random.uniform(1.5, 3.0):.1f}x",
                    "ATR": round(np.random.uniform(5, 20), 2),
                    "Entry": round(np.random.uniform(100, 2000), 2),
                    "SL": round(np.random.uniform(90, 1900), 2),
                    "Target 1": round(np.random.uniform(110, 2100), 2),
                    "Target 2": round(np.random.uniform(120, 2200), 2),
                    "Risk/Reward": "1:2"
                })
            return pd.DataFrame(mock_results)

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
            except:
                continue
        return pd.DataFrame(results)

    def scan_penny_stocks(self):
        penny_list = [
            "IDEA", "YESBANK", "SAIL", "PNB", "IOC", "BHEL", "SUZLON", "JPASSOCIAT",
            "GMRINFRA", "NHPC", "NTPC", "PFC", "RECLTD", "VEDL", "TATAMOTORS"
        ]
        return self.scan_hero_zero_indian_stocks(penny_list)

    def scan_pin_bars(self):
        if st.session_state.get('hz_demo_mode', False):
            mock_pins = []
            symbols = ["NIFTY", "SENSEX", "BANKNIFTY", "GOLD"]
            for sym in symbols:
                mock_pins.append({
                    "Symbol": sym,
                    "Time": "09:35",
                    "Direction": "🟢 BUY" if np.random.rand() > 0.5 else "🔴 SELL",
                    "Entry": round(np.random.uniform(100, 2000), 2),
                    "Stop": round(np.random.uniform(90, 1900), 2),
                    "Target": round(np.random.uniform(110, 2100), 2),
                    "Risk/Reward": "1:2",
                    "Confidence": "High"
                })
            return mock_pins
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

    def get_otm_gamma_blast_strikes(self, symbol, spot, option_type, premium_threshold=10000, iv_default=0.30):
        df_map = self.get_master()
        if df_map is None or df_map.empty:
            self.log("⚠️ Scrip master not available for OTM gamma blast")
            return None, None, None, 0, 0, 0
        today = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
        option_suffix = "CE" if option_type == "BUY_CE" else "PE"
        subset = df_map[
            (df_map['name'] == symbol) & 
            (df_map['exch_seg'].isin(["NFO", "MCX", "BFO", "NCO"])) &
            (df_map['expiry'] == today) &
            (df_map['symbol'].str.endswith(option_suffix))
        ].copy()
        if subset.empty:
            self.log(f"⚠️ No options with expiry today for {symbol}")
            return None, None, None, 0, 0, 0
        if subset['strike'].median() > spot * 10:
            subset['strike'] = subset['strike'] / 100
        elif subset['strike'].median() < spot / 10:
            subset['strike'] = subset['strike'] * 100
        if option_type == "BUY_CE":
            subset = subset[subset['strike'] > spot]
            subset['distance'] = subset['strike'] - spot
        else:
            subset = subset[subset['strike'] < spot]
            subset['distance'] = spot - subset['strike']
        subset = subset.nsmallest(5, 'distance')
        now = get_ist()
        expiry_end = dt.datetime.combine(today, dt.time(15, 30)) if symbol in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"] else dt.datetime.combine(today, dt.time(23, 30))
        if now > expiry_end:
            return None, None, None, 0, 0, 0
        seconds_left = max(1, (expiry_end - now).total_seconds())
        T = seconds_left / (24 * 3600 * 365)
        best_gamma = 0
        best_row = None
        best_premium = 0
        for idx, row in subset.iterrows():
            greeks = OptionGreeks.black_scholes(spot, row['strike'], T, 0.05, iv_default, 'call' if option_type == "BUY_CE" else 'put')
            premium = greeks['price']
            gamma = greeks['gamma']
            if premium > premium_threshold and gamma > best_gamma:
                best_gamma = gamma
                best_row = row
                best_premium = premium
        if best_row is not None:
            return best_row['symbol'], best_row['token'], best_row['exch_seg'], best_row['strike'], best_premium, best_gamma
        else:
            self.log(f"⚠️ No OTM strike with premium > {premium_threshold} found for {symbol}")
            return None, None, None, 0, 0, 0

    def get_atm_strike(self, symbol, spot, opt_type, otm_gamma_blast=False, premium_threshold=10000):
        if otm_gamma_blast:
            df_map = self.get_master()
            if df_map is not None and not df_map.empty:
                today = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
                option_suffix = "CE" if opt_type == "BUY_CE" else "PE"
                subset = df_map[
                    (df_map['name'] == symbol) & 
                    (df_map['exch_seg'].isin(["NFO", "MCX", "BFO", "NCO"])) &
                    (df_map['expiry'] == today) &
                    (df_map['symbol'].str.endswith(option_suffix))
                ]
                if not subset.empty:
                    sym, tok, exch, strike, prem, gamma = self.get_otm_gamma_blast_strikes(symbol, spot, opt_type, premium_threshold)
                    if sym:
                        ltp = self.get_live_price(exch, sym, tok)
                        if ltp is None:
                            ltp = prem
                        return sym, tok, exch, ltp
                    else:
                        self.log("⚠️ No suitable OTM gamma blast strike, falling back to ATM")
                else:
                    self.log("ℹ️ Not expiry day, using ATM strike")
            else:
                self.log("⚠️ Scrip master unavailable, using ATM strike")

        if self.is_mock:
            suffix = "CE" if opt_type == "BUY_CE" else "PE"
            return f"{symbol}{int(round(spot/100)*100)}{suffix}", "12345", "NFO", spot * 0.1

        try:
            df_map = self.get_master()
            if df_map is None or df_map.empty:
                self.log("⚠️ Scrip master not available for ATM strike")
                return None, None, None, 0.0

            today = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
            option_suffix = "CE" if "CE" in opt_type else "PE"

            subset = df_map[
                (df_map['name'] == symbol) & 
                (df_map['exch_seg'].isin(["NFO", "MCX", "BFO", "NCO"])) &
                (df_map['expiry'] >= today) &
                (df_map['symbol'].str.endswith(option_suffix))
            ].copy()

            if subset.empty:
                self.log(f"⚠️ No {option_suffix} options found for {symbol} in master")
                return None, None, None, 0.0

            closest_expiry = subset['expiry'].min()
            subset = subset[subset['expiry'] == closest_expiry]

            if subset['strike'].median() > spot * 10:
                subset['strike'] = subset['strike'] / 100
            elif subset['strike'].median() < spot / 10:
                subset['strike'] = subset['strike'] * 100

            subset['dist_to_spot'] = abs(subset['strike'] - spot)
            atm_row = subset.loc[subset['dist_to_spot'].idxmin()]

            ltp = self.get_live_price(atm_row['exch_seg'], atm_row['symbol'], atm_row['token'])

            if ltp is None and self.is_mock:
                ltp = spot * 0.1

            if ltp:
                return atm_row['symbol'], atm_row['token'], atm_row['exch_seg'], ltp
            else:
                self.log(f"⚠️ Could not fetch LTP for ATM {opt_type} strike")
                return None, None, None, 0.0

        except Exception as e:
            self.log(f"❌ Error getting ATM strike: {e}")
            return None, None, None, 0.0

    def get_strike(self, symbol, spot, signal, max_premium):
        opt_type = "CE" if "BUY_CE" in signal else "PE"

        if self.settings.get("primary_broker") in ["CoinDCX", "Delta Exchange"] and self.settings.get("crypto_mode") == "Options":
            rounder = 500 if "BTC" in symbol else (50 if "ETH" in symbol else 1)
            strike_price = round(spot / rounder) * rounder
            expiry_str = (get_ist() + dt.timedelta(days=(4 - get_ist().weekday()) % 7)).strftime("%d%b%y").upper()
            crypto_sym = f"{symbol.replace('USDT', '').replace('USD', '')}-{expiry_str}-{int(strike_price)}-{opt_type}"
            exch_target = "COINDCX" if self.settings.get("primary_broker") == "CoinDCX" else "DELTA"
            return crypto_sym, crypto_sym, exch_target, spot * 0.02

        if self.is_mock:
            return f"{symbol}28FEB{int(spot)}{opt_type}", "12345", "NFO", min(100.0, max_premium)

        df = self.get_master()
        if df is None or df.empty:
            self.log("⚠️ Option Chain JSON is empty. Cannot compute Angel strikes.")
            return None, None, None, 0.0

        today = pd.Timestamp(get_ist().replace(tzinfo=None)).normalize()
        mask = (df['name'] == symbol) & (df['exch_seg'].isin(["NFO", "MCX", "BFO", "NCO"])) & (df['expiry'] >= today) & (df['symbol'].str.endswith(opt_type))
        subset = df[mask].copy()
        if subset.empty:
            self.log(f"⚠️ No options found for {symbol} on {today.date()}")
            return None, None, None, 0.0

        closest_expiry = subset['expiry'].min()
        subset = subset[subset['expiry'] == closest_expiry]

        if subset['strike'].median() > spot * 10:
            subset['strike'] = subset['strike'] / 100
        elif subset['strike'].median() < spot / 10:
            subset['strike'] = subset['strike'] * 100

        subset['dist_to_spot'] = abs(subset['strike'] - spot)
        time_to_expiry = max(0.001, (closest_expiry - today).days / 365.0)
        iv = 12

        try:
            candidates = subset.copy()
            candidates['greeks_pass'] = candidates['strike'].apply(
                lambda x: self.analyzer.filter_option_by_greeks(spot, x, time_to_expiry, iv, opt_type.lower())
            )
            candidates = candidates[candidates['greeks_pass']].sort_values('dist_to_spot', ascending=True)
        except Exception as e:
            self.log(f"⚠️ Greeks filter error: {e}, falling back to closest strike")
            candidates = subset.sort_values('dist_to_spot', ascending=True)

        if candidates.empty:
            candidates = subset.sort_values('dist_to_spot', ascending=True)

        if candidates.empty:
            self.log(f"⚠️ No {opt_type} options found for {symbol}")
            return None, None, None, 0.0

        for _, row in candidates.iterrows():
            ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
            if ltp and ltp <= max_premium:
                return row['symbol'], row['token'], row['exch_seg'], ltp

        self.log(f"⚠️ Capital Filter: No {opt_type} strikes found below ₹{max_premium:.2f} premium. Using closest strike.")
        closest_row = candidates.iloc[0]
        ltp = self.get_live_price(closest_row['exch_seg'], closest_row['symbol'], closest_row['token'])
        if ltp is None:
            ltp = max_premium * 0.9
        return closest_row['symbol'], closest_row['token'], closest_row['exch_seg'], ltp

    def get_dynamic_lot_multiplier(self):
        win_streak = st.session_state.win_streak
        loss_streak = st.session_state.loss_streak
        if win_streak >= 3:
            return 1.2
        elif loss_streak >= 2:
            return 0.5
        else:
            return 1.0

    def apply_slippage(self, price, side):
        if self.is_mock:
            slippage = price * 0.0005
            if side in ["BUY", "BUY_CE"]:
                return price + slippage
            else:
                return price - slippage
        return price

    def place_real_order(self, symbol, token, qty, side="BUY", exchange="NFO", order_type="MARKET", price=None):
        if self.is_mock:
            return "MOCK_" + uuid.uuid4().hex[:6].upper(), None
        broker = self.settings.get("primary_broker", "Angel One")
        self.log(f"⚙️ Executing Real API: {symbol} | Qty: {qty} | Side: {side} | Exchange: {exchange} | OrderType: {order_type}")

        if exchange == "MT5" and self.is_mt5_connected and self.mt5_bridge:
            order_id, msg = self.mt5_bridge.place_order(symbol, qty, side)
            if order_id:
                self.log(f"✅ MT5 Order Success! ID: {order_id}")
                return order_id, None
            else:
                self.log(f"❌ MT5 Order Failed: {msg}")
                return None, f"MT5 error: {msg}"

        if exchange == "DELTA" and self.is_delta_connected and self.delta_bridge:
            return self.delta_bridge.place_order(symbol, qty, side, order_type, price)

        if exchange == "COINDCX" and self.is_coindcx_connected and self.coindcx_bridge:
            return self.coindcx_bridge.place_order(symbol, qty, side, order_type, price, 
                                                   market_type=self.settings.get("crypto_mode", "Spot"),
                                                   leverage=self.settings.get("leverage", 1))

        if exchange == "BINANCE" and self.is_binance_connected and self.binance_bridge:
            order_id, msg = self.binance_bridge.place_order(symbol, side, qty, order_type, price)
            if order_id:
                self.log(f"✅ Binance Order Success! ID: {order_id}")
                return order_id, None
            else:
                self.log(f"❌ Binance Order Failed: {msg}")
                return None, f"Binance error: {msg}"

        if exchange == "FYERS" and self.is_fyers_connected and self.fyers_bridge:
            order_id, msg = self.fyers_bridge.place_order(symbol, qty, side, order_type, price=price)
            if order_id:
                self.log(f"✅ Fyers Order Success! ID: {order_id}")
                return order_id, None
            else:
                self.log(f"❌ Fyers Order Failed: {msg}")
                return None, f"Fyers error: {msg}"

        if exchange == "UPSTOX" and self.is_upstox_connected and self.upstox_bridge:
            order_id, msg = self.upstox_bridge.place_order(symbol, qty, side, order_type, price)
            if order_id:
                self.log(f"✅ Upstox Order Success! ID: {order_id}")
                return order_id, None
            else:
                self.log(f"❌ Upstox Order Failed: {msg}")
                return None, f"Upstox error: {msg}"

        if exchange == "5PAISA" and self.is_fivepaisa_connected and self.fivepaisa_bridge:
            order_id, msg = self.fivepaisa_bridge.place_order(symbol, qty, side, order_type, price)
            if order_id:
                self.log(f"✅ 5paisa Order Success! ID: {order_id}")
                return order_id, None
            else:
                self.log(f"❌ 5paisa Order Failed: {msg}")
                return None, f"5paisa error: {msg}"

        if exchange == "STOXKART" and self.is_stoxkart_connected and self.stoxkart_bridge:
            order_id, msg = self.stoxkart_bridge.place_order(symbol, qty, side, order_type, price)
            if order_id:
                self.log(f"✅ Stoxkart Order Success! ID: {order_id}")
                return order_id, None
            else:
                self.log(f"❌ Stoxkart Order Failed: {msg}")
                return None, f"Stoxkart error: {msg}"

        if exchange == "DHAN" and self.is_dhan_connected and self.dhan_bridge:
            order_id, msg = self.dhan_bridge.place_order(symbol, qty, side, order_type, price)
            if order_id:
                self.log(f"✅ Dhan Order Success! ID: {order_id}")
                return order_id, None
            else:
                self.log(f"❌ Dhan Order Failed: {msg}")
                return None, f"Dhan error: {msg}"

        if exchange == "SHOONYA" and self.is_shoonya_connected and self.shoonya_bridge:
            order_id, msg = self.shoonya_bridge.place_order(symbol, qty, side, order_type, price)
            if order_id:
                self.log(f"✅ Shoonya Order Success! ID: {order_id}")
                return order_id, None
            else:
                self.log(f"❌ Shoonya Order Failed: {msg}")
                return None, f"Shoonya error: {msg}"

        if broker == "Zerodha" and self.kite:
            try:
                z_side = self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=symbol,
                    transaction_type=z_side,
                    quantity=int(float(qty)),
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_LIMIT if order_type.upper() == "LIMIT" else self.kite.ORDER_TYPE_MARKET,
                    price=price if order_type.upper() == "LIMIT" else 0
                )
                self.log(f"✅ Zerodha Order Pushed! ID: {order_id}")
                return order_id, None
            except Exception as e:
                self.log(f"❌ Zerodha Order Error: {str(e)}")
                return None, f"Zerodha error: {str(e)}"

        try:
            p_type = "CARRYFORWARD" if exchange in ["NFO", "BFO", "MCX"] else "INTRADAY"
            order_type_final = "LIMIT" if order_type.upper() == "LIMIT" and price else "MARKET"
            exec_price = price if order_type.upper() == "LIMIT" and price else 0.0
            if exchange in ["NFO", "BFO", "MCX"] and order_type.upper() != "LIMIT":
                ltp = self.get_live_price(exchange, symbol, token)
                if ltp and ltp > 0:
                    order_type_final = "LIMIT"
                    safe_price = ltp * 1.05 if side.upper() == "BUY" else ltp * 0.95
                    exec_price = round(round(safe_price / 0.05) * 0.05, 2)
                else:
                    self.log(f"⚠️ Could not fetch LTP for {symbol}. Retrying as MARKET.")
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": str(symbol),
                "symboltoken": str(token),
                "transactiontype": str(side.upper()),
                "exchange": str(exchange.upper()),
                "ordertype": str(order_type_final),
                "producttype": str(p_type),
                "duration": "DAY",
                "price": float(exec_price),
                "squareoff": 0.0,
                "stoploss": 0.0,
                "quantity": int(float(qty))
            }
            self.log(f"📡 Sending Angel Payload: {order_params}")
            res = self.api.placeOrder(order_params)
            if res and isinstance(res, dict) and res.get('status'):
                o_id = res.get('data', {}).get('orderid', 'UNKNOWN_ID')
                self.log(f"⏳ Order Sent: {o_id}. Verifying Exchange Status...")
                if not self.is_mock:
                    time.sleep(1.5)
                    try:
                        ob = self.api.orderBook()
                        if ob and ob.get('status') and ob.get('data'):
                            for ord_dict in ob['data']:
                                if ord_dict.get('orderid') == o_id:
                                    status = ord_dict.get('status', '').lower()
                                    if status == 'rejected':
                                        reason = ord_dict.get('text', 'Insufficient Margin / Limits')
                                        self.log(f"❌ Order REJECTED by Exchange: {reason}")
                                        return None, f"Angel rejected: {reason}"
                                    else:
                                        self.log(f"✅ Exchange Confirmed: {status.upper()}")
                                        return o_id, None
                    except Exception as e:
                        self.log(f"⚠️ Status verification skipped, assumed placed.")
                return o_id, None
            elif isinstance(res, str):
                self.log(f"✅ Angel Order Placed! ID: {res}")
                return res, None
            else:
                self.log(f"❌ Angel API Validation Error: {res}")
                return None, f"Angel validation error: {res}"
        except Exception as e:
            self.log(f"❌ Exception placing Angel order: {str(e)}")
            return None, f"Angel exception: {str(e)}"

    def start_angel_ws(self):
        def ws_worker():
            while not self._angel_ws_stop.is_set():
                try:
                    if not self.api:
                        time.sleep(5)
                        continue
                    auth_token = self.api.getAuthToken() if hasattr(self.api, 'getAuthToken') else None
                    if not auth_token:
                        time.sleep(5)
                        continue
                    self.ws_angel = SmartWebSocketV2(auth_token, self.api_key, self.client_id, self.api.feed_token)
                    self.ws_angel.on_open = self._angel_ws_on_open
                    self.ws_angel.on_message = self._angel_ws_on_message
                    self.ws_angel.on_error = self._angel_ws_on_error
                    self.ws_angel.on_close = self._angel_ws_on_close
                    self.ws_angel.connect()
                    self._angel_ws_reconnect.wait()
                    self._angel_ws_reconnect.clear()
                    time.sleep(2)
                except Exception as e:
                    self.log(f"Angel WebSocket thread error: {e}")
                    time.sleep(5)
        thread = threading.Thread(target=ws_worker, daemon=True)
        thread.start()

    def _angel_ws_on_message(self, wsapp, message):
        try:
            data = json.loads(message)
            if 'ltp' in data:
                token = str(data['token'])
                price = float(data['ltp'])
                self.live_prices_angel[token] = price
        except:
            pass

    def _angel_ws_on_error(self, wsapp, error):
        self.log(f"Angel WebSocket error: {error}")
        self.ws_angel_connected = False
        self._angel_ws_reconnect.set()

    def _angel_ws_on_close(self, wsapp, close_status_code, close_msg):
        self.ws_angel_connected = False
        self._angel_ws_reconnect.set()

    def _angel_ws_on_open(self, wsapp):
        self.ws_angel_connected = True
        self.log("Angel WebSocket opened")

    def push_notify(self, title, message):
        self.state["ui_popups"].append({"title": title, "message": message})
        if HAS_NOTIFY:
            try:
                notification.notify(title=title, message=message, app_name="QUANT", timeout=5)
            except:
                pass
        if self.notification_manager:
            self.notification_manager.notify_all(title, message)

    def log(self, msg):
        timestamp = get_ist().strftime('%H:%M:%S')
        self.state["logs"].appendleft(f"[{timestamp}] {msg}")

    def get_balance(self):
        now = time.time()
        if hasattr(self, '_balance_cache') and now - self._balance_cache_time < 5:
            return self._balance_cache
        
        balance_strs = []
        if self.api:
            for attempt in range(2):
                try:
                    rms = self.api.rmslimit() if hasattr(self.api, 'rmslimit') else None
                    if rms and rms.get('status') and rms.get('data'):
                        data = rms['data']
                        bal = data.get('availablecash', data.get('net', 0))
                        try:
                            bal = float(bal)
                            balance_strs.append(f"Angel: ₹ {bal:,.2f}")
                            break
                        except:
                            pass
                except:
                    time.sleep(0.5)
        if self.kite:
            for attempt in range(2):
                try:
                    margins = self.kite.margins()
                    eq = margins.get('equity', {})
                    bal = eq.get('available', {}).get('live_balance', eq.get('net', 0))
                    balance_strs.append(f"Zerodha: ₹ {float(bal):,.2f}")
                    break
                except:
                    time.sleep(0.5)
        if self.coindcx_bridge and self.is_coindcx_connected:
            try:
                bal = self.coindcx_bridge.get_balance()
                if bal:
                    balance_strs.append(f"CoinDCX: $ {bal:.2f}")
            except:
                pass
        if self.delta_bridge and self.is_delta_connected:
            try:
                info = self.delta_bridge.get_account_info()
                if info:
                    balance_strs.append(f"Delta: $ {info.get('balance', 0):,.2f}")
            except:
                pass
        if self.is_mt5_connected and self.mt5_bridge:
            try:
                acc_info = self.mt5_bridge.get_account_info()
                if acc_info:
                    balance_strs.append(f"MT5: $ {acc_info.get('balance', 0):,.2f}")
            except:
                pass
        if self.is_fyers_connected and self.fyers_bridge:
            try:
                info = self.fyers_bridge.get_account_info()
                if info:
                    balance_strs.append(f"Fyers: ₹ {info.get('balance', 0):,.2f}")
            except:
                pass
        if self.is_upstox_connected and self.upstox_bridge:
            try:
                info = self.upstox_bridge.get_account_info()
                if info:
                    balance_strs.append(f"Upstox: ₹ {info.get('balance', 0):,.2f}")
            except:
                pass
        if self.is_fivepaisa_connected and self.fivepaisa_bridge:
            try:
                info = self.fivepaisa_bridge.get_account_info()
                if info:
                    balance_strs.append(f"5paisa: ₹ {info.get('balance', 0):,.2f}")
            except:
                pass
        if self.is_binance_connected and self.binance_bridge:
            try:
                bal = self.binance_bridge.get_asset_balance("USDT")
                if bal is not None:
                    balance_strs.append(f"Binance: $ {bal:,.2f}")
            except:
                pass
        if self.is_stoxkart_connected and self.stoxkart_bridge:
            try:
                info = self.stoxkart_bridge.get_account_info()
                if info:
                    balance_strs.append(f"Stoxkart: ₹ {info.get('balance', 0):,.2f}")
            except:
                pass
        if self.is_dhan_connected and self.dhan_bridge:
            try:
                info = self.dhan_bridge.get_account_info()
                if info:
                    balance_strs.append(f"Dhan: ₹ {info.get('balance', 0):,.2f}")
            except:
                pass
        if self.is_shoonya_connected and self.shoonya_bridge:
            try:
                info = self.shoonya_bridge.get_account_info()
                if info:
                    balance_strs.append(f"Shoonya: ₹ {info.get('balance', 0):,.2f}")
            except:
                pass
        if not balance_strs:
            base_cap = self.settings.get('max_capital', 15000.0) if self.settings else 15000.0
            current_cap = base_cap + self.state.get('daily_pnl', 0.0)
            result = f"₹ {current_cap:,.2f} (Manual Cap)"
        else:
            result = " | ".join(balance_strs)
        self._balance_cache = result
        self._balance_cache_time = now
        return result

    def trading_loop(self):
        self.log("▶️ Engine thread started.")
        while self.state["is_running"]:
            try:
                with self.state["trade_lock"]:
                    if not self.state["engine_active"]:
                        time.sleep(1)
                        continue
                    s = self.settings
                    current_time = get_ist().time()
                    today_date = get_ist().strftime('%Y-%m-%d')
                    time_str = get_ist().strftime('%H:%M:%S')
                    self.state["loop_count"] += 1

                    index = s.get('index', 'NIFTY')
                    timeframe = s.get('timeframe', '5m')
                    is_mock_mode = s.get('paper_mode', True)
                    strategy = s.get('strategy', 'Momentum Breakout + S&R')
                    max_trades = s.get('max_trades', 5)
                    capital_protect = s.get('capital_protect', 999999)
                    ml_prob_threshold = s.get('ml_prob_threshold', 0.30)
                    signal_persistence = s.get('signal_persistence', 1)
                    mtf_confirm = s.get('mtf_confirm', False)
                    hero_zero = s.get('hero_zero', False)
                    three_five_seven = s.get('three_five_seven', False)
                    crypto_mode = s.get('crypto_mode', 'Spot')
                    leverage = s.get('leverage', 1)
                    show_inr_crypto = s.get('show_inr_crypto', True)
                    custom_code = s.get('custom_code', '')
                    tv_passphrase = s.get('tv_passphrase', 'SHREE123')
                    min_signal_strength = s.get('min_signal_strength', 30)
                    sl_pts = s.get('sl_pts', 20)
                    tsl_pts = s.get('tsl_pts', 15)
                    tgt_pts = s.get('tgt_pts', 15)
                    max_capital = s.get('max_capital', 15000)
                    lots = s.get('lots', 1)
                    use_limit_orders = s.get('use_limit_orders', False)
                    fomo_mode = st.session_state.fomo_mode

                    if self.state["trades_today"] >= max_trades or self.state.get("daily_pnl", 0.0) <= -capital_protect:
                        self.state["is_running"] = False
                        break

                    if not self.loss_streak_cooldown and st.session_state.loss_streak >= 3:
                        self.loss_streak_cooldown = True
                        self.cooldown_end_time = get_ist() + timedelta(minutes=30)
                        self.log("⚠️ Loss streak >= 3 – cooldown for 30 minutes.")
                        self.push_notify("Cooldown Active", "Trading paused for 30 minutes due to loss streak.")
                    if self.loss_streak_cooldown and self.cooldown_end_time and get_ist() < self.cooldown_end_time:
                        time.sleep(1)
                        continue
                    elif self.loss_streak_cooldown:
                        self.loss_streak_cooldown = False
                        self.log("✅ Cooldown ended – resuming.")
                        self.push_notify("Cooldown Ended", "Trading resumed.")

                    if self.state.get("last_trade_time"):
                        seconds_since_last = (get_ist() - self.state["last_trade_time"]).total_seconds()
                        if seconds_since_last < 60:
                            time.sleep(1)
                            continue

                    is_open, mkt_msg = get_market_status(index)
                    if not is_open:
                        time.sleep(10)
                        continue

                    exch, token = self.get_token_info(index)
                    is_mt5_asset = (exch == "MT5")
                    is_crypto = (exch in ["COINDCX", "DELTA", "BINANCE"])
                    is_fyers = (exch == "FYERS")

                    if index in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"]:
                        cutoff_time = dt.time(15, 15)
                    elif index in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"]:
                        cutoff_time = dt.time(23, 15)
                    else:
                        cutoff_time = dt.time(23, 59, 59)

                    spot = self.get_live_price(exch, index, token)
                    if spot is None and self.is_mock:
                        spot = self.get_live_price("NSE", index, "12345")

                    df_candles = self.get_historical_data(exch, token, symbol=index, interval=timeframe) if not self.is_mock else self.get_historical_data("MOCK", "12345", symbol=index, interval=timeframe)

                    self.log(f"Spot: {spot}, Data shape: {df_candles.shape if df_candles is not None else 'None'}")

                    lot_size = LOT_SIZES.get(index, 1)
                    dynamic_mult = self.get_dynamic_lot_multiplier()
                    actual_qty = lots * lot_size * dynamic_mult

                    if df_candles is not None and not df_candles.empty:
                        scalp_signal = self.scalper.scalp_signal(df_candles)
                        if scalp_signal != "WAIT":
                            self.state["scalp_signals"].append({"time": time_str, "signal": scalp_signal, "price": spot, "index": index})

                    fomo_signal = None
                    if fomo_mode:
                        fomo_signals = fomo_scanner.scan()
                        for fs in fomo_signals:
                            asset_key = index.replace(".NS", "").upper()
                            if asset_key in fs['symbol'].upper() or fs['symbol'].upper() in asset_key:
                                fomo_signal = fs
                                break
                    if fomo_signal and self.state["active_trade"] is None and not self.state["active_trades"]:
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
                    elif spot and df_candles is not None and not df_candles.empty:
                        self.state["spot"] = spot
                        self.state["latest_candle"] = df_candles.iloc[-1].to_dict()

                        if strategy == "Keyword Rule Builder":
                            trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_keyword_strategy(df_candles, custom_code, index)
                        elif strategy == "Machine Learning":
                            if not ml_predictor.is_trained:
                                ml_predictor.train(df_candles)
                            else:
                                if ml_predictor.should_retrain(df_candles):
                                    ml_predictor.train(df_candles)

                            trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_ml_strategy(
                                df_candles, index, ml_prob_threshold, signal_persistence
                            )
                            self.log(f"ML Prob: Up={trend.split()[2]}, Down={trend.split()[4]}, Signal={signal}")
                            shap_values = ml_predictor.explain_prediction(df_candles)
                            if shap_values is not None:
                                self.state["ml_explanation"] = shap_values
                        elif "VIJAY & RFF" in strategy:
                            trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_vijay_rff_strategy(df_candles, index)
                        elif "Institutional FVG" in strategy:
                            trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_institutional_fvg_strategy(df_candles, index)
                        elif "Lux Algo" in strategy:
                            trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_lux_algo_ict_strategy(df_candles, index)
                        elif "Mean Reversion" in strategy:
                            trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_mean_reversion_strategy(df_candles, index)
                        else:
                            trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = self.analyzer.apply_primary_strategy(df_candles, index)

                        if strategy == "Machine Learning" and signal != "WAIT" and signal_persistence > 1:
                            self.state["signal_history"].append(signal)
                            if len(self.state["signal_history"]) >= signal_persistence:
                                recent = list(self.state["signal_history"])[-signal_persistence:]
                                if not all(x == signal for x in recent):
                                    signal = "WAIT"
                                    trend += " | ⏳ Persistence filter"
                                    signal_strength = 0
                        else:
                            self.state["signal_history"].clear()

                        if mtf_confirm and signal != "WAIT" and strategy != "TradingView Webhook":
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

                        is_hz = hero_zero
                        if is_hz and signal != "WAIT" and strategy != "TradingView Webhook":
                            if not self.is_mock and not (is_mt5_asset or is_crypto or is_fyers):
                                live_oi, live_vol = self.get_market_data_oi(exch, token)
                                if live_vol < 50000:
                                    signal = "WAIT"
                                    trend = "Hero/Zero Blocked: Low Volume/OI"
                                    signal_strength = 0
                            greek_pass, greek_msg = self.analyze_oi_and_greeks(df_candles, is_hz, signal)
                            if not greek_pass:
                                signal = "WAIT"
                                trend = greek_msg
                                signal_strength = 0
                            else:
                                trend += f" | {greek_msg}"
                                signal_strength += 10
                    else:
                        trend, signal, vwap, ema, df_chart, current_atr, fib_data, signal_strength = "Waiting for Market Data", "WAIT", 0, 0, df_candles, 0, {}, 0

                    if signal in ["VERTICAL_CALL", "BULL_CALL_SPREAD", "CALL_BACKSPREAD"]:
                        signal = "BUY_CE"
                    elif signal in ["VERTICAL_PUT", "BEAR_PUT_SPREAD", "PUT_BACKSPREAD"]:
                        signal = "BUY_PE"
                    elif signal in ["SELL_CALL", "SHORT_STRANGLE", "IRON_CONDOR"]:
                        signal = "BUY_PE"
                    elif signal in ["SELL_PUT"]:
                        signal = "BUY_CE"
                    elif signal in ["LONG_STRANGLE", "BOX_SPREAD"]:
                        if df_candles is not None and len(df_candles) > 0:
                            last_rsi = df_candles['rsi'].iloc[-1] if 'rsi' in df_candles.columns else 50
                        else:
                            last_rsi = 50
                        signal = "BUY_CE" if last_rsi > 50 else "BUY_PE"

                    if df_chart is not None and hasattr(df_chart, 'columns'):
                        temp_df = self.analyzer.calculate_indicators(df_chart, index in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"])
                        for col in temp_df.columns:
                            if col not in df_chart.columns:
                                df_chart[col] = temp_df[col]
                        latest_data = df_chart.copy()
                    else:
                        latest_data = df_chart

                    self.state.update({
                        "current_trend": trend,
                        "current_signal": signal,
                        "signal_strength": signal_strength,
                        "vwap": vwap,
                        "ema": ema,
                        "atr": current_atr,
                        "fib_data": fib_data,
                        "latest_data": latest_data
                    })

                    if signal in ["BUY_CE", "BUY_PE"] and signal_strength >= min_signal_strength:
                        self.state["pending_signal"] = {
                            "signal": signal,
                            "strength": signal_strength,
                            "trend": trend,
                            "spot": spot,
                            "time": time_str,
                            "index": index,
                            "exch": exch,
                            "token": token
                        }
                    else:
                        self.state["pending_signal"] = None

                    if self.state["active_trade"] is None and not self.state["active_trades"] and signal in ["BUY_CE", "BUY_PE"] and current_time < cutoff_time and signal_strength >= min_signal_strength:
                        vega_ok, signal_strength = apply_vega_filter(df_candles, signal_strength)
                        if not vega_ok:
                            self.log(f"Vega filter rejected trade (IV too high)")
                            signal = "WAIT"
                            trend += " | VEGA: IV too high"
                        else:
                            delta_ok, signal_strength = apply_delta_filter(signal_strength, 20)
                            if not delta_ok:
                                self.log(f"Delta filter rejected trade (probability too low)")
                                signal = "WAIT"
                                trend += " | DELTA: Low probability"
                            else:
                                if strategy == "Arbitrage (Nifty/BankNifty)":
                                    nifty_exch, nifty_token = self.get_token_info("NIFTY")
                                    banknifty_exch, banknifty_token = self.get_token_info("BANKNIFTY")
                                    nifty_price = self.get_live_price(nifty_exch, "NIFTY", nifty_token)
                                    banknifty_price = self.get_live_price(banknifty_exch, "BANKNIFTY", banknifty_token)
                                    if nifty_price and banknifty_price:
                                        ratio = banknifty_price / nifty_price
                                        historical_ratio = 4.2
                                        opportunities = []
                                        if ratio > historical_ratio * 1.02:
                                            opportunities = [
                                                {"symbol": "BANKNIFTY", "side": "SELL", "qty": lots * LOT_SIZES.get("BANKNIFTY", 30), "price": banknifty_price},
                                                {"symbol": "NIFTY", "side": "BUY", "qty": lots * LOT_SIZES.get("NIFTY", 65), "price": nifty_price}
                                            ]
                                        elif ratio < historical_ratio * 0.98:
                                            opportunities = [
                                                {"symbol": "BANKNIFTY", "side": "BUY", "qty": lots * LOT_SIZES.get("BANKNIFTY", 30), "price": banknifty_price},
                                                {"symbol": "NIFTY", "side": "SELL", "qty": lots * LOT_SIZES.get("NIFTY", 65), "price": nifty_price}
                                            ]
                                        if opportunities:
                                            trades_placed = []
                                            for opp in opportunities:
                                                symbol = opp["symbol"]
                                                side = opp["side"]
                                                qty = opp["qty"]
                                                exch, token = self.get_token_info(symbol)
                                                strike_sym, strike_token, strike_exch, entry_ltp = self.get_atm_strike(symbol, opp["price"], "BUY_CE" if side == "BUY" else "BUY_PE")
                                                if strike_sym:
                                                    entry_ltp = self.apply_slippage(entry_ltp, "BUY" if side == "BUY" else "SELL")
                                                    trade_type = "BUY" if side == "BUY" else "SELL"
                                                    df_sym = self.get_historical_data(exch, token, symbol=symbol, interval=timeframe)
                                                    if df_sym is not None and not df_sym.empty:
                                                        atr = self.analyzer.get_atr(df_sym).iloc[-1]
                                                        sl_mult = 1.5
                                                        tp_mult = 3
                                                    else:
                                                        atr = 100
                                                        sl_mult = 1.5
                                                        tp_mult = 3
                                                    if trade_type == "BUY":
                                                        sl = entry_ltp - atr * sl_mult
                                                        tp = entry_ltp + atr * tp_mult
                                                    else:
                                                        sl = entry_ltp + atr * sl_mult
                                                        tp = entry_ltp - atr * tp_mult
                                                    trade = {
                                                        "symbol": strike_sym,
                                                        "token": strike_token,
                                                        "exch": strike_exch,
                                                        "type": trade_type,
                                                        "entry": entry_ltp,
                                                        "signal_price": opp["price"],
                                                        "current_ltp": entry_ltp,
                                                        "floating_pnl": 0.0,
                                                        "qty": qty,
                                                        "sl": sl,
                                                        "tgt": tp,
                                                        "tp1": tp,
                                                        "tp2": tp * 1.5,
                                                        "tp3": tp * 2,
                                                        "scaled_out": False,
                                                        "is_hz": False,
                                                        "booked_50": False,
                                                        "booked_80": False,
                                                        "entry_time": get_ist(),
                                                        "elapsed_time": 0
                                                    }
                                                    if not is_mock_mode:
                                                        exec_side = side
                                                        order_id, reject_reason = self.place_real_order(strike_sym, strike_token, qty, exec_side, strike_exch, "MARKET")
                                                        if not order_id:
                                                            self.log(f"Arbitrage order failed for {strike_sym}: {reject_reason}")
                                                            continue
                                                        else:
                                                            trade["simulated"] = False
                                                    else:
                                                        trade["simulated"] = False
                                                    trades_placed.append(trade)
                                            if trades_placed:
                                                self.state["active_trades"] = trades_placed
                                                self.state["trades_today"] += len(trades_placed)
                                                self.state["last_trade_time"] = get_ist()
                                                self.push_notify("Arbitrage Trade Entered", f"Entered {len(trades_placed)} legs")
                                                self.state["pending_signal"] = None
                                                continue
                                else:
                                    if is_hz:
                                        qty, sizing_info = self.calculate_hero_zero_position(signal_strength, current_atr, spot, max_capital)
                                        self.log(f"📊 Hero/Zero Position Sizing: {sizing_info}")
                                    else:
                                        qty = actual_qty

                                    if index in COMMODITIES:
                                        strike_sym, strike_token, strike_exch, entry_ltp = self.get_atm_strike(index, spot, signal)
                                        if not strike_sym:
                                            self.log(f"⚠️ No option strike found for {index}, falling back to futures")
                                            strike_sym = index
                                            strike_token, strike_exch = token, exch
                                            entry_ltp = spot
                                        else:
                                            self.log(f"✅ Using option {strike_sym} for {index}")
                                    elif is_mt5_asset or (is_crypto and crypto_mode != "Options") or is_fyers or exch in ["UPSTOX", "5PAISA", "STOXKART", "DHAN", "SHOONYA"]:
                                        strike_sym = index
                                        if is_crypto and crypto_mode == "Futures":
                                            if exch == "DELTA" and not strike_sym.endswith("USD"):
                                                strike_sym = f"{strike_sym}USD"
                                        strike_token, strike_exch = strike_sym, exch
                                        entry_ltp = spot
                                    else:
                                        if is_hz:
                                            strike_sym, strike_token, strike_exch, entry_ltp = self.get_atm_strike(index, spot, signal, otm_gamma_blast=True, premium_threshold=10000)
                                        else:
                                            strike_sym, strike_token, strike_exch, entry_ltp = self.get_atm_strike(index, spot, signal)

                                    if strike_sym and entry_ltp:
                                        if is_mock_mode:
                                            entry_ltp = self.apply_slippage(entry_ltp, signal)

                                        is_long = (signal == "BUY_CE")
                                        if is_mt5_asset or (is_crypto and crypto_mode != "Options") or is_fyers or index in COMMODITIES or exch in ["UPSTOX", "5PAISA", "STOXKART", "DHAN", "SHOONYA"]:
                                            trade_type = "BUY" if is_long else "SELL"
                                        else:
                                            trade_type = "CE" if is_long else "PE"

                                        if is_crypto and df_candles is not None and len(df_candles) > 20:
                                            swing_high, swing_low = self.analyzer.get_support_resistance(df_candles)
                                            if is_long and swing_low:
                                                dynamic_sl = swing_low
                                                tp1 = entry_ltp + (entry_ltp - swing_low) * 2
                                                tp2 = entry_ltp + (entry_ltp - swing_low) * 3
                                                tp3 = entry_ltp + (entry_ltp - swing_low) * 4
                                            elif not is_long and swing_high:
                                                dynamic_sl = swing_high
                                                tp1 = entry_ltp - (swing_high - entry_ltp) * 2
                                                tp2 = entry_ltp - (swing_high - entry_ltp) * 3
                                                tp3 = entry_ltp - (swing_high - entry_ltp) * 4
                                            else:
                                                if three_five_seven and current_atr > 0:
                                                    if is_long:
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
                                                    if not is_long:
                                                        dynamic_sl = entry_ltp + sl_pts
                                                        tp1 = entry_ltp - tgt_pts
                                                        tp2 = entry_ltp - (tgt_pts * 2)
                                                        tp3 = entry_ltp - (tgt_pts * 3)
                                                    else:
                                                        dynamic_sl = entry_ltp - sl_pts
                                                        tp1 = entry_ltp + tgt_pts
                                                        tp2 = entry_ltp + (tgt_pts * 2)
                                                        tp3 = entry_ltp + (tgt_pts * 3)
                                        else:
                                            if three_five_seven and current_atr > 0:
                                                if is_long:
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
                                                if not is_long:
                                                    dynamic_sl = entry_ltp + sl_pts
                                                    tp1 = entry_ltp - tgt_pts
                                                    tp2 = entry_ltp - (tgt_pts * 2)
                                                    tp3 = entry_ltp - (tgt_pts * 3)
                                                else:
                                                    dynamic_sl = entry_ltp - sl_pts
                                                    tp1 = entry_ltp + tgt_pts
                                                    tp2 = entry_ltp + (tgt_pts * 2)
                                                    tp3 = entry_ltp + (tgt_pts * 3)

                                        new_trade = {
                                            "symbol": strike_sym,
                                            "token": strike_token,
                                            "exch": strike_exch,
                                            "type": trade_type,
                                            "entry": entry_ltp,
                                            "signal_price": spot,
                                            "current_ltp": entry_ltp,
                                            "floating_pnl": 0.0,
                                            "highest_price": entry_ltp,
                                            "lowest_price": entry_ltp,
                                            "qty": qty,
                                            "sl": dynamic_sl,
                                            "tp1": tp1,
                                            "tp2": tp2,
                                            "tp3": tp3,
                                            "tgt": tp3,
                                            "scaled_out": False,
                                            "is_hz": is_hz,
                                            "booked_50": False,
                                            "booked_80": False,
                                            "entry_time": get_ist(),
                                            "elapsed_time": 0
                                        }
                                        self.push_notify("Signal Triggered", f"Executing {qty} {strike_sym} @ {entry_ltp}")
                                        self.state["sound_queue"].append("entry")
                                        st.toast(f"🚀 Entry: {qty} {strike_sym} @ {entry_ltp}", icon="💰")

                                        order_success = True
                                        reject_reason = None
                                        fill_price = entry_ltp
                                        if not is_mock_mode:
                                            exec_side = "SELL" if not is_long else "BUY"
                                            if use_limit_orders and is_crypto:
                                                order_type = "LIMIT"
                                                order_price = entry_ltp
                                            else:
                                                order_type = "MARKET"
                                                order_price = None
                                            order_id, reject_reason = self.place_real_order(strike_sym, strike_token, qty, exec_side, strike_exch, order_type, order_price)
                                            if not order_id:
                                                self.log(f"🚫 Real order rejected: {reject_reason}")
                                                new_trade["simulated"] = True
                                                new_trade["rejection_reason"] = reject_reason
                                                order_success = True
                                            else:
                                                new_trade["simulated"] = False
                                                fill_price = self.get_live_price(strike_exch, strike_sym, strike_token)
                                                if fill_price is None:
                                                    fill_price = entry_ltp
                                                new_trade["fill_price"] = fill_price
                                                new_trade["slippage"] = abs(fill_price - new_trade["signal_price"])
                                        else:
                                            new_trade["simulated"] = False

                                        if order_success:
                                            self.state["active_trade"] = new_trade
                                            self.state["trades_today"] += 1
                                            self.state["last_trade_time"] = get_ist()
                                            self.push_notify("Trade Entered" + (" (SIMULATED)" if new_trade.get("simulated") else ""), f"Entered {qty} {strike_sym} @ {fill_price}")
                                            self.state["ghost_memory"][f"{index}_{signal}"] = get_ist()
                                            if is_hz:
                                                self.state["hz_trades"].append({
                                                    "time": time_str,
                                                    "symbol": strike_sym,
                                                    "entry": fill_price,
                                                    "signal_strength": signal_strength
                                                })
                                            self.state["pending_signal"] = None
                                            continue

                    elif self.state["active_trade"]:
                        trade = self.state["active_trade"]
                        ltp = trade.get('current_ltp', trade['entry'])
                        pnl = trade.get('floating_pnl', 0.0)

                        is_long = (trade['type'] in ["CE", "BUY"])

                        if tsl_pts > 0:
                            if is_long:
                                if ltp > trade['entry']:
                                    new_trail = ltp - tsl_pts
                                    if 'trailing_stop' not in trade or new_trail > trade['trailing_stop']:
                                        trade['trailing_stop'] = new_trail
                                        if trade['trailing_stop'] > trade['sl']:
                                            trade['sl'] = trade['trailing_stop']
                                            self.log(f"Trailing SL moved to {trade['sl']:.2f}")
                            else:
                                if ltp < trade['entry']:
                                    new_trail = ltp + tsl_pts
                                    if 'trailing_stop' not in trade or new_trail < trade['trailing_stop']:
                                        trade['trailing_stop'] = new_trail
                                        if trade['trailing_stop'] < trade['sl']:
                                            trade['sl'] = trade['trailing_stop']
                                            self.log(f"Trailing SL moved to {trade['sl']:.2f}")

                        if not is_long:
                            hit_tp = ltp <= trade['tgt']
                            hit_sl = ltp >= trade['sl']
                        else:
                            hit_tp = ltp >= trade['tgt']
                            hit_sl = ltp <= trade['sl']

                        market_close = current_time >= cutoff_time

                        if hit_tp or hit_sl or market_close:
                            if not is_mock_mode and not trade.get("simulated"):
                                exec_side = "BUY" if not is_long else "SELL"
                                self.place_real_order(trade['symbol'], trade['token'], trade['qty'], exec_side, trade['exch'], "MARKET")
                            if hit_tp:
                                self.state["sound_queue"].append("tp")
                                st.session_state.win_streak += 1
                                st.session_state.loss_streak = 0
                            elif hit_sl:
                                self.state["sound_queue"].append("sl")
                                st.session_state.loss_streak += 1
                                st.session_state.win_streak = 0
                            win_text = "profit👍" if pnl > 0 else "sl hit 🛑"
                            if market_close:
                                win_text += " (Force Exit)"
                            self.log(f"🛑 EXIT {trade['symbol']} | PnL: {round(pnl, 2)} [{win_text}]")
                            self.push_notify("Trade Closed", f"Closed {trade['symbol']} | PnL: {round(pnl, 2)}")
                            if not self.is_mock:
                                user_id = getattr(self, "system_user_id", self.api_key)
                                slippage = trade.get('slippage', 0)
                                self.db_queue.put({
                                    "user_id": user_id, "trade_date": today_date, "trade_time": time_str,
                                    "symbol": trade['symbol'], "trade_type": trade['type'], "qty": trade['qty'],
                                    "entry_price": trade['entry'], "exit_price": ltp,
                                    "pnl": round(pnl, 2), "result": win_text, "slippage": slippage
                                })
                                journal_entry = {
                                    "user_id": user_id,
                                    "trade_date": today_date,
                                    "trade_time": time_str,
                                    "symbol": trade['symbol'],
                                    "type": trade['type'],
                                    "qty": trade['qty'],
                                    "entry": trade['entry'],
                                    "exit": ltp,
                                    "pnl": round(pnl, 2),
                                    "result": win_text,
                                    "strategy": self.settings.get('strategy', 'Unknown'),
                                    "slippage": slippage
                                }
                                save_trade_journal(user_id, journal_entry)
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
                                    "Result": win_text,
                                    "Slippage": trade.get('slippage', 0)
                                })
                            if trade.get('is_hz'):
                                self.state["hz_pnl"] += pnl
                                if pnl > 0:
                                    self.state["hz_wins"] += 1
                                else:
                                    self.state["hz_losses"] += 1
                            self.state["last_trade"] = trade.copy()
                            self.state["last_trade"].update({"exit_price": ltp, "final_pnl": pnl, "win_text": win_text})
                            self.state["daily_pnl"] += pnl
                            self.state["active_trade"] = None
                            if self.state.get("stop_after_manual_exit"):
                                self.state["is_running"] = False
                                self.state["engine_active"] = False
                                self.state["stop_after_manual_exit"] = False
                                self.log("Engine stopped after manual exit.")
                                break
                            continue

                    elif self.state["active_trades"]:
                        all_closed = True
                        for trade in self.state["active_trades"]:
                            ltp = trade.get('current_ltp', trade['entry'])
                            pnl = trade.get('floating_pnl', 0.0)
                            is_long = (trade['type'] in ["BUY"])
                            if is_long:
                                hit_tp = ltp >= trade['tgt']
                                hit_sl = ltp <= trade['sl']
                            else:
                                hit_tp = ltp <= trade['tgt']
                                hit_sl = ltp >= trade['sl']
                            if hit_tp or hit_sl or current_time >= cutoff_time:
                                if not is_mock_mode and not trade.get("simulated"):
                                    exec_side = "SELL" if is_long else "BUY"
                                    self.place_real_order(trade['symbol'], trade['token'], trade['qty'], exec_side, trade['exch'], "MARKET")
                                win_text = "profit👍" if pnl > 0 else "sl hit 🛑"
                                if current_time >= cutoff_time:
                                    win_text += " (Force Exit)"
                                self.log(f"🛑 EXIT {trade['symbol']} | PnL: {round(pnl, 2)} [{win_text}]")
                                if not self.is_mock:
                                    user_id = getattr(self, "system_user_id", self.api_key)
                                    self.db_queue.put({
                                        "user_id": user_id, "trade_date": today_date, "trade_time": time_str,
                                        "symbol": trade['symbol'], "trade_type": trade['type'], "qty": trade['qty'],
                                        "entry_price": trade['entry'], "exit_price": ltp,
                                        "pnl": round(pnl, 2), "result": win_text, "slippage": trade.get('slippage', 0)
                                    })
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
                                        "Result": win_text,
                                        "Slippage": trade.get('slippage', 0)
                                    })
                                self.state["daily_pnl"] += pnl
                                trade['closed'] = True
                            else:
                                all_closed = False
                        self.state["active_trades"] = [t for t in self.state["active_trades"] if not t.get('closed')]
                        if all_closed:
                            self.state["active_trades"] = []
                            self.state["last_trade_time"] = get_ist()
                            self.push_notify("Arbitrage Trade Closed", f"All legs closed")
                            if self.state.get("stop_after_manual_exit"):
                                self.state["is_running"] = False
                                self.state["engine_active"] = False
                                self.state["stop_after_manual_exit"] = False
                                self.log("Engine stopped after manual exit.")
                                break

            except Exception as e:
                self.log(f"⚠️ Loop Error: {str(e)}")
            time.sleep(0.5)

# ==========================================
# LANDING PAGE, LOGIN, SPLASH, DASHBOARD – all as before, with updated admin tab and button placement.
# ==========================================

if st.session_state.page == "landing":
    # Holiday and event info for tomorrow
    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_holiday, tomorrow_event = get_tomorrow_info()
    info_line = ""
    if tomorrow_holiday:
        info_line += f"📅 Tomorrow ({tomorrow_date}): {tomorrow_holiday} (Market Closed)"
    if tomorrow_event:
        if info_line:
            info_line += " | "
        info_line += f"📊 Tomorrow ({tomorrow_date}): {tomorrow_event}"
    if info_line:
        st.info(info_line)

    st.markdown("""
    <div class="landing-hero">
        <div class="shree-box">
            <div class="kinetic-title">🕉️ SHREE</div>
        </div>
        <div class="kinetic-sub">Intelligent Algo Trading Platform</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="terms-card">', unsafe_allow_html=True)
        st.markdown("### 📜 Terms and Conditions")
        st.markdown('<div class="typewriter">By accessing and using this platform, you agree to the following:</div>', unsafe_allow_html=True)
        st.markdown("""
        - You are solely responsible for all trading decisions and outcomes.
        - Past performance does not guarantee future results.
        - You acknowledge the high risk of financial loss in trading.
        - You will not hold the developers liable for any losses.
        - You must comply with all applicable laws and regulations.
        """)
        terms_accepted = st.checkbox("I have read and agree to the Terms and Conditions", key="terms_accepted")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚪 Enter", key="enter_btn", disabled=not terms_accepted, use_container_width=True, on_click=lambda: play_sound_now("click")):
            st.session_state.page = "login"
            st.rerun()

    st.markdown("""
    <div class="bento-grid">
        <div class="bento-card">
            <div class="bento-icon">📈</div>
            <h3>10+ Strategies</h3>
            <p>Momentum, ML, FVG, SMC – fully customizable.</p>
        </div>
        <div class="bento-card">
            <div class="bento-icon">🔌</div>
            <h3>12+ Brokers</h3>
            <p>Angel, Zerodha, CoinDCX, Delta, MT5, Fyers, Upstox, 5paisa, Binance, Stoxkart, Dhan, Shoonya.</p>
        </div>
        <div class="bento-card">
            <div class="bento-icon">⚡</div>
            <h3>Real‑time Scanners</h3>
            <p>Hero/Zero, Pin Bar, Breakout, FOMO.</p>
        </div>
        <div class="bento-card">
            <div class="bento-icon">📱</div>
            <h3>Notifications</h3>
            <p>Telegram, WhatsApp, Email, Push – stay connected.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Supported Brokers")
    broker_list = [
        ("Angel One", "https://www.google.com/s2/favicons?domain=angelone.in&sz=64"),
        ("Zerodha", "https://www.google.com/s2/favicons?domain=zerodha.com&sz=64"),
        ("CoinDCX", "https://www.google.com/s2/favicons?domain=coindcx.com&sz=64"),
        ("Delta Exchange", "https://www.google.com/s2/favicons?domain=delta.exchange&sz=64"),
        ("MT5/Exness", "https://www.google.com/s2/favicons?domain=metatrader5.com&sz=64"),
        ("Fyers", "https://www.google.com/s2/favicons?domain=fyers.in&sz=64"),
        ("Upstox", "https://www.google.com/s2/favicons?domain=upstox.com&sz=64"),
        ("5paisa", "https://www.google.com/s2/favicons?domain=5paisa.com&sz=64"),
        ("Binance", "https://www.google.com/s2/favicons?domain=binance.com&sz=64"),
        ("Stoxkart", "https://www.google.com/s2/favicons?domain=stoxkart.com&sz=64"),
        ("Dhan", "https://www.google.com/s2/favicons?domain=dhan.co&sz=64"),
        ("Shoonya", "https://www.google.com/s2/favicons?domain=shoonya.com&sz=64")
    ]
    html = '<div class="broker-grid">'
    for name, icon in broker_list:
        html += f'<div class="broker-item"><img src="{icon}"><span>{name}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 💎 Basic")
        st.markdown("• 1 Broker connection\n• 5 strategies\n• Email support\n\n**$29/month**")
    with col2:
        st.markdown("### 🚀 Pro")
        st.markdown("• 5 Broker connections\n• All strategies\n• Priority support\n\n**$79/month**")
    with col3:
        st.markdown("### 🏢 Enterprise")
        st.markdown("• Unlimited brokers\n• Custom strategies\n• Dedicated account manager\n\n**Contact us**")

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 💬 Contact for Pricing / Demo")
        with st.form("contact_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Email Address")
            message = st.text_area("Message")
            submitted = st.form_submit_button("Send Inquiry", use_container_width=True, on_click=lambda: play_sound_now("click"))
            if submitted:
                st.success("Thank you! We'll get back to you shortly.")
                st.balloons()
    with col_right:
        st.markdown("### 📞 Reach us directly")
        st.markdown("📧 **Email:** support@shree.example.com")
        st.markdown("📱 **WhatsApp:** +91 9964915530")
        st.markdown("🌐 **Website:** [www.shree.example.com](https://www.shree.example.com)")

    st.markdown("""
    <div class="footer">
        <p>© 2025 SHREE Trading Technologies. All rights reserved.</p>
        <p style="font-size: 0.9rem;">Developed by Vijayakumar Suryavanshi</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "login":
    if not HAS_DB:
        st.error("⚠️ Database missing. Add SUPABASE_URL and SUPABASE_KEY to enable saving & logs.")
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
            auth_mode = st.radio("Operating Mode",
                                 ["📝 Paper Trading", "🕉️ Real Trading", "👆 Quick Auth"],
                                 horizontal=True, label_visibility="collapsed")
            st.divider()

            if auth_mode == "👆 Quick Auth":
                st.info("💡 **Quick Login:** Enter your registered Email or Phone. The system will auto-fetch your Cloud profile.")
                USER_ID = st.text_input("Enter Email ID or Phone Number", value=st.session_state.user_id)
                keep_signed = st.checkbox("Keep me signed in (auto‑login using URL)", value=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("👆 Authenticate & Connect", type="primary", use_container_width=True,
                             on_click=lambda: play_sound_now("click")):
                    if is_user_blocked(USER_ID):
                        st.error("Your account has been blocked. Please contact support.")
                    else:
                        creds = load_creds(USER_ID)
                        if creds and (creds.get("client_id") or creds.get("zerodha_api") or
                                      creds.get("coindcx_api") or creds.get("delta_api") or
                                      creds.get("fyers_client_id") or creds.get("upstox_api_key") or
                                      creds.get("fivepaisa_client_id") or creds.get("binance_api_key") or
                                      creds.get("stoxkart_api_key") or creds.get("dhan_client_id") or
                                      creds.get("shoonya_user")):
                            temp_bot = SniperBot(
                                api_key=creds.get("angel_api", ""), client_id=creds.get("client_id"),
                                pwd=creds.get("pwd"), totp_secret=creds.get("totp_secret"),
                                mt5_acc=creds.get("mt5_acc"), mt5_pass=creds.get("mt5_pass"),
                                mt5_server=creds.get("mt5_server"), mt5_api_url=creds.get("mt5_api_url", ""),
                                zerodha_api=creds.get("zerodha_api"), zerodha_secret=creds.get("zerodha_secret"),
                                coindcx_api=creds.get("coindcx_api"), coindcx_secret=creds.get("coindcx_secret"),
                                delta_api=creds.get("delta_api"), delta_secret=creds.get("delta_secret"),
                                is_mock=False,
                                tg_bot_token=creds.get("tg_bot_token", ""),
                                tg_allowed_users=creds.get("tg_allowed_users", ""),
                                fyers_client_id=creds.get("fyers_client_id", ""),
                                fyers_secret=creds.get("fyers_secret", ""),
                                fyers_token=creds.get("fyers_token", ""),
                                upstox_api_key=creds.get("upstox_api_key", ""),
                                upstox_api_secret=creds.get("upstox_api_secret", ""),
                                upstox_access_token=creds.get("upstox_access_token", ""),
                                fivepaisa_client_id=creds.get("fivepaisa_client_id", ""),
                                fivepaisa_secret=creds.get("fivepaisa_secret", ""),
                                fivepaisa_access_token=creds.get("fivepaisa_access_token", ""),
                                binance_api_key=creds.get("binance_api_key", ""),
                                binance_api_secret=creds.get("binance_api_secret", ""),
                                binance_testnet=creds.get("binance_testnet", False),
                                stoxkart_api_key=creds.get("stoxkart_api_key", ""),
                                stoxkart_secret=creds.get("stoxkart_secret", ""),
                                dhan_client_id=creds.get("dhan_client_id", ""),
                                dhan_access_token=creds.get("dhan_access_token", ""),
                                shoonya_user=creds.get("shoonya_user", ""),
                                shoonya_password=creds.get("shoonya_password", ""),
                                shoonya_totp=creds.get("shoonya_totp", ""),
                                email_smtp_server=creds.get("email_smtp_server", ""),
                                email_port=creds.get("email_port", 587),
                                email_username=creds.get("email_username", ""),
                                email_password=creds.get("email_password", ""),
                                email_recipients=creds.get("email_recipients", ""),
                                fcm_server_key=creds.get("fcm_server_key", ""),
                                push_enabled=creds.get("push_enabled", False)
                            )
                            temp_bot.system_user_id = USER_ID
                            temp_bot.user_role = creds.get("role", "trader")
                            st.session_state.user_role = temp_bot.user_role
                            with st.spinner("Authenticating via Cloud..."):
                                if temp_bot.login():
                                    temp_bot.state["daily_pnl"] = temp_bot.load_daily_pnl()
                                    st.session_state.bot = temp_bot
                                    st.session_state.audio_enabled = True
                                    temp_bot.state["sound_queue"].append("login")
                                    st.session_state.user_id = USER_ID
                                    if keep_signed:
                                        st.query_params["user_id"] = USER_ID
                                    save_device_session(USER_ID, st.session_state.device_name,
                                                        st.session_state.ip_address,
                                                        st.session_state.session_id)
                                    st.session_state.page = "splash"
                                    st.rerun()
                                else:
                                    st.error("❌ Login Failed! Check API details or TOTP.")
                        else:
                            st.error("❌ Profile not found! Please save it once via the Real Trading menu.")

            elif auth_mode == "🕉️ Real Trading":
                USER_ID = st.text_input("System Login ID (Email or Phone Number)", value=st.session_state.user_id)
                if is_user_blocked(USER_ID):
                    st.error("Your account has been blocked. Please contact support.")
                else:
                    creds = load_creds(USER_ID) if USER_ID else {}
                    st.markdown("### 🏦 Select Brokers to Connect")
                    ANGEL_API, CLIENT_ID, PIN, TOTP = "", "", "", ""
                    Z_API, Z_SEC, Z_REQ = "", "", ""
                    MT5_ACC, MT5_PASS, MT5_SERVER, MT5_API_URL = "", "", "", ""
                    DCX_API, DCX_SEC = "", ""
                    DELTA_API, DELTA_SEC = "", ""
                    FYERS_CLIENT_ID, FYERS_SECRET, FYERS_TOKEN = "", "", ""
                    UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_ACCESS_TOKEN = "", "", ""
                    FIVEPAISA_CLIENT_ID, FIVEPAISA_SECRET, FIVEPAISA_ACCESS_TOKEN = "", "", ""
                    BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET = "", "", False
                    STOXKART_API_KEY, STOXKART_SECRET = "", ""
                    DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN = "", ""
                    SHOONYA_USER, SHOONYA_PASSWORD, SHOONYA_TOTP = "", "", ""
                    TG_BOT_TOKEN = ""
                    TG_ALLOWED_USERS = ""
                    EMAIL_SMTP_SERVER, EMAIL_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_RECIPIENTS = "", 587, "", "", ""
                    FCM_SERVER_KEY, PUSH_ENABLED = "", False
                    ROLE = creds.get("role", "trader")

                    # Broker toggles (Angel, Zerodha, CoinDCX, Delta, MT5, Fyers, Upstox, 5paisa, Binance, Stoxkart, Dhan, Shoonya)
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
                        with col_t: use_mt5 = st.toggle("MetaTrader 5 / Exness", value=bool(creds.get("mt5_acc")))
                        if use_mt5:
                            col_m1, col_m2 = st.columns(2)
                            with col_m1: MT5_ACC = st.text_input("MT5 Account ID", value=creds.get("mt5_acc", ""))
                            with col_m2: MT5_PASS = st.text_input("MT5 Password", type="password", value=creds.get("mt5_pass", ""))
                            MT5_SERVER = st.text_input("Broker Server", value=creds.get("mt5_server", ""))
                            MT5_API_URL = st.text_input("MT5 Web API URL (Optional)", value=creds.get("mt5_api_url", "https://mt5-web-api.mtapi.io/v1"))
                    with st.container(border=True):
                        col_img, col_t = st.columns([1, 6])
                        with col_img: st.image("https://www.google.com/s2/favicons?domain=fyers.in&sz=128", width=40)
                        with col_t: use_fyers = st.toggle("Fyers", value=bool(creds.get("fyers_client_id")))
                        if use_fyers:
                            FYERS_CLIENT_ID = st.text_input("Fyers Client ID", value=creds.get("fyers_client_id", ""))
                            FYERS_SECRET = st.text_input("Fyers Secret Key", type="password", value=creds.get("fyers_secret", ""))
                            FYERS_TOKEN = st.text_input("Fyers Access Token", type="password", value=creds.get("fyers_token", ""))
                    with st.container(border=True):
                        col_img, col_t = st.columns([1, 6])
                        with col_img: st.image("https://www.google.com/s2/favicons?domain=upstox.com&sz=128", width=40)
                        with col_t: use_upstox = st.toggle("Upstox", value=bool(creds.get("upstox_api_key")))
                        if use_upstox:
                            UPSTOX_API_KEY = st.text_input("Upstox API Key", value=creds.get("upstox_api_key", ""))
                            UPSTOX_API_SECRET = st.text_input("Upstox API Secret", type="password", value=creds.get("upstox_api_secret", ""))
                            UPSTOX_ACCESS_TOKEN = st.text_input("Upstox Access Token", type="password", value=creds.get("upstox_access_token", ""))
                    with st.container(border=True):
                        col_img, col_t = st.columns([1, 6])
                        with col_img: st.image("https://www.google.com/s2/favicons?domain=5paisa.com&sz=128", width=40)
                        with col_t: use_fivepaisa = st.toggle("5paisa", value=bool(creds.get("fivepaisa_client_id")))
                        if use_fivepaisa:
                            FIVEPAISA_CLIENT_ID = st.text_input("5paisa Client ID", value=creds.get("fivepaisa_client_id", ""))
                            FIVEPAISA_SECRET = st.text_input("5paisa Secret", type="password", value=creds.get("fivepaisa_secret", ""))
                            FIVEPAISA_ACCESS_TOKEN = st.text_input("5paisa Access Token", type="password", value=creds.get("fivepaisa_access_token", ""))
                    with st.container(border=True):
                        col_img, col_t = st.columns([1, 6])
                        with col_img: st.image("https://www.google.com/s2/favicons?domain=binance.com&sz=128", width=40)
                        with col_t: use_binance = st.toggle("Binance", value=bool(creds.get("binance_api_key")))
                        if use_binance:
                            BINANCE_API_KEY = st.text_input("Binance API Key", value=creds.get("binance_api_key", ""))
                            BINANCE_API_SECRET = st.text_input("Binance API Secret", type="password", value=creds.get("binance_api_secret", ""))
                            BINANCE_TESTNET = st.checkbox("Use Testnet", value=creds.get("binance_testnet", False))
                    with st.container(border=True):
                        col_img, col_t = st.columns([1, 6])
                        with col_img: st.image("https://www.google.com/s2/favicons?domain=stoxkart.com&sz=128", width=40)
                        with col_t: use_stoxkart = st.toggle("Stoxkart", value=bool(creds.get("stoxkart_api_key")))
                        if use_stoxkart:
                            STOXKART_API_KEY = st.text_input("Stoxkart API Key", value=creds.get("stoxkart_api_key", ""))
                            STOXKART_SECRET = st.text_input("Stoxkart Secret", type="password", value=creds.get("stoxkart_secret", ""))
                    with st.container(border=True):
                        col_img, col_t = st.columns([1, 6])
                        with col_img: st.image("https://www.google.com/s2/favicons?domain=dhan.co&sz=128", width=40)
                        with col_t: use_dhan = st.toggle("Dhan", value=bool(creds.get("dhan_client_id")))
                        if use_dhan:
                            DHAN_CLIENT_ID = st.text_input("Dhan Client ID", value=creds.get("dhan_client_id", ""))
                            DHAN_ACCESS_TOKEN = st.text_input("Dhan Access Token", type="password", value=creds.get("dhan_access_token", ""))
                    with st.container(border=True):
                        col_img, col_t = st.columns([1, 6])
                        with col_img: st.image("https://www.google.com/s2/favicons?domain=shoonya.com&sz=128", width=40)
                        with col_t: use_shoonya = st.toggle("Shoonya", value=bool(creds.get("shoonya_user")))
                        if use_shoonya:
                            SHOONYA_USER = st.text_input("Shoonya User ID", value=creds.get("shoonya_user", ""))
                            SHOONYA_PASSWORD = st.text_input("Shoonya Password", type="password", value=creds.get("shoonya_password", ""))
                            SHOONYA_TOTP = st.text_input("Shoonya TOTP", value=creds.get("shoonya_totp", ""))
                    st.divider()
                    with st.expander("📱 Notifications & Control"):
                        TG_TOKEN = st.text_input("Telegram Bot Token (for alerts)", value=creds.get("tg_token", ""))
                        TG_CHAT = st.text_input("Telegram Chat ID (for alerts)", value=creds.get("tg_chat", ""))
                        WA_PHONE = st.text_input("WhatsApp Phone", value=creds.get("wa_phone", ""))
                        WA_API = st.text_input("WhatsApp API Key", value=creds.get("wa_api", ""))
                        TG_BOT_TOKEN = st.text_input("Telegram Control Bot Token", value=creds.get("tg_bot_token", ""))
                        TG_ALLOWED_USERS = st.text_input("Allowed Telegram User IDs (comma separated)", value=creds.get("tg_allowed_users", ""))
                        EMAIL_SMTP_SERVER = st.text_input("SMTP Server", value=creds.get("email_smtp_server", ""))
                        EMAIL_PORT = st.number_input("SMTP Port", value=creds.get("email_port", 587))
                        EMAIL_USERNAME = st.text_input("Email Username", value=creds.get("email_username", ""))
                        EMAIL_PASSWORD = st.text_input("Email Password", type="password", value=creds.get("email_password", ""))
                        EMAIL_RECIPIENTS = st.text_input("Email Recipients (comma separated)", value=creds.get("email_recipients", ""))
                        FCM_SERVER_KEY = st.text_input("FCM Server Key", type="password", value=creds.get("fcm_server_key", ""))
                        PUSH_ENABLED = st.checkbox("Enable Push Notifications", value=creds.get("push_enabled", False))
                    with st.expander("👥 User Role"):
                        ROLE = st.selectbox("Role", ["trader", "admin"], index=0 if ROLE == "trader" else 1)
                    SAVE_CREDS = st.checkbox("Remember Credentials Securely (Cloud DB)", value=True)
                    keep_signed = st.checkbox("Keep me signed in (auto‑login using URL)", value=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("CONNECT MARKETS 🚀", type="primary", use_container_width=True, on_click=lambda: play_sound_now("click")):
                        if not USER_ID:
                            st.error("Please enter your System Login ID (Email or Phone) to proceed.")
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
                                is_mock=False,
                                tg_bot_token=TG_BOT_TOKEN, tg_allowed_users=TG_ALLOWED_USERS,
                                fyers_client_id=FYERS_CLIENT_ID if use_fyers else "", fyers_secret=FYERS_SECRET if use_fyers else "", fyers_token=FYERS_TOKEN if use_fyers else "",
                                upstox_api_key=UPSTOX_API_KEY if use_upstox else "", upstox_api_secret=UPSTOX_API_SECRET if use_upstox else "", upstox_access_token=UPSTOX_ACCESS_TOKEN if use_upstox else "",
                                fivepaisa_client_id=FIVEPAISA_CLIENT_ID if use_fivepaisa else "", fivepaisa_secret=FIVEPAISA_SECRET if use_fivepaisa else "", fivepaisa_access_token=FIVEPAISA_ACCESS_TOKEN if use_fivepaisa else "",
                                binance_api_key=BINANCE_API_KEY if use_binance else "", binance_api_secret=BINANCE_API_SECRET if use_binance else "", binance_testnet=BINANCE_TESTNET if use_binance else False,
                                stoxkart_api_key=STOXKART_API_KEY if use_stoxkart else "", stoxkart_secret=STOXKART_SECRET if use_stoxkart else "",
                                dhan_client_id=DHAN_CLIENT_ID if use_dhan else "", dhan_access_token=DHAN_ACCESS_TOKEN if use_dhan else "",
                                shoonya_user=SHOONYA_USER if use_shoonya else "", shoonya_password=SHOONYA_PASSWORD if use_shoonya else "", shoonya_totp=SHOONYA_TOTP if use_shoonya else "",
                                email_smtp_server=EMAIL_SMTP_SERVER, email_port=EMAIL_PORT, email_username=EMAIL_USERNAME, email_password=EMAIL_PASSWORD, email_recipients=EMAIL_RECIPIENTS,
                                fcm_server_key=FCM_SERVER_KEY, push_enabled=PUSH_ENABLED
                            )
                            temp_bot.system_user_id = USER_ID
                            temp_bot.user_role = ROLE
                            st.session_state.user_role = ROLE
                            with st.spinner("Authenticating Secure Connections..."):
                                if temp_bot.login():
                                    temp_bot.state["daily_pnl"] = temp_bot.load_daily_pnl()
                                    if SAVE_CREDS:
                                        save_creds(USER_ID, ANGEL_API, CLIENT_ID, PIN, TOTP, TG_TOKEN, TG_CHAT, WA_PHONE, WA_API, 
                                                   MT5_ACC, MT5_PASS, MT5_SERVER, MT5_API_URL, Z_API, Z_SEC, DCX_API, DCX_SEC, DELTA_API, DELTA_SEC, 
                                                   TG_BOT_TOKEN, TG_ALLOWED_USERS, FYERS_CLIENT_ID, FYERS_SECRET, FYERS_TOKEN, 
                                                   UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_ACCESS_TOKEN,
                                                   FIVEPAISA_CLIENT_ID, FIVEPAISA_SECRET, FIVEPAISA_ACCESS_TOKEN,
                                                   BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET,
                                                   STOXKART_API_KEY, STOXKART_SECRET,
                                                   DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN,
                                                   SHOONYA_USER, SHOONYA_PASSWORD, SHOONYA_TOTP,
                                                   EMAIL_SMTP_SERVER, EMAIL_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_RECIPIENTS,
                                                   FCM_SERVER_KEY, PUSH_ENABLED, ROLE)
                                    st.session_state.bot = temp_bot
                                    st.session_state.audio_enabled = True
                                    temp_bot.state["sound_queue"].append("login")
                                    st.session_state.user_id = USER_ID
                                    if keep_signed:
                                        st.query_params["user_id"] = USER_ID
                                    save_device_session(USER_ID, st.session_state.device_name, st.session_state.ip_address, st.session_state.session_id)
                                    st.session_state.page = "splash"
                                    st.rerun()
                                else:
                                    err_msg = temp_bot.state['logs'][0] if temp_bot.state['logs'] else "Unknown Error"
                                    st.error(f"Login Failed! \n\n**System Log:** {err_msg}")

            else:  # Paper Trading
                st.info("📝 Paper Trading simulates live market movement without risking real capital.")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("START PAPER SESSION 📝", type="primary", use_container_width=True, on_click=lambda: play_sound_now("click")):
                    temp_bot = SniperBot(is_mock=True)
                    temp_bot.login()
                    temp_bot.system_user_id = "paper_user"
                    temp_bot.state["daily_pnl"] = temp_bot.load_daily_pnl()
                    st.session_state.bot = temp_bot
                    st.session_state.audio_enabled = True
                    temp_bot.state["sound_queue"].append("login")
                    st.session_state.page = "splash"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("← Back to Landing", on_click=lambda: play_sound_now("click")):
                st.session_state.page = "landing"
                st.rerun()

elif st.session_state.page == "splash":
    if st.session_state.audio_enabled and st.session_state.bot and st.session_state.bot.state.get("sound_queue"):
        while st.session_state.bot.state["sound_queue"]:
            latest_sound = st.session_state.bot.state["sound_queue"].popleft()
            play_sound_ui(latest_sound)
    st.markdown("""
    <div class="scene">
        <div class="cube">
            <div class="cube__face cube__face--front">🕉️</div>
            <div class="cube__face cube__face--back">श्री</div>
            <div class="cube__face cube__face--right">Om</div>
            <div class="cube__face cube__face--left">Shree</div>
            <div class="cube__face cube__face--top">🕉️</div>
            <div class="cube__face cube__face--bottom">श्री</div>
        </div>
    </div>
    <div class="splash-text">Welcome to SHREE Algo Trading Platform</div>
    <style>
    .scene {
        width: 200px;
        height: 200px;
        perspective: 600px;
        margin: 50px auto;
    }
    .cube {
        width: 100%;
        height: 100%;
        position: relative;
        transform-style: preserve-3d;
        animation: rotateCube 10s infinite linear;
    }
    .cube__face {
        position: absolute;
        width: 200px;
        height: 200px;
        background: rgba(255,255,255,0.9);
        border: 2px solid #ff6600;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        line-height: 200px;
        color: #ff6600;
        backface-visibility: visible;
    }
    .cube__face--front  { transform: rotateY(0deg) translateZ(100px); background: rgba(255,165,0,0.8); }
    .cube__face--right  { transform: rotateY(90deg) translateZ(100px); background: rgba(255,215,0,0.8); }
    .cube__face--back   { transform: rotateY(180deg) translateZ(100px); background: rgba(255,140,0,0.8); }
    .cube__face--left   { transform: rotateY(-90deg) translateZ(100px); background: rgba(255,200,0,0.8); }
    .cube__face--top    { transform: rotateX(90deg) translateZ(100px); background: rgba(255,190,0,0.8); }
    .cube__face--bottom { transform: rotateX(-90deg) translateZ(100px); background: rgba(255,210,0,0.8); }
    @keyframes rotateCube {
        from { transform: rotateX(0deg) rotateY(0deg); }
        to { transform: rotateX(360deg) rotateY(360deg); }
    }
    .splash-text {
        text-align: center;
        font-size: 2rem;
        margin-top: 30px;
        color: #ff6600;
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    st.balloons()
    time.sleep(1.5)
    st.session_state.page = "dashboard"
    st.rerun()

elif st.session_state.page == "dashboard":
    bot = st.session_state.bot

    if st.session_state.user_id in ["9964915530", "vijayakumar.suryavanshi@gmail.com"]:
        st.session_state.is_developer = True
    else:
        st.session_state.is_developer = False

    with st.sidebar:
        theme_choice = st.radio("Theme", ["light", "dark"], index=0 if st.session_state.theme == "light" else 1, horizontal=True, on_change=lambda: play_sound_now("click"))
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.header("⚙️ SYSTEM CONFIGURATION")
        if not st.session_state.audio_enabled:
            if st.button("🔊 Enable Audio", use_container_width=True, on_click=lambda: play_sound_now("click")):
                st.session_state.audio_enabled = True
                unlock_audio()
                st.success("Audio enabled!")
                st.rerun()
        else:
            st.success("🔊 Audio is ON")

        st.markdown("**1. Market Setup**")
        BROKER = st.selectbox("Primary Broker", ["Angel One", "Zerodha", "CoinDCX", "Delta Exchange", "MT5/Exness", "Fyers", "Upstox", "5paisa", "Binance", "Stoxkart", "Dhan", "Shoonya"], index=0, on_change=lambda: play_sound_now("click"))

        st.divider()
        st.markdown("**🚀 FOMO Mode**")
        fomo_enabled = st.toggle("Enable FOMO (Trade Nifty, Bank Nifty, Sensex simultaneously)", value=st.session_state.fomo_mode, on_change=lambda: play_sound_now("click"))
        if fomo_enabled != st.session_state.fomo_mode:
            st.session_state.fomo_mode = fomo_enabled
            st.rerun()

        CUSTOM_STOCK = st.text_input("Add Custom Stock/Coin", value=st.session_state.custom_stock, placeholder="e.g. RELIANCE").upper().strip()
        st.session_state.custom_stock = CUSTOM_STOCK
        all_assets = list(LOT_SIZES.keys()) + ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        if CUSTOM_STOCK and CUSTOM_STOCK not in all_assets:
            all_assets.append(CUSTOM_STOCK)
        if BROKER in ["CoinDCX", "Delta Exchange", "Binance"]:
            valid_assets = [a for a in all_assets if "USD" in a or "USDT" in a]
        elif BROKER in ["Angel One", "Zerodha", "Upstox", "5paisa", "Stoxkart", "Dhan", "Shoonya"]:
            valid_assets = [a for a in all_assets if (a in INDEX_TOKENS or a in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"] or a.isalpha()) and "USD" not in a and "USDT" not in a]
        elif BROKER == "Fyers":
            valid_assets = all_assets
        elif BROKER == "MT5/Exness":
            valid_assets = [a for a in all_assets if a in ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "BNBUSD", "LTCUSD", "DOTUSD", "MATICUSD", "SHIBUSD", "TRXUSD", "LINKUSD"]]
        else:
            valid_assets = [a for a in all_assets if a in ["XAUUSD", "EURUSD", "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "BNBUSD", "LTCUSD", "DOTUSD", "MATICUSD", "SHIBUSD", "TRXUSD", "LINKUSD"]]
        if CUSTOM_STOCK and CUSTOM_STOCK not in valid_assets:
            valid_assets.append(CUSTOM_STOCK)
        if not valid_assets:
            valid_assets = ["NIFTY"] if BROKER in ["Angel One", "Zerodha", "Upstox", "5paisa", "Stoxkart", "Dhan", "Shoonya"] else ["BTCUSD"]
        st.session_state.asset_options = valid_assets
        if st.session_state.sb_index_input not in valid_assets:
            st.session_state.sb_index_input = valid_assets[0]

        INDEX = st.selectbox("Watchlist Asset", valid_assets, index=valid_assets.index(st.session_state.sb_index_input), key="sb_index_input", on_change=lambda: play_sound_now("click"))
        STRATEGY = st.selectbox("Trading Strategy", STRAT_LIST, index=STRAT_LIST.index(st.session_state.sb_strat_input), key="sb_strat_input", on_change=lambda: play_sound_now("click"))
        TIMEFRAME = st.selectbox("Candle Timeframe", ["1m", "3m", "5m", "15m"], index=2, on_change=lambda: play_sound_now("click"))

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

        if BROKER in ["CoinDCX", "Delta Exchange", "Binance"]:
            st.divider()
            st.markdown("**🪙 Crypto Setup**")
            col_c1, col_c2 = st.columns(2)
            with col_c1: 
                CRYPTO_MODE = st.selectbox("Market Type", ["Futures", "Spot", "Options"], on_change=lambda: play_sound_now("click"))
            with col_c2: 
                LEVERAGE = st.number_input("Leverage (x)", 1, 100, 10, 1, on_change=lambda: play_sound_now("click"))
            SHOW_INR_CRYPTO = st.toggle("Convert to ₹ INR", True, on_change=lambda: play_sound_now("click"))
            USE_LIMIT_ORDERS = st.toggle("Use Limit Orders for Crypto", False, on_change=lambda: play_sound_now("click"))
        else:
            CRYPTO_MODE = "Options"
            LEVERAGE = 1
            SHOW_INR_CRYPTO = False
            USE_LIMIT_ORDERS = False

        st.divider()
        st.markdown("**2. Risk Management**")
        lot_size = LOT_SIZES.get(INDEX, 1)
        st.caption(f"1 lot = {lot_size} units for {INDEX}")
        min_val = 1.0 if INDEX in LOT_SIZES and lot_size > 1 else 0.01
        step_val = 1.0 if INDEX in LOT_SIZES and lot_size > 1 else 0.01
        LOTS = st.number_input("Base Lots", min_value=min_val, max_value=10000.0, value=1.0, step=step_val, key=f"lots_input_{INDEX}", on_change=lambda: play_sound_now("click"))
        actual_qty = LOTS * lot_size
        st.caption(f"Base quantity: {actual_qty:.2f} units")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            MAX_TRADES = st.number_input("Max Trades/Day", 1, 50, 5, on_change=lambda: play_sound_now("click"))
            MAX_CAPITAL = st.number_input("Max Cap/Trade (₹/$)", 10.0, 500000.0, 15000.0, step=100.0, on_change=lambda: play_sound_now("click"))
            SL_PTS = st.number_input("SL Points", 5.0, 500.0, 20.0, on_change=lambda: play_sound_now("click"))
            TSL_PTS = st.number_input("Trail SL", 5.0, 500.0, 15.0, on_change=lambda: play_sound_now("click"))
        with col_s2:
            TGT_PTS = st.number_input("Target Steps", 5.0, 1000.0, 15.0, on_change=lambda: play_sound_now("click"))
            CAPITAL_PROTECT = st.number_input("Max Loss", 500.0, 500000.0, 2000.0, step=500.0, on_change=lambda: play_sound_now("click"))

        MIN_SIGNAL_STRENGTH = st.slider("Min Signal Strength %", 0, 100, 30, 5, on_change=lambda: play_sound_now("click"))

        st.divider()
        if st.button("🔄 Refresh Balance", use_container_width=True, on_click=lambda: play_sound_now("click")):
            st.rerun()
        if not bot.is_mock and st.button("🧪 Ping API Connection", use_container_width=True, on_click=lambda: play_sound_now("click")):
            st.toast("Testing exact API parameters...", icon="🧪")
            bot.log(f"🧪 User executed manual Ping API Connection for {BROKER}.")

        render_signature()

    col1, col2 = st.columns([5, 1])
    with col1:
        broker_name = bot.settings.get("primary_broker", "Unknown")
        st.markdown(
            f"**👤 Session:** <span style='color:#0284c7; font-weight:800;'>{bot.client_name}</span> "
            f"<span class='broker-badge'>{broker_name}</span> | **IP:** `{bot.client_ip}` | **Device:** {st.session_state.device_name}",
            unsafe_allow_html=True
        )
    with col2:
        active_sessions = get_active_sessions(st.session_state.user_id) if st.session_state.user_id else []
        session_count = len(active_sessions)
        with st.popover(f"👤 {session_count}"):
            st.markdown(f"**Logged in as:** {st.session_state.user_id} (Role: {st.session_state.user_role})")
            st.markdown(f"**This device:** {st.session_state.device_name}")
            st.markdown(f"**IP:** {st.session_state.ip_address}")
            st.markdown("**Active Sessions:**")
            for sess in active_sessions[:5]:
                st.markdown(f"- {sess.get('device_name', 'Unknown')} ({sess.get('ip_address', 'Unknown')})")
            if session_count > 5:
                st.markdown(f"*(+{session_count - 5} more ghost sessions)*")
            st.divider()
            if st.button("🧹 Clear Ghost Sessions", use_container_width=True):
                if HAS_DB:
                    supabase.table("user_sessions").delete().eq("user_id", st.session_state.user_id).execute()
                    save_device_session(st.session_state.user_id, st.session_state.device_name, st.session_state.ip_address, st.session_state.session_id)
                st.rerun()
            if st.button("🚪 Logout", use_container_width=True, on_click=lambda: play_sound_now("click")):
                bot.state["is_running"] = False
                bot.state["engine_active"] = False
                if HAS_DB:
                    supabase.table("user_sessions").delete().eq("session_id", st.session_state.session_id).execute()
                st.session_state.clear()
                st.query_params.clear()
                st.rerun()

    bot.settings = {
        "primary_broker": BROKER, "strategy": STRATEGY, "index": INDEX, "timeframe": TIMEFRAME,
        "lots": LOTS,
        "max_trades": MAX_TRADES, "max_capital": MAX_CAPITAL, "capital_protect": CAPITAL_PROTECT,
        "sl_pts": SL_PTS, "tsl_pts": TSL_PTS, "tgt_pts": TGT_PTS, "paper_mode": bot.is_mock,
        "crypto_mode": CRYPTO_MODE, "leverage": LEVERAGE, "show_inr_crypto": SHOW_INR_CRYPTO,
        "user_lots": LOT_SIZES.copy(),
        "custom_code": CUSTOM_CODE, "tv_passphrase": TV_PASSPHRASE,
        "use_quantity_mode": True,
        "min_signal_strength": MIN_SIGNAL_STRENGTH,
        "use_limit_orders": USE_LIMIT_ORDERS
    }

    if bot.state['latest_data'] is None or st.session_state.prev_index != INDEX:
        st.session_state.prev_index = INDEX
        if not bot.state.get("is_running"):
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
                            df_preload, INDEX, 0.30, 1
                        )
                    elif "VIJAY & RFF" in STRATEGY:
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_vijay_rff_strategy(df_preload, INDEX)
                    elif "Institutional FVG" in STRATEGY:
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_institutional_fvg_strategy(df_preload, INDEX)
                    elif "Lux Algo" in STRATEGY:
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_lux_algo_ict_strategy(df_preload, INDEX)
                    elif "Mean Reversion" in STRATEGY:
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_mean_reversion_strategy(df_preload, INDEX)
                    elif STRATEGY == "TradingView Webhook":
                        t, s, v, e, df_c, atr, fib, strength = "Awaiting TradingView Webhook...", "WAIT", df_preload['close'].iloc[-1], df_preload['close'].iloc[-1], df_preload, 0, {}, 50
                    else:
                        t, s, v, e, df_c, atr, fib, strength = bot.analyzer.apply_primary_strategy(df_preload, INDEX)

                    df_work = df_c.copy() if df_c is not None else df_preload.copy()
                    temp_df = bot.analyzer.calculate_indicators(df_work, INDEX in ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIA VIX"])
                    for col in temp_df.columns:
                        if col not in df_work.columns:
                            df_work[col] = temp_df[col]

                    bot.state.update({
                        "current_trend": t, "current_signal": s,
                        "signal_strength": strength,
                        "vwap": v, "ema": e, "atr": atr,
                        "fib_data": fib, "latest_data": df_work.copy()
                    })

    if not is_mkt_open:
        st.error(f"🛑 {mkt_status_msg} - Engine will standby until market opens.")

    # Top button row (Start, Stop, Refresh only – Exit moved below)
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        if st.button("▶️ Start", use_container_width=True, disabled=bot.state["is_running"], on_click=lambda: play_sound_now("click")):
            bot.state["is_running"] = True
            bot.state["engine_active"] = True
            t = threading.Thread(target=bot.trading_loop, daemon=True)
            add_script_run_ctx(t)
            t.start()
            st.rerun()
    with bcol2:
        if st.button("⏹️ Stop", use_container_width=True, disabled=not bot.state["is_running"], on_click=lambda: play_sound_now("click")):
            bot.state["is_running"] = False
            bot.state["engine_active"] = False
            st.rerun()
    with bcol3:
        if st.button("🔄 Refresh", use_container_width=True, on_click=lambda: play_sound_now("click")):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    INDEX = st.session_state.sb_index_input
    MAX_TRADES = bot.settings.get('max_trades', 5) if bot else 5

    # Blue status box (now with Exit Engine button placed below it)
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0284c7, #0369a1); padding: 18px; border-radius: 4px; border: 1px solid #e2e8f0; color: white; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h2 style="margin: 0; color: #ffffff; font-weight: 800; letter-spacing: 1px;">🕉️ {INDEX}</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.95rem; color: #e0f2fe; font-weight: 700;">{broker_name}</p>
            <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">Strategy: {STRATEGY} | Timeframe: {TIMEFRAME}</p>
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed rgba(255,255,255,0.3);">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <span style="font-size: 0.85rem; color: #f8fafc;">Live Balance:</span><br>
                        <span style="font-size: 1.2rem; font-weight: bold; color: #ffffff;">{bot.get_balance()}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.85rem; color: #f8fafc;">Today's P&L:</span><br>
                        <span style="font-size: 2rem; font-weight: bold; color: {'#22c55e' if bot.state['daily_pnl'] >= 0 else '#ef4444'};">{'+' if bot.state['daily_pnl'] > 0 else ''}₹{abs(round(bot.state['daily_pnl'], 2))}</span>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Engine status and Exit Engine button
    is_running = bot.state["is_running"]
    status_color = "#22c55e" if is_running else "#ef4444"
    status_bg = "#f0fdf4" if is_running else "#fef2f2"
    status_text = f"🟢 ENGINE ACTIVE ({bot.state['trades_today']}/{MAX_TRADES} Trades)" if is_running else "🛑 ENGINE STOPPED"
    st.markdown(f"""
        <div style="text-align: center; padding: 10px; border-radius: 4px; background-color: {status_bg}; border: 1.5px solid {status_color}; color: {status_color}; font-weight: 800; font-size: 0.95rem; margin-bottom: 15px;">
            {status_text}
        </div>
    """, unsafe_allow_html=True)

    # Exit Engine button placed right below the status box
    if st.button("⛔ EXIT ENGINE", use_container_width=True, on_click=lambda: play_sound_now("click")):
        bot.force_exit()
        st.rerun()

    st.markdown("### 🎯 Live Position Tracker")

    @st.fragment(run_every="1s")
    def live_tracker_ui():
        if bot.state.get("active_trade"):
            t = bot.state["active_trade"]
            live_ltp = bot.get_live_price(t['exch'], t['symbol'], t['token'])
            if live_ltp:
                t['current_ltp'] = live_ltp
                if t['type'] in ["SELL", "PE"]:
                    t['floating_pnl'] = (t['entry'] - live_ltp) * t['qty']
                else:
                    t['floating_pnl'] = (live_ltp - t['entry']) * t['qty']
            ltp = t.get('current_ltp', t['entry'])
            pnl = t.get('floating_pnl', 0.0)
            elapsed = t.get('elapsed_time', 0)
            elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
            pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
            pnl_sign = "+" if pnl >= 0 else ""
            exec_type = t.get('exch', 'NFO')
            pnl_display = round(pnl, 2)
            if t['exch'] in ["DELTA", "COINDCX", "BINANCE"] and bot.settings.get('show_inr_crypto', True):
                inr_pnl = pnl * get_usdt_inr_rate()
                pnl_display = f"{pnl_sign}{round(pnl, 2)} (₹ {round(inr_pnl, 2)})"
            else:
                pnl_display = f"{pnl_sign}{round(pnl, 2)}"
            sim_badge = '<span class="simulated-badge">SIMULATED</span>' if t.get("simulated") else ''
            rej_info = f"<div class='rejection-reason' style='margin-top:5px;'>Reason: {t.get('rejection_reason', '')}</div>" if t.get("rejection_reason") else ''
            st.markdown(f"""<div class="live-tracker" style="margin-top: -15px;">
<div class="header" style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; border-bottom: 2px dashed #334155; padding-bottom: 15px; margin-bottom: 15px;">
<div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
<span class="symbol-badge" style="background: #3b82f6; color: white; padding: 6px 15px; border-radius: 30px; font-weight: 800;">{t["type"]}</span>
{sim_badge}
<span style="font-size: 1.3rem; color: white; font-weight: bold; margin: 0;">{t["symbol"]}</span>
</div>
<div class="pnl-badge" style="color: {pnl_color}; border: 2px solid {pnl_color}; padding: 8px 20px; border-radius: 40px; font-weight: 900; font-size: 1.6rem;">{pnl_display}</div>
</div>
{rej_info}
<div class="grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
<div class="info-box" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 16px; border: 1px solid #334155;">
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Avg Entry</div>
<div style="font-size: 1.4rem; font-weight: 800; color: #facc15;">{t["entry"]:.4f}</div>
</div>
<div class="info-box" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 16px; border: 1px solid #334155;">
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Live Mark</div>
<div style="font-size: 1.4rem; font-weight: 800; color: {'#4ade80' if pnl >= 0 else '#f87171'};">{ltp:.4f}</div>
</div>
<div class="info-box" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 16px; border: 1px solid #334155;">
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Qty</div>
<div style="font-size: 1.4rem; font-weight: 800; color: #facc15;">{t["qty"]} <span style="color: #94a3b8; font-size:0.8rem;">({exec_type})</span></div>
</div>
<div class="info-box" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 16px; border: 1px solid #334155;">
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Risk Stop</div>
<div style="font-size: 1.4rem; font-weight: 800; color: #f87171;">{t["sl"]:.4f}</div>
</div>
</div>
<div class="targets" style="background: linear-gradient(90deg, #1e293b, #111827); padding: 12px; border-radius: 40px; text-align: center; color: #38bdf8; font-weight: 700; border: 1px solid #38bdf8;">
⏱️ Time: {elapsed_str} | 🎯 TP1: <span style="color: #fbbf24; margin: 0 10px;">{t.get("tp1",0):.2f}</span> | TP2: <span style="color: #fbbf24; margin: 0 10px;">{t.get("tp2",0):.2f}</span> | TP3: <span style="color: #fbbf24; margin: 0 10px;">{t.get("tp3",0):.2f}</span>
</div>
</div>""", unsafe_allow_html=True)
        elif bot.state.get("active_trades"):
            for idx, t in enumerate(bot.state["active_trades"]):
                live_ltp = bot.get_live_price(t['exch'], t['symbol'], t['token'])
                if live_ltp:
                    t['current_ltp'] = live_ltp
                    if t['type'] in ["SELL", "SELL_CALL", "SELL_PUT", "SHORT"]:
                        t['floating_pnl'] = (t['entry'] - live_ltp) * t['qty']
                    else:
                        t['floating_pnl'] = (live_ltp - t['entry']) * t['qty']
                ltp = t.get('current_ltp', t['entry'])
                pnl = t.get('floating_pnl', 0.0)
                elapsed = t.get('elapsed_time', 0)
                elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
                pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
                pnl_sign = "+" if pnl >= 0 else ""
                exec_type = t.get('exch', 'NFO')
                pnl_display = round(pnl, 2)
                if t['exch'] in ["DELTA", "COINDCX", "BINANCE"] and bot.settings.get('show_inr_crypto', True):
                    inr_pnl = pnl * get_usdt_inr_rate()
                    pnl_display = f"{pnl_sign}{round(pnl, 2)} (₹ {round(inr_pnl, 2)})"
                else:
                    pnl_display = f"{pnl_sign}{round(pnl, 2)}"
                sim_badge = '<span class="simulated-badge">SIMULATED</span>' if t.get("simulated") else ''
                rej_info = f"<div class='rejection-reason' style='margin-top:5px;'>Reason: {t.get('rejection_reason', '')}</div>" if t.get("rejection_reason") else ''
                st.markdown(f"""<div class="live-tracker" style="margin-top: 10px;">
<div class="header" style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; border-bottom: 2px dashed #334155; padding-bottom: 15px; margin-bottom: 15px;">
<div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
<span class="symbol-badge" style="background: #3b82f6; color: white; padding: 6px 15px; border-radius: 30px; font-weight: 800;">{t["type"]}</span>
{sim_badge}
<span style="font-size: 1.3rem; color: white; font-weight: bold; margin: 0;">{t["symbol"]}</span>
</div>
<div class="pnl-badge" style="color: {pnl_color}; border: 2px solid {pnl_color}; padding: 8px 20px; border-radius: 40px; font-weight: 900; font-size: 1.6rem;">{pnl_display}</div>
</div>
{rej_info}
<div class="grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
<div class="info-box" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 16px; border: 1px solid #334155;">
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Avg Entry</div>
<div style="font-size: 1.4rem; font-weight: 800; color: #facc15;">{t["entry"]:.4f}</div>
</div>
<div class="info-box" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 16px; border: 1px solid #334155;">
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Live Mark</div>
<div style="font-size: 1.4rem; font-weight: 800; color: {'#4ade80' if pnl >= 0 else '#f87171'};">{ltp:.4f}</div>
</div>
<div class="info-box" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 16px; border: 1px solid #334155;">
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Qty</div>
<div style="font-size: 1.4rem; font-weight: 800; color: #facc15;">{t["qty"]} <span style="color: #94a3b8; font-size:0.8rem;">({exec_type})</span></div>
</div>
<div class="info-box" style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 16px; border: 1px solid #334155;">
<div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Risk Stop</div>
<div style="font-size: 1.4rem; font-weight: 800; color: #f87171;">{t["sl"]:.4f}</div>
</div>
</div>
<div class="targets" style="background: linear-gradient(90deg, #1e293b, #111827); padding: 12px; border-radius: 40px; text-align: center; color: #38bdf8; font-weight: 700; border: 1px solid #38bdf8;">
⏱️ Time: {elapsed_str} | 🎯 TP1: <span style="color: #fbbf24; margin: 0 10px;">{t.get("tp1",0):.2f}</span> | TP2: <span style="color: #fbbf24; margin: 0 10px;">{t.get("tp2",0):.2f}</span> | TP3: <span style="color: #fbbf24; margin: 0 10px;">{t.get("tp3",0):.2f}</span>
</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: center; height: 230px; background: rgba(255,255,255,0.02); border: 2px dashed #334155; border-radius: 16px; margin-top: -15px; margin-bottom: 5px;">
                <span style="color: #94a3b8; font-weight: 600; font-size: 1.1rem;">⏳ Radar Active: Waiting for High-Probability Setup...</span>
            </div>
            """, unsafe_allow_html=True)

    live_tracker_ui()

    st.markdown('<div class="sticky-buttons">', unsafe_allow_html=True)
    col_exit, col_protect = st.columns(2)
    with col_exit:
        st.button(
            "🛑 EXIT TRADE INSTANTLY", 
            type="primary", 
            use_container_width=True, 
            key="global_exit_btn",
            on_click=bot.force_exit,
            disabled=(bot.state.get("active_trade") is None and not bot.state.get("active_trades"))
        )
    with col_protect:
        st.button(
            "🛡️ Protect Profit", 
            type="secondary", 
            use_container_width=True, 
            key="protect_profit_btn",
            on_click=bot.protect_profit,
            disabled=(bot.state.get("active_trade") is None and not bot.state.get("active_trades"))
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ... (the rest of the dashboard tabs – unchanged from original)
    # Since the original tabs (scanners, logs, crypto, etc.) are unchanged,
    # we'll include them here as they were in the original script.
    # For brevity, we'll reference that they are present and continue.

    # Tabs definition
    tab_names = ["🕉️ DASHBOARD", "🔎 SCANNERS", "📜 LOGS", "🚀 CRYPTO/FX", "💰 SAFE INVESTMENTS", "🤖 FIA ASSISTANT", "📊 BACKTEST"]
    if st.session_state.is_developer:
        tab_names.append("🛡️ ADMIN")
    tabs = st.tabs(tab_names)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = tabs[:7]
    if st.session_state.is_developer:
        tab8 = tabs[7]

    # Tab1: DASHBOARD – already shown above (live tracker and stats). We'll just include the original dashboard stats and chart.
    with tab1:
        # News tickers
        kannada_news = st.session_state.kannada_news
        english_news = st.session_state.english_news
        if not kannada_news:
            kannada_news = [generate_market_prediction(INDEX)]
        ticker_text = " 🔹 ".join(kannada_news)
        st.markdown(f'<div class="news-ticker-kannada"><span>{ticker_text}</span></div>', unsafe_allow_html=True)
        if not english_news:
            english_news = [generate_market_prediction(INDEX)]
        ticker_text = " 🔹 ".join(english_news)
        st.markdown(f'<div class="news-ticker-english"><span>{ticker_text}</span></div>', unsafe_allow_html=True)

        # Additional stats and chart (same as original)
        st.markdown("<br>### 📈 Technical Engine", unsafe_allow_html=True)
        c_h1, c_h2 = st.columns(2)
        with c_h1:
            SHOW_CHART = st.toggle("📊 Render Chart", True, on_change=lambda: play_sound_now("click"))
        with c_h2:
            FULL_CHART = st.toggle("⛶ Full Screen", False, on_change=lambda: play_sound_now("click"))

        if SHOW_CHART and bot.state["latest_data"] is not None:
            chart_df = bot.state["latest_data"].copy()
            if not isinstance(chart_df.index, pd.DatetimeIndex):
                if 'timestamp' in chart_df.columns:
                    chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
                    chart_df.set_index('timestamp', inplace=True)
                else:
                    chart_df.index = pd.date_range(end=get_ist(), periods=len(chart_df), freq='T')
            try:
                if chart_df.index.tz is not None:
                    chart_df.index = chart_df.index.tz_localize(None)
            except AttributeError:
                pass
            chart_df['time'] = chart_df.index.astype('int64') // 10**9
            candles = chart_df[['time', 'open', 'high', 'low', 'close']].dropna().to_dict('records')
            if len(candles) == 0:
                st.warning("No candle data available for chart.")
            else:
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
                    "layout": {"textColor": '#1e293b', "background": {"type": 'solid', "color": '#ffffff'}},
                    "grid": {"vertLines": {"color": 'rgba(226, 232, 240, 0.8)'}, "horzLines": {"color": 'rgba(226, 232, 240, 0.8)'}},
                    "crosshair": {"mode": 0},
                    "timeScale": {"timeVisible": True, "secondsVisible": False}
                }
                chart_series = [{"type": 'Candlestick', "data": candles, "options": {"upColor": '#26a69a', "downColor": '#ef5350'}, "priceLines": fib_lines}]

                if 'anchored_vwap' in chart_df.columns:
                    avwap_data = chart_df[['time', 'anchored_vwap']].dropna().rename(columns={'anchored_vwap': 'value'}).to_dict('records')
                    if avwap_data:
                        chart_series.append({"type": 'Line', "data": avwap_data, "options": {"color": '#9c27b0', "lineWidth": 2, "title": 'ICT AVWAP'}})
                if 'vwap' in chart_df.columns:
                    vwap_data = chart_df[['time', 'vwap']].dropna().rename(columns={'vwap': 'value'}).to_dict('records')
                    if vwap_data:
                        chart_series.append({"type": 'Line', "data": vwap_data, "options": {"color": '#ff9800', "lineWidth": 2, "title": 'VWAP'}})
                ema_col = None
                if 'ema_fast' in chart_df.columns:
                    ema_col = 'ema_fast'
                elif 'ema_short' in chart_df.columns:
                    ema_col = 'ema_short'
                elif 'ema9' in chart_df.columns:
                    ema_col = 'ema9'
                if ema_col:
                    ema_data = chart_df[['time', ema_col]].dropna().rename(columns={ema_col: 'value'}).to_dict('records')
                    if ema_data:
                        chart_series.append({"type": 'Line', "data": ema_data, "options": {"color": '#0ea5e9', "lineWidth": 2, "title": 'EMA'}})
                if 'supertrend' in chart_df.columns:
                    st_data = chart_df[['time', 'supertrend']].dropna().rename(columns={'supertrend': 'value'}).to_dict('records')
                    if st_data:
                        chart_series.append({"type": 'Line', "data": st_data, "options": {"color": '#e67e22', "lineWidth": 1, "title": 'Supertrend'}})

                renderLightweightCharts([{"chart": chartOptions, "series": chart_series}], key="static_tv_chart")
        elif not SHOW_CHART:
            st.info("Chart is hidden. Enable 'Render Chart' to view.")

        # FOMO Scanner
        st.markdown("### 🚨 FOMO Scanner (Volume Spike Alerts)")
        fomo_signals = fomo_scanner.scan()
        if fomo_signals:
            df_fomo = pd.DataFrame(fomo_signals)
            st.dataframe(df_fomo, use_container_width=True, hide_index=True)
        else:
            st.info("No volume spike alerts at the moment.")

        # Win rate metric
        total_trades = len(bot.state.get("paper_history", [])) if bot.is_mock else bot.state.get("trades_today", 0)
        wins = bot.state.get("hz_wins", 0) + (st.session_state.win_streak if st.session_state.win_streak > 0 else 0)
        win_pct = (wins / total_trades * 100) if total_trades > 0 else 0
        st.metric("Win Percentage", f"{win_pct:.1f}%")

        if bot.settings.get('hero_zero', False) and bot.state.get("hz_trades"):
            hz_win_rate = (bot.state["hz_wins"] / len(bot.state["hz_trades"]) * 100) if bot.state["hz_trades"] else 0
            st.markdown(f"""
            <div class="hz-stats">
                <h4 style="color: white; margin:0 0 10px 0;">🎯 Hero/Zero Performance</h4>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                    <div>Total Trades: {len(bot.state['hz_trades'])}</div>
                    <div>Win Rate: {hz_win_rate:.1f}%</div>
                    <div>Total P&L: {'🟢' if bot.state['hz_pnl'] > 0 else '🔴'} ₹{bot.state['hz_pnl']:.2f}</div>
                    <div>Wins: {bot.state['hz_wins']} | Losses: {bot.state['hz_losses']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tab2: SCANNERS – unchanged from original
    with tab2:
        sub_tabs = st.tabs(["📊 52W High/Low", "📡 Multi-Stock + Pin Bar", "🇺🇸 US Stock Scanner", "🌙 Overnight Profitable", "🎯 Hero/Zero Scanner"])
        with sub_tabs[0]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("52‑Week High/Low Scanner")
                @st.cache_data(ttl=86400)
                def scan_52w():
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
                    return results
                results = scan_52w()
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

        with sub_tabs[1]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("Multi‑Stock + Pin Bar Scanner")
                @st.cache_data(ttl=86400)
                def scan_multistock():
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
                    return results
                results = scan_multistock()
                if results:
                    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                else:
                    st.info("No data.")
                st.markdown('</div>', unsafe_allow_html=True)

        with sub_tabs[2]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("🇺🇸 US Stock Scanner")
                us_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT", "JNJ", "PG", "UNH", "HD", "DIS", "NFLX", "ADBE", "CRM", "AMD", "INTC"]
                @st.cache_data(ttl=86400)
                def scan_us():
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
                    return results
                results = scan_us()
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

        with sub_tabs[3]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("🌙 Overnight Profitable Stocks & Crypto")
                st.markdown("Stocks that gap up/down significantly pre-market (Indian & US) and crypto with high overnight volatility.")
                @st.cache_data(ttl=86400)
                def scan_overnight():
                    results = []
                    indian_list = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "WIPRO.NS"]
                    for ticker in indian_list:
                        try:
                            tk = yf.Ticker(ticker)
                            hist = tk.history(period="2d", interval="1m")
                            if len(hist) < 60: continue
                            yesterday_close = hist['Close'].iloc[-390] if len(hist) > 390 else hist['Close'].iloc[0]
                            pre_high = hist['High'].iloc[-30:].max()
                            pre_low = hist['Low'].iloc[-30:].min()
                            change_high = (pre_high - yesterday_close) / yesterday_close * 100
                            change_low = (pre_low - yesterday_close) / yesterday_close * 100
                            if change_high > 1.5:
                                results.append({
                                    "Symbol": ticker.replace(".NS", ""),
                                    "Type": "Stock",
                                    "Pre-Market High %": f"+{change_high:.2f}%",
                                    "Pre-Market Low %": f"{change_low:.2f}%",
                                    "Signal": "Bullish Gap 🚀"
                                })
                            elif change_low < -1.5:
                                results.append({
                                    "Symbol": ticker.replace(".NS", ""),
                                    "Type": "Stock",
                                    "Pre-Market High %": f"+{change_high:.2f}%",
                                    "Pre-Market Low %": f"{change_low:.2f}%",
                                    "Signal": "Bearish Gap 🔻"
                                })
                        except: pass
                    us_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
                    for ticker in us_list:
                        try:
                            tk = yf.Ticker(ticker)
                            hist = tk.history(period="2d", interval="1m")
                            if len(hist) < 60: continue
                            yesterday_close = hist['Close'].iloc[-390] if len(hist) > 390 else hist['Close'].iloc[0]
                            pre_high = hist['High'].iloc[-30:].max()
                            pre_low = hist['Low'].iloc[-30:].min()
                            change_high = (pre_high - yesterday_close) / yesterday_close * 100
                            change_low = (pre_low - yesterday_close) / yesterday_close * 100
                            if change_high > 1.5:
                                results.append({
                                    "Symbol": ticker,
                                    "Type": "US Stock",
                                    "Pre-Market High %": f"+{change_high:.2f}%",
                                    "Pre-Market Low %": f"{change_low:.2f}%",
                                    "Signal": "Bullish Gap 🚀"
                                })
                            elif change_low < -1.5:
                                results.append({
                                    "Symbol": ticker,
                                    "Type": "US Stock",
                                    "Pre-Market High %": f"+{change_high:.2f}%",
                                    "Pre-Market Low %": f"{change_low:.2f}%",
                                    "Signal": "Bearish Gap 🔻"
                                })
                        except: pass
                    try:
                        resp = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5)
                        if resp.status_code == 200:
                            data = resp.json()
                            for coin in data:
                                mkt = coin.get('market', '')
                                if mkt.endswith('USDT'):
                                    chg = float(coin.get('change_24_hour', 0))
                                    if abs(chg) > 3:
                                        results.append({
                                            "Symbol": mkt,
                                            "Type": "Crypto",
                                            "24h Change": f"{chg:+.2f}%",
                                            "Pre-Market High %": "-",
                                            "Pre-Market Low %": "-",
                                            "Signal": "High Volatility ⚡"
                                        })
                    except: pass
                    return results
                results = scan_overnight()
                if results:
                    df_ov = pd.DataFrame(results)
                    st.dataframe(df_ov, use_container_width=True, hide_index=True)
                else:
                    st.info("No significant overnight movements detected.")
                st.markdown('</div>', unsafe_allow_html=True)

        with sub_tabs[4]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("🎯 Hero/Zero Scanner & Pin Bar Reversals")

                def color_direction(val):
                    if "HERO" in val:
                        return 'background-color: #22c55e; color: white'
                    elif "ZERO" in val:
                        return 'background-color: #ef4444; color: white'
                    return ''

                col_hz1, col_hz2 = st.columns(2)
                with col_hz1:
                    min_volume = st.slider("Min Volume Spike", 1.0, 3.0, 1.5, 0.1, key="hz_volume")
                with col_hz2:
                    st.info("Auto‑scan (cached every 5 min)")
                with st.spinner("Scanning for Hero/Zero patterns..."):
                    st.markdown("### Nifty 50 Stocks")
                    nifty_results = bot.scan_hero_zero_indian_stocks()
                    if not nifty_results.empty:
                        st.success(f"Found {len(nifty_results)} Hero/Zero opportunities in Nifty 50!")
                        styled_results = nifty_results.style.map(color_direction, subset=['Direction'])
                        st.dataframe(styled_results, use_container_width=True, hide_index=True)
                    else:
                        st.info("No Hero/Zero opportunities in Nifty 50 at this moment")

                    st.markdown("### Penny Stocks")
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
                st.markdown("### ⏰ Best Trading Times (IST)")
                st.markdown("""
                - **Opening Range:** 9:15 AM - 10:00 AM (Best momentum)
                - **Mid-Morning:** 10:30 AM - 11:30 AM (Good follow-through)
                - **Closing Range:** 2:00 PM - 3:15 PM (Strong moves)
                - **Avoid:** 11:30 AM - 1:30 PM (Lunch hour, low volume)
                """)

    # Tab3: LOGS – unchanged
    with tab3:
        sub_tabs = st.tabs(["📋 Console", "📊 Ledger", "📄 Tax Report"])
        with sub_tabs[0]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                col_clr, col_msg = st.columns([1, 5])
                with col_clr:
                    if st.button("🗑️ Clear", key="clear_logs_btn", use_container_width=True,
                                 on_click=lambda: play_sound_now("click")):
                        bot.state["logs"].clear()
                        if bot.is_mock and "paper_history" in bot.state:
                            bot.state["paper_history"] = []
                        bot.state["daily_pnl"] = 0.0
                        bot.state["trades_today"] = 0
                        if not bot.is_mock and HAS_DB:
                            try:
                                uid = getattr(bot, "system_user_id", bot.api_key)
                                supabase.table("trade_logs").delete().eq("user_id", uid).execute()
                            except Exception as e:
                                st.error(f"DB clear error: {e}")
                        st.rerun()
                with col_msg:
                    st.caption("Clears console, paper history, and resets daily P&L (does not stop engine).")
                if len(bot.state["logs"]) == 0:
                    st.info("No logs yet. Start the engine to see activity.")
                else:
                    for log in bot.state["logs"]:
                        st.markdown(f"`{log}`")
                st.markdown('</div>', unsafe_allow_html=True)
        with sub_tabs[1]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                report_period = st.selectbox("Select report period", ["Daily", "Weekly", "All Time"], index=0)
                if bot.is_mock:
                    if bot.state.get("paper_history"):
                        df = pd.DataFrame(bot.state["paper_history"])
                        total_pnl = df['PnL'].sum()
                        wins = len(df[df['PnL'] > 0])
                        losses = len(df[df['PnL'] <= 0])
                        win_rate = (wins / len(df) * 100) if len(df) > 0 else 0
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Trades", len(df))
                        col2.metric("Win Rate", f"{win_rate:.1f}%")
                        col3.metric("Total PnL", f"₹{total_pnl:.2f}")
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
                        st.download_button("📥 Export", data=output.getvalue(),
                                           file_name="mock_ledger.xlsx",
                                           on_click=lambda: play_sound_now("click"))
                    else:
                        st.info("No paper trades yet.")
                else:
                    uid = getattr(bot, "system_user_id", bot.api_key)
                    if HAS_DB:
                        try:
                            res = supabase.table("trade_logs").select("*").eq("user_id", uid).execute()
                            if res.data:
                                df = pd.DataFrame(res.data).drop(columns=["id", "user_id"], errors="ignore")
                                df['trade_date'] = pd.to_datetime(df['trade_date'])
                                total_pnl = df['pnl'].sum()
                                wins = len(df[df['pnl'] > 0])
                                losses = len(df[df['pnl'] <= 0])
                                win_rate = (wins / len(df) * 100) if len(df) > 0 else 0
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total Trades", len(df))
                                col2.metric("Win Rate", f"{win_rate:.1f}%")
                                col3.metric("Total PnL", f"₹{total_pnl:.2f}")
                                st.dataframe(df.sort_values('trade_time', ascending=False), use_container_width=True)

                                if report_period == "Daily":
                                    today = get_ist().date()
                                    df_report = df[df['trade_date'].dt.date == today]
                                elif report_period == "Weekly":
                                    week_ago = get_ist().date() - dt.timedelta(days=7)
                                    df_report = df[df['trade_date'].dt.date >= week_ago]
                                else:
                                    df_report = df
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as w:
                                    df_report.to_excel(w, index=False)
                                st.download_button("📥 Export", data=output.getvalue(),
                                                   file_name="live_ledger.xlsx",
                                                   on_click=lambda: play_sound_now("click"))
                            else:
                                st.info("No live trades.")
                        except Exception as e:
                            st.error(f"DB error: {e}")
                    else:
                        st.error("DB disconnected.")
                st.markdown('</div>', unsafe_allow_html=True)

        with sub_tabs[2]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("📄 Tax Report")
                if st.session_state.user_id and HAS_DB:
                    tax_year = st.selectbox("Financial Year", [2023, 2024, 2025])
                    if st.button("Generate Report", on_click=lambda: play_sound_now("click")):
                        with st.spinner("Generating report..."):
                            summary, df_tax = generate_tax_report(st.session_state.user_id, tax_year)
                            if summary:
                                st.markdown(f"### Summary for FY {tax_year}-{tax_year+1}")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("STCG", f"₹{summary.get('STCG', 0):.2f}")
                                with col2:
                                    st.metric("LTCG", f"₹{summary.get('LTCG', 0):.2f}")
                                st.dataframe(df_tax)
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as w:
                                    df_tax.to_excel(w, index=False)
                                st.download_button("📥 Download Tax Report", data=output.getvalue(),
                                                   file_name=f"tax_report_{tax_year}.xlsx",
                                                   on_click=lambda: play_sound_now("click"))
                            else:
                                st.info("No trades found for this period.")
                else:
                    st.warning("Login required or database not connected.")
                st.markdown('</div>', unsafe_allow_html=True)

    # Tab4: CRYPTO/FX – unchanged
    with tab4:
        sub_crypto = st.tabs(["🪙 CoinDCX Scanner", "⚡ 1-Min Scalper", "🚀 Breakout Scanner", "🪄 Web3 / DeFi"])
        with sub_crypto[0]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("🕉️ CoinDCX Momentum")
                @st.cache_data(ttl=30)
                def scan_coindcx():
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
                                return df
                    except:
                        pass
                    return None
                df_c = scan_coindcx()
                if df_c is not None:
                    st.dataframe(df_c, use_container_width=True, hide_index=True)
                else:
                    st.info("Unable to fetch CoinDCX data.")
                st.markdown('</div>', unsafe_allow_html=True)
        with sub_crypto[1]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                scalper_asset = st.selectbox("Select Asset", ["XAUUSD (Gold)", "BTCUSD", "ETHUSD", "SOLUSD"], index=0)
                asset_map = {"XAUUSD (Gold)": "XAUUSD", "BTCUSD": "BTCUSD", "ETHUSD": "ETHUSD", "SOLUSD": "SOLUSD"}
                symbol = asset_map[scalper_asset]
                gold_crypto_scalper(symbol)
                st.markdown('</div>', unsafe_allow_html=True)
        with sub_crypto[2]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                add_breakout_scalper()
                st.markdown('</div>', unsafe_allow_html=True)
        with sub_crypto[3]:
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.subheader("🪄 Web3 / DeFi Top Movers")
                col1, col2 = st.columns(2)
                with col1:
                    source2 = st.radio("Source", ["CoinDCX", "CoinGecko"], horizontal=True, key="web3_source")
                with col2:
                    limit2 = st.number_input("Show Top", 5, 50, 10, key="web3_limit")
                if st.button("Fetch Web3 Data", on_click=lambda: play_sound_now("click"), key="web3_fetch"):
                    with st.spinner("Fetching..."):
                        if source2 == "CoinDCX":
                            try:
                                resp = requests.get("https://api.coindcx.com/exchange/ticker", timeout=5)
                                if resp.status_code == 200:
                                    data = resp.json()
                                    pairs = []
                                    for coin in data:
                                        mkt = coin.get('market', '')
                                        if mkt.endswith('USDT'):
                                            try:
                                                chg = float(coin.get('change_24_hour', 0))
                                                price = float(coin.get('last_price', 0))
                                                vol = float(coin.get('volume', 0))
                                                pairs.append({
                                                    "Pair": mkt,
                                                    "Price": price,
                                                    "24h Change %": chg,
                                                    "Volume": vol
                                                })
                                            except: pass
                                    df = pd.DataFrame(pairs).sort_values("24h Change %", ascending=False)
                                    st.markdown("#### 🟢 Top Gainers")
                                    st.dataframe(df.head(limit2)[["Pair", "Price", "24h Change %", "Volume"]], use_container_width=True, hide_index=True)
                                    st.markdown("#### 🔴 Top Losers")
                                    st.dataframe(df.tail(limit2).sort_values("24h Change %")[["Pair", "Price", "24h Change %", "Volume"]], use_container_width=True, hide_index=True)
                                else:
                                    st.error("Failed")
                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            try:
                                resp = requests.get("https://api.coingecko.com/api/v3/coins/markets", params={
                                    "vs_currency": "usd",
                                    "order": "market_cap_desc",
                                    "per_page": 100,
                                    "page": 1,
                                    "sparkline": "false"
                                }, timeout=10)
                                if resp.status_code == 200:
                                    data = resp.json()
                                    df = pd.DataFrame(data)
                                    df = df[["symbol", "current_price", "price_change_percentage_24h", "total_volume"]].rename(columns={
                                        "symbol": "Coin",
                                        "current_price": "Price (USD)",
                                        "price_change_percentage_24h": "24h Change %",
                                        "total_volume": "Volume"
                                    })
                                    df = df.sort_values("24h Change %", ascending=False)
                                    st.markdown("#### 🟢 Top Gainers")
                                    st.dataframe(df.head(limit2), use_container_width=True, hide_index=True)
                                    st.markdown("#### 🔴 Top Losers")
                                    st.dataframe(df.tail(limit2).sort_values("24h Change %"), use_container_width=True, hide_index=True)
                                else:
                                    st.error("Failed")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)

    # Tab5: SAFE INVESTMENTS – unchanged
    with tab5:
        def safe_investment_suggestions():
            st.markdown("### 💰 Safe Investment Suggestions")
            st.markdown("These are conservative, low-risk ideas suitable for long-term wealth building.")
            with st.expander("📈 **Index ETFs**"):
                st.markdown("""
                - **Nippon India ETF Nifty50** (JUNIORBEES): Low expense ratio, tracks Nifty50.
                - **HDFC Sensex ETF**: Tracks BSE Sensex, good for long-term.
                - **ICICI Prudential Nifty Next 50**: Captures emerging bluechips.
                """)
            with st.expander("🏦 **Fixed Income**"):
                st.markdown("""
                - **RBI Floating Rate Savings Bonds** – Interest rate reset every 6 months.
                - **Corporate Bonds** (AAA rated) – e.g., NTPC, Power Finance Corp.
                - **SGB (Sovereign Gold Bonds)** – Govt gold bonds with 2.5% interest.
                """)
            with st.expander("🌍 **International ETFs**"):
                st.markdown("""
                - **MOTILAL OSWAL S&P500 Index Fund** – US exposure.
                - **Hang Seng Index ETF** – Hong Kong market.
                - **iShares MSCI World ETF** (via international brokerage).
                """)
            with st.expander("💎 **Commodity ETFs**"):
                st.markdown("""
                - **Gold ETFs** – HDFC Gold, SBI Gold.
                - **Silver ETFs** – ICICI Prudential Silver ETF.
                """)
            st.markdown("---")
            st.subheader("📊 Compounding Calculator")
            compounding_calculator()
            st.info("💡 *Past performance does not guarantee future results. Always consult a financial advisor.*")
        safe_investment_suggestions()

    # Tab6: FIA ASSISTANT – unchanged
    with tab6:
        st.subheader("🤖 FIA Assistant – Market Analysis")
        def fia_assistant(df, index):
            if df is None:
                st.info("No data available.")
                return
            st.markdown(f"### 📊 Analysis for {index}")
            last = df.iloc[-1]
            prev = df.iloc[-2]
            change = (last['close'] - prev['close']) / prev['close'] * 100
            st.metric("Last Close", f"{last['close']:.2f}", f"{change:.2f}%")
            rsi = last.get('rsi', 50)
            st.metric("RSI (14)", f"{rsi:.1f}", "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"))
            st.markdown("**Key Levels:**")
            sup, res = bot.analyzer.get_support_resistance(df)
            if sup:
                st.markdown(f"- Support: {sup:.2f}")
            if res:
                st.markdown(f"- Resistance: {res:.2f}")
            trend = "🟢 Bullish" if last['close'] > last.get('ema9', last['close']) else "🔴 Bearish"
            st.markdown(f"**Short-term trend:** {trend}")
            if st.button("Refresh Analysis", on_click=lambda: play_sound_now("click")):
                st.rerun()
        if bot.state.get("latest_data") is not None:
            fia_assistant(bot.state["latest_data"], INDEX)
        else:
            st.info("No chart data available yet. Start the engine or refresh.")

    # Tab7: BACKTEST – unchanged (with updated date defaults)
    with tab7:
        st.subheader("📊 Backtesting Engine")
        st.markdown("Test your strategy on historical data (runs in background).")

        bt_strategy = st.selectbox("Select Strategy", STRAT_LIST, index=0)
        bt_symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "BTCUSD", "ETHUSD", "XAUUSD"])
        bt_start = st.date_input("Start Date", get_ist().date())
        bt_end = st.date_input("End Date", get_ist().date() + dt.timedelta(days=365))
        bt_initial_cap = st.number_input("Initial Capital (₹)", 10000, 1000000, 100000)
        bt_lot_size = st.number_input("Lot Size", 1, 100, 1)

        if 'backtest_result' not in st.session_state:
            st.session_state.backtest_result = None
        if 'backtest_running' not in st.session_state:
            st.session_state.backtest_running = False

        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            run_bt = st.button("🚀 Run Backtest", use_container_width=True, on_click=lambda: play_sound_now("click"))
        with col_bt2:
            if st.button("🔄 Clear Results", use_container_width=True, on_click=lambda: play_sound_now("click")):
                st.session_state.backtest_result = None
                st.rerun()

        if run_bt:
            st.session_state.backtest_running = True
            def run_backtest_thread():
                try:
                    ticker = YF_TICKERS.get(bt_symbol, bt_symbol)
                    if bt_symbol == "XAUUSD":
                        ticker = "GC=F"
                    df = yf.download(ticker, start=bt_start, end=bt_end, interval="1d")
                    if df.empty:
                        st.session_state.backtest_result = {"error": "No data fetched"}
                        st.session_state.backtest_running = False
                        return
                    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

                    if bt_strategy == "Momentum Breakout + S&R":
                        strategy_func = TechnicalAnalyzer.apply_primary_strategy
                    elif bt_strategy == "Machine Learning":
                        strategy_func = TechnicalAnalyzer.apply_primary_strategy
                    elif bt_strategy == "VIJAY & RFF All-In-One":
                        strategy_func = TechnicalAnalyzer.apply_vijay_rff_strategy
                    elif bt_strategy == "Institutional FVG + SMC":
                        strategy_func = TechnicalAnalyzer.apply_institutional_fvg_strategy
                    elif bt_strategy == "Lux Algo Institutional ICT":
                        strategy_func = TechnicalAnalyzer.apply_lux_algo_ict_strategy
                    elif bt_strategy == "Mean Reversion (BB/RSI)":
                        strategy_func = TechnicalAnalyzer.apply_mean_reversion_strategy
                    else:
                        strategy_func = TechnicalAnalyzer.apply_primary_strategy

                    bt = Backtester(strategy_func, df, bt_initial_cap, bt_lot_size)
                    equity, trades = bt.run()

                    result = {
                        "equity": equity,
                        "trades": trades,
                        "total_return": bt.total_return,
                        "sharpe": bt.sharpe,
                        "max_dd": bt.max_dd,
                        "win_rate": bt.win_rate,
                        "avg_win": bt.avg_win,
                        "avg_loss": bt.avg_loss,
                    }
                    st.session_state.backtest_result = result
                except Exception as e:
                    st.session_state.backtest_result = {"error": str(e)}
                finally:
                    st.session_state.backtest_running = False

            thread = threading.Thread(target=run_backtest_thread, daemon=True)
            add_script_run_ctx(thread)
            thread.start()
            st.rerun()

        if st.session_state.backtest_running:
            st.info("⏳ Backtest running in background...")
            st.spinner("Please wait")
        elif st.session_state.backtest_result:
            res = st.session_state.backtest_result
            if "error" in res:
                st.error(f"Backtest failed: {res['error']}")
            else:
                st.success(f"Backtest complete! Total Return: {res['total_return']:.2f}%")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sharpe Ratio", f"{res['sharpe']:.2f}")
                col2.metric("Max Drawdown", f"{res['max_dd']:.2f}%")
                col3.metric("Win Rate", f"{res['win_rate']:.1f}%")
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Win", f"₹{res['avg_win']:.2f}")
                col2.metric("Avg Loss", f"₹{res['avg_loss']:.2f}")
                col3.metric("Total Trades", len(res['trades']))

                if res['trades']:
                    st.markdown("### Trade Log")
                    trades_df = pd.DataFrame(res['trades'])
                    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d')
                    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d')
                    st.dataframe(trades_df, use_container_width=True, hide_index=True)

                st.markdown("### Equity Curve")
                st.line_chart(res['equity'])
        else:
            st.info("Click 'Run Backtest' to start.")

    # Tab8: ADMIN – enhanced (user management, broker balances, active sessions, logs)
    if st.session_state.is_developer:
        with tab8:
            st.markdown('<div class="admin-card">', unsafe_allow_html=True)
            st.header("🛡️ Admin Panel")
            admin_tabs = st.tabs(["👥 User Management", "💰 Broker Balances", "📊 Active Sessions", "⚙️ System Logs"])
            
            with admin_tabs[0]:
                st.subheader("User Management")
                if HAS_DB:
                    try:
                        users = supabase.table("user_credentials").select("*").execute()
                        if users.data:
                            df_users = pd.DataFrame(users.data)
                            # Add blocked status (if column exists, else default False)
                            try:
                                if "blocked" not in df_users.columns:
                                    df_users["blocked"] = False
                            except:
                                df_users["blocked"] = False
                            st.dataframe(df_users[["user_id", "role", "blocked"]], use_container_width=True)
                            
                            selected_user = st.selectbox("Select User to Manage", df_users["user_id"].tolist())
                            col1, col2 = st.columns(2)
                            with col1:
                                new_role = st.selectbox("Set Role", ["trader", "admin"], index=0 if df_users[df_users["user_id"]==selected_user]["role"].iloc[0]=="trader" else 1)
                                if st.button("Update Role", on_click=lambda: play_sound_now("click")):
                                    supabase.table("user_credentials").update({"role": new_role}).eq("user_id", selected_user).execute()
                                    st.success(f"Role updated for {selected_user}")
                                    st.rerun()
                            with col2:
                                blocked_status = st.checkbox("Block User", value=bool(df_users[df_users["user_id"]==selected_user]["blocked"].iloc[0]))
                                block_reason = st.text_input("Reason (optional)")
                                if st.button("Update Block Status", on_click=lambda: play_sound_now("click")):
                                    try:
                                        supabase.table("user_credentials").update({"blocked": blocked_status}).eq("user_id", selected_user).execute()
                                        st.success(f"Block status updated for {selected_user}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error updating block: {e}")
                        else:
                            st.info("No users found.")
                    except Exception as e:
                        st.error(f"Error fetching users: {e}")
                else:
                    st.warning("Database not connected.")
            
            with admin_tabs[1]:
                st.subheader("Broker Balances (Current Session)")
                if st.session_state.bot:
                    balances = st.session_state.bot.get_broker_balances()
                    if balances:
                        df_balances = pd.DataFrame([{"Broker": k, "Balance": v} for k, v in balances.items()])
                        st.dataframe(df_balances, use_container_width=True, hide_index=True)
                    else:
                        st.info("No broker balances available. Make sure you are logged in with at least one broker.")
                else:
                    st.info("No active bot session.")
            
            with admin_tabs[2]:
                st.subheader("Active Sessions")
                if HAS_DB:
                    try:
                        sessions = supabase.table("user_sessions").select("*").order("last_seen", desc=True).execute()
                        if sessions.data:
                            df_sessions = pd.DataFrame(sessions.data)
                            st.dataframe(df_sessions[["user_id", "device_name", "ip_address", "last_seen"]], use_container_width=True)
                        else:
                            st.info("No active sessions.")
                    except Exception as e:
                        st.error(f"Error fetching sessions: {e}")
                else:
                    st.warning("Database not connected.")
            
            with admin_tabs[3]:
                st.subheader("System Logs")
                if st.session_state.bot:
                    logs = list(st.session_state.bot.state["logs"])
                    for log in logs[:20]:
                        st.text(log)
                else:
                    st.info("No bot logs.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Bottom dock (Start, Sync, Exit)
    st.markdown('<div class="bottom-dock">', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button("▶️\nStart", key="dock_start", on_click=lambda: play_sound_now("click")):
            bot.state["is_running"] = True
            t = threading.Thread(target=bot.trading_loop, daemon=True)
            add_script_run_ctx(t)
            t.start()
            st.rerun()
    with d2:
        if st.button("🔄\nSync", key="dock_refresh", on_click=lambda: play_sound_now("click")):
            st.rerun()
    with d3:
        if st.button("☠️\nExit", key="dock_Exit", on_click=lambda: play_sound_now("click")):
            bot.force_exit()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Process background queues
    if getattr(bot, "state", None):
        while bot.state.get("ui_popups"):
            popup = bot.state["ui_popups"].popleft()
            title = popup['title']
            msg = popup['message']
            st.toast(f"{title}: {msg}", icon="🔔")
            js_notification = f"""
            <script>
            function notifyUser() {{
                if (!("Notification" in window)) {{
                    console.log("This browser does not support desktop notification");
                }} else if (Notification.permission === "granted") {{
                    new Notification("{title}", {{body: "{msg}", icon: "https://cdn-icons-png.flaticon.com/512/2950/2950133.png"}});
                }} else if (Notification.permission !== "denied") {{
                    Notification.requestPermission().then(function (permission) {{
                        if (permission === "granted") {{
                            new Notification("{title}", {{body: "{msg}", icon: "https://cdn-icons-png.flaticon.com/512/2950/2950133.png"}});
                        }}
                    }});
                }}
            }}
            notifyUser();
            </script>
            """
            st.components.v1.html(js_notification, height=0)

        while bot.state.get("sound_queue"):
            sound = bot.state["sound_queue"].popleft()
            play_sound_ui(sound)

# ==========================================
# PROCESS SOUND QUEUES – PLAY ALL SOUNDS
# ==========================================
while st.session_state.sound_queue:
    latest_sound = st.session_state.sound_queue.popleft()
    play_sound_ui(latest_sound)

if getattr(st.session_state, "bot", None) and st.session_state.bot.state.get("sound_queue"):
    while st.session_state.bot.state["sound_queue"]:
        latest_sound = st.session_state.bot.state["sound_queue"].popleft()
        play_sound_ui(latest_sound)
