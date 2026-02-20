import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import requests
import pyotp
from SmartApi import SmartConnect

# ==========================================
# TECHNICAL ANALYZER (UPGRADED LOGIC)
# ==========================================
class TechnicalAnalyzer:

    def calculate_indicators(self, df):
        df = df.copy()

        # EMA Trend Filter
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()

        # VWAP
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # ATR
        df['tr'] = df['high'] - df['low']
        df['atr'] = df['tr'].rolling(14).mean()

        # Volume Average
        df['avg_vol'] = df['volume'].rolling(20).mean().shift(1)

        return df

    def generate_signal(self, df, vol_multiplier=1.8):
        if df is None or len(df) < 50:
            return "WAIT", "WAIT", None

        df = self.calculate_indicators(df)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        trend = "FLAT"
        signal = "WAIT"

        # Trend Logic
        if last['ema20'] > last['ema50']:
            trend = "UPTREND"
        elif last['ema20'] < last['ema50']:
            trend = "DOWNTREND"

        # Volume Breakout
        if last['volume'] > last['avg_vol'] * vol_multiplier:

            # CE Conditions
            if (last['close'] > prev['close'] and
                last['close'] > last['vwap'] and
                trend == "UPTREND"):
                signal = "BUY_CE"

            # PE Conditions
            elif (last['close'] < prev['close'] and
                  last['close'] < last['vwap'] and
                  trend == "DOWNTREND"):
                signal = "BUY_PE"

        return trend, signal, last['atr']


# ==========================================
# BOT ENGINE
# ==========================================
class SniperBot:

    def __init__(self, api_key, client_id, pwd, totp_secret):
        self.api_key = api_key
        self.client_id = client_id
        self.pwd = pwd
        self.totp_secret = totp_secret
        self.api = None
        self.token_map = None
        self.analyzer = TechnicalAnalyzer()

    def login(self):
        try:
            obj = SmartConnect(api_key=self.api_key)
            token = pyotp.TOTP(self.totp_secret).now()
            res = obj.generateSession(self.client_id, self.pwd, token)
            if res['status']:
                self.api = obj
                return True
        except Exception as e:
            st.error(e)
        return False

    def fetch_master(self):
        df = pd.DataFrame(requests.get(
            "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        ).json())
        df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce') / 100
        self.token_map = df

    def get_live_price(self, exchange, symbol, token):
        res = self.api.ltpData(exchange, symbol, str(token))
        if res['status']:
            return float(res['data']['ltp'])
        return None

    def get_historical_data(self, exchange, token, interval, minutes=300):
        todate = dt.datetime.now()
        fromdate = todate - dt.timedelta(minutes=minutes)

        res = self.api.getCandleData({
            "exchange": exchange,
            "symboltoken": str(token),
            "interval": interval,
            "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"),
            "todate": todate.strftime("%Y-%m-%d %H:%M")
        })

        if res['status'] and res['data']:
            return pd.DataFrame(
                res['data'],
                columns=['timestamp','open','high','low','close','volume']
            )
        return None


# ==========================================
# STREAMLIT APP
# ==========================================
st.set_page_config(page_title="Pro Algo Trader", layout="wide")

if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.bot_active = False
    st.session_state.active_trade = None
    st.session_state.daily_pnl = 0
    st.session_state.trade_count = 0
    st.session_state.trade_history = []

# ------------------------------------------
# LOGIN
# ------------------------------------------
with st.sidebar:
    st.header("🔐 Login")

    API = st.text_input("API Key", type="password")
    CID = st.text_input("Client ID")
    PIN = st.text_input("PIN", type="password")
    TOTP = st.text_input("TOTP Secret", type="password")

    if st.button("Login"):
        bot = SniperBot(API, CID, PIN, TOTP)
        if bot.login():
            bot.fetch_master()
            st.session_state.bot = bot
            st.session_state.auth = True
            st.success("Connected")

if not st.session_state.auth:
    st.stop()

bot = st.session_state.bot

# ------------------------------------------
# SETTINGS
# ------------------------------------------
INDEX = st.selectbox("Index", ["NIFTY", "BANKNIFTY"])
TIMEFRAME = st.selectbox("Timeframe", ["FIVE_MINUTE"])
LOTS = st.number_input("Lots", 1, 10, 1)
DAILY_LOSS_LIMIT = -5000
MAX_TRADES = 5

# ------------------------------------------
# CONTROL
# ------------------------------------------
col1, col2 = st.columns(2)
if col1.button("START"):
    st.session_state.bot_active = True
if col2.button("STOP"):
    st.session_state.bot_active = False

# ------------------------------------------
# MAIN LOOP
# ------------------------------------------
if st.session_state.bot_active:

    # No trade first 10 min
    if dt.datetime.now().time() < dt.time(9,25):
        st.warning("Waiting for market stabilization...")
        st.stop()

    # Risk controls
    if st.session_state.daily_pnl <= DAILY_LOSS_LIMIT:
        st.error("Daily loss limit hit. Bot stopped.")
        st.session_state.bot_active = False
        st.stop()

    if st.session_state.trade_count >= MAX_TRADES:
        st.warning("Max trades reached today.")
        st.stop()

    # Fetch futures (simplified example for NIFTY)
    df_map = bot.token_map
    today = pd.Timestamp.today().normalize()

    futs = df_map[
        (df_map['name'] == INDEX) &
        (df_map['instrumenttype'] == 'FUTIDX') &
        (df_map['expiry'] >= today)
    ]

    if not futs.empty:
        fut = futs.sort_values("expiry").iloc[0]

        hist = bot.get_historical_data(
            fut['exch_seg'],
            fut['token'],
            TIMEFRAME,
            300
        )

        trend, signal, atr = bot.analyzer.generate_signal(hist)

        st.metric("Trend", trend)
        st.metric("Signal", signal)

        # Entry
        if st.session_state.active_trade is None and signal != "WAIT":

            ltp = bot.get_live_price(
                fut['exch_seg'],
                fut['symbol'],
                fut['token']
            )

            qty = LOTS * 75

            st.session_state.active_trade = {
                "entry": ltp,
                "sl": ltp - atr,
                "tgt": ltp + (atr * 2),
                "qty": qty
            }

            st.session_state.trade_count += 1

        # Exit Logic
        if st.session_state.active_trade is not None:

            trade = st.session_state.active_trade
            ltp = bot.get_live_price(
                fut['exch_seg'],
                fut['symbol'],
                fut['token']
            )

            # Trailing SL
            if ltp > trade['entry'] + atr:
                trade['sl'] = trade['entry']

            if ltp >= trade['tgt'] or ltp <= trade['sl']:

                pnl = (ltp - trade['entry']) * trade['qty']
                st.session_state.daily_pnl += pnl

                st.session_state.trade_history.append({
                    "Entry": trade['entry'],
                    "Exit": ltp,
                    "PnL": pnl
                })

                st.session_state.active_trade = None

    time.sleep(4)
    st.rerun()

# ------------------------------------------
# ANALYTICS
# ------------------------------------------
if st.session_state.trade_history:
    df = pd.DataFrame(st.session_state.trade_history)

    st.subheader("Performance")
    st.metric("Total PnL", round(df["PnL"].sum(),2))
    st.metric("Win Rate",
              round(len(df[df["PnL"]>0])/len(df)*100,2))

    df["Equity"] = df["PnL"].cumsum()
    st.line_chart(df["Equity"])
