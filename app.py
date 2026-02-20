import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import requests
import pyotp
from SmartApi import SmartConnect

# ==========================================
# 1. TECHNICAL ANALYZER (VOLUME BREAKOUT)
# ==========================================
class TechnicalAnalyzer:
    def calculate_volume_breakout(self, df, vol_length=20, vol_multiplier=1.8):
        if df is None or len(df) < vol_length + 1: return "WAIT", "WAIT"
        df = df.copy()
        df['avg_vol'] = df['volume'].rolling(window=vol_length).mean().shift(1)
        current_close, current_open = df['close'].iloc[-1], df['open'].iloc[-1]
        current_vol, prev_avg_vol = df['volume'].iloc[-1], df['avg_vol'].iloc[-1]

        if current_vol == 0 or pd.isna(prev_avg_vol) or prev_avg_vol == 0:
            return "WAIT", "WAIT"

        if current_vol > (prev_avg_vol * vol_multiplier):
            if current_close > current_open: return "VOL SURGE 🟢", "BUY_CE"
            elif current_close < current_open: return "VOL DUMP 🔴", "BUY_PE"
        return "WAIT", "WAIT"

# ==========================================
# 2. CORE BOT ENGINE
# ==========================================
class SniperBot:
    def __init__(self, api_key, client_id, pwd, totp_secret):
        self.api_key, self.client_id, self.pwd, self.totp_secret = api_key, client_id, pwd, totp_secret
        self.api, self.token_map = None, None
        self.analyzer = TechnicalAnalyzer()

    def login(self):
        try:
            obj = SmartConnect(api_key=self.api_key)
            token = pyotp.TOTP(self.totp_secret).now()
            res = obj.generateSession(self.client_id, self.pwd, token)
            if res['status']:
                self.api = obj
                return True
            return False
        except Exception as e:
            st.error(f"Login Exception: {e}")
            return False

    def fetch_master(self):
        try:
            df = pd.DataFrame(requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json").json())
            df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
            df['strike'] = pd.to_numeric(df['strike'], errors='coerce') / 100 
            self.token_map = df
            return df
        except Exception: return None

    def get_live_price(self, exchange, symbol, token):
        if not self.api: return None
        try:
            res = self.api.ltpData(exchange, symbol, str(token))
            if res['status']: return float(res['data']['ltp'])
        except: return None
        return None

    def get_historical_data(self, exchange, token, interval="FIVE_MINUTE", minutes=300):
        if not self.api: return None
        try:
            todate = dt.datetime.now()
            fromdate = todate - dt.timedelta(minutes=minutes)
            res = self.api.getCandleData({
                "exchange": exchange, "symboltoken": str(token), "interval": interval,
                "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"), "todate": todate.strftime("%Y-%m-%d %H:%M")
            })
            if res and res.get('status') and res.get('data'):
                return pd.DataFrame(res['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception: pass
        return None

    def get_strike(self, symbol, spot, signal):
        df = self.token_map
        if df is None: return None, None, None
        opt_type = "CE" if "BUY_CE" in signal else "PE"
        name = symbol
        exch_list = ["NFO"]
        valid_instruments = ['OPTIDX', 'OPTSTK']
        
        if symbol in ["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS"]: 
            exch_list, valid_instruments = ["MCX", "NCO"], ['OPTFUT', 'OPTCOM', 'OPTENR', 'OPTBLN']
        elif symbol == "SENSEX": 
            exch_list, valid_instruments = ["BFO", "BSE"], ['OPTIDX']
            
        today = pd.Timestamp.today().normalize()
        mask = (df['name'] == name) & (df['exch_seg'].isin(exch_list)) & (df['expiry'] >= today) & (df['symbol'].str.endswith(opt_type)) & (df['instrumenttype'].isin(valid_instruments))
        subset = df[mask].copy()
        if subset.empty: return None, None, None
        subset = subset[subset['expiry'] == subset['expiry'].min()]
        subset['diff'] = abs(subset['strike'] - spot)
        best = subset.sort_values('diff').iloc[0]
        return best['symbol'], best['token'], best['exch_seg']

# ==========================================
# 3. STREAMLIT UI & SECURITY
# ==========================================
st.set_page_config(page_title="Pro Algo Trader", page_icon="📈", layout="wide")

for key in ['auth', 'bot_active', 'logs']:
    if key not in st.session_state: st.session_state[key] = False if key != 'logs' else []
if 'active_trade' not in st.session_state: st.session_state.active_trade = None
if 'current_trend' not in st.session_state: st.session_state.current_trend = "WAIT"
if 'current_signal' not in st.session_state: st.session_state.current_signal = "WAIT"

def log(msg): 
    st.session_state.logs.insert(0, f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
    if len(st.session_state.logs) > 50: st.session_state.logs.pop()

with st.sidebar:
    st.header("🔐 Secure Connect")
    
    if not st.session_state.auth:
        st.info("Log in with your own Angel One credentials.")
        # Notice that value defaults have been completely removed
        API_KEY = st.text_input("API Key", type="password")
        CLIENT_ID = st.text_input("Client ID")
        PIN = st.text_input("PIN", type="password")
        TOTP = st.text_input("TOTP secret", type="password")
        
        if st.button("Connect to Exchange", type="primary"):
            temp_bot = SniperBot(API_KEY, CLIENT_ID, PIN, TOTP)
            with st.spinner("Authenticating..."):
                if temp_bot.login():
                    st.session_state.bot = temp_bot
                    st.session_state.auth = True
                    temp_bot.fetch_master()
                    log("System Authenticated Successfully")
                    st.rerun()
                else:
                    st.error("Login Failed. Check credentials or IP whitelist.")
    else:
        st.success(f"Connected: {st.session_state.bot.client_id}")
        if st.button("Logout & Clear"):
            st.session_state.clear()
            st.rerun()

    st.divider()
    INDEX = st.selectbox("Watchlist", ["NIFTY", "BANKNIFTY", "SENSEX"])
    TIMEFRAME = st.selectbox("Timeframe", ["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE"], index=2)
    PAPER = st.toggle("📝 Paper Mode", True)

if not st.session_state.auth:
    st.title("Welcome to Pro Algo Trader")
    st.warning("Please connect using the sidebar with your Angel One details to begin.")
    st.stop()

tab1, tab2 = st.tabs(["⚡ Live Dashboard", "🔎 Scanner"])
bot = st.session_state.bot

with tab1:
    col1, col2 = st.columns(2)
    if col1.button("🟢 START BOT"): st.session_state.bot_active = True
    if col2.button("🔴 STOP BOT"): st.session_state.bot_active = False

    if st.session_state.bot_active:
        st.info("Bot is active. Waiting for setup...")
        
        df_map = bot.token_map
        today = pd.Timestamp.today().normalize()
        
        if INDEX in ["NIFTY", "BANKNIFTY", "SENSEX"]:
            exch_seg = "BFO" if INDEX == "SENSEX" else "NFO"
            mask = (df_map['name'] == INDEX) & (df_map['exch_seg'] == exch_seg) & (df_map['instrumenttype'] == 'FUTIDX')
        else:
            mask = (df_map['name'] == INDEX) & (df_map['exch_seg'].isin(['MCX', 'NCO'])) & (df_map['symbol'].str.contains('FUT'))
            
        futs = df_map[mask].copy()
        futs = futs[futs['expiry'] >= today]
        
        if not futs.empty:
            best_fut = futs[futs['expiry'] == futs['expiry'].min()].iloc[0]
            exch, spot_sym, token = best_fut['exch_seg'], best_fut['symbol'], best_fut['token']
            
            spot = bot.get_live_price(exch, spot_sym, token)
            
            if spot:
                df_candles = bot.get_historical_data(exch, token, interval=TIMEFRAME, minutes=300)
                if df_candles is not None:
                    trend, signal = bot.analyzer.calculate_volume_breakout(df_candles, vol_length=20, vol_multiplier=1.8)
                    st.session_state.current_trend = trend
                    st.session_state.current_signal = signal

                st.metric(f"Live {INDEX} Futures Price", spot)
                st.metric("Algo Action", st.session_state.current_signal)

                if st.session_state.active_trade is None and signal in ["BUY_CE", "BUY_PE"]:
                    strike_sym, strike_token, strike_exch = bot.get_strike(INDEX, spot, signal)
                    if strike_sym:
                        opt_ltp = bot.get_live_price(strike_exch, strike_sym, strike_token)
                        if opt_ltp:
                            log(f"📝 PAPER ENTRY: {strike_sym} @ {opt_ltp}")
                            st.session_state.active_trade = {"symbol": strike_sym, "token": strike_token, "entry": opt_ltp}
                
                elif st.session_state.active_trade is None and signal == "WAIT":
                    st.write("Looking for Volume Breakouts...")

                elif st.session_state.active_trade is not None:
                    trade = st.session_state.active_trade
                    st.success(f"Open Trade: {trade['symbol']} | Entry: {trade['entry']}")
                    if st.button("Close Trade Manually"):
                        log(f"Closed {trade['symbol']}")
                        st.session_state.active_trade = None
        else:
            st.error(f"Could not load Futures data for {INDEX}.")

        time.sleep(4)
        st.rerun()

    else:
        st.warning("Bot is currently stopped.")
        
    st.divider()
    st.subheader("📜 Logs")
    for l in st.session_state.logs: st.text(l)

with tab2:
    st.write("Run manual scans without locking up the dashboard.")
    if st.button("🔍 Scan Momentum Stocks"):
        was_active = st.session_state.bot_active
        st.session_state.bot_active = False 

        BASKET = ["HDFCBANK", "RELIANCE", "ICICIBANK", "INFY"]
        df_map = bot.token_map
        suggestions = []
        progress_bar = st.progress(0)
        
        if df_map is not None:
            eq_df = df_map[(df_map['exch_seg'] == 'NSE') & (df_map['symbol'].str.endswith('-EQ'))]
            for i, stock in enumerate(BASKET):
                row = eq_df[eq_df['name'] == stock]
                if not row.empty:
                    token = row.iloc[0]['token']
                    hist = bot.get_historical_data("NSE", token, "FIVE_MINUTE", 300)
                    if hist is not None and not hist.empty:
                        trend, signal = bot.analyzer.calculate_volume_breakout(hist, vol_length=20, vol_multiplier=1.8)
                        if signal != "WAIT":
                            suggestions.append({"Stock": stock, "Action": signal})
                time.sleep(0.4)
                progress_bar.progress((i + 1) / len(BASKET))
            
            if suggestions:
                st.dataframe(pd.DataFrame(suggestions))
            else:
                st.info("No volume breakouts found right now.")
        
        if was_active: 
            st.session_state.bot_active = True
            st.success("Scan complete. Live dashboard resumed.")
