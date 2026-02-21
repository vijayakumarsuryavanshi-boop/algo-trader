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
import threading
import uuid
import yfinance as yf
import plotly.graph_objects as go
from SmartApi import SmartConnect
from streamlit.runtime.scriptrunner import add_script_run_ctx
from collections import deque

# ==========================================
# 1. SECURITY, CONFIG & HIGH-CONTRAST CSS
# ==========================================
CRED_FILE = "secure_creds.json"
TRADE_FILE = "persistent_trades.csv"

st.markdown("""
<style>
    /* Neon Dark Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        background-image: linear-gradient(180deg, #0b0f19 0%, #111827 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #38bdf8 !important; }
    
    /* High Contrast for Main Area (Black Text in White Boxes) */
    .stDataFrame, .stTable, .stDataFrame * {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stAlert { color: #000000 !important; }
    
    /* Pad the bottom of the app so you can scroll past the bottom dock */
    .main .block-container { padding-bottom: 100px; }
</style>
""", unsafe_allow_html=True)

def load_creds():
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, 'r') as f: return json.load(f)
        except Exception: pass
    return {"client_id": "", "pwd": "", "totp_secret": "", "api_key": ""}

def save_creds(client_id, pwd, totp_secret, api_key):
    with open(CRED_FILE, 'w') as f:
        json.dump({"client_id": client_id, "pwd": pwd, "totp_secret": totp_secret, "api_key": api_key}, f)

def save_trade(trade_record):
    df_new = pd.DataFrame([trade_record])
    if not os.path.exists(TRADE_FILE): df_new.to_csv(TRADE_FILE, index=False)
    else: df_new.to_csv(TRADE_FILE, mode='a', header=False, index=False)

DEFAULT_LOTS = {"NIFTY": 75, "BANKNIFTY": 30, "SENSEX": 20, "CRUDEOIL": 100, "NATURALGAS": 1250, "GOLD": 100, "SILVER": 30}
YF_TICKERS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}

# ==========================================
# 2. MARKET STATUS & TECHNICALS (WITH ATR)
# ==========================================
def get_market_status():
    now_ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    if now_ist.weekday() >= 5: return False, "Market Closed (Weekend)"
    if dt.time(9, 15) <= now_ist.time() <= dt.time(15, 30): return True, "Market Live 🟢"
    return False, "Market Closed (After Hours)"

def check_btst_stbt(df):
    if df is None or len(df) < 5: return "NO DATA"
    df = df.copy()
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    last = df.iloc[-1]
    
    if last['close'] > last['ema9'] > last['ema21'] and (last['high'] - last['close']) < (last['close'] - last['open']):
        return "🔥 BTST Suggested"
    elif last['close'] < last['ema9'] < last['ema21'] and (last['close'] - last['low']) < (last['open'] - last['close']):
        return "🩸 STBT Suggested"
    return "⚖️ Neutral (No Hold)"

class TechnicalAnalyzer:
    def get_atr(self, df, period=14):
        df = df.copy()
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        return df['tr'].rolling(period).mean().iloc[-1]

    def apply_vwap_ema_strategy(self, df, short_ema=9, long_ema=21):
        if df is None or len(df) < long_ema: return "WAIT", "WAIT", 0, 0, df, 0
        df = df.copy()
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['ema_short'] = df['close'].ewm(span=short_ema, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=long_ema, adjust=False).mean()
        atr = self.get_atr(df)
        
        signal, trend = "WAIT", "FLAT"
        if df['ema_short'].iloc[-1] > df['ema_long'].iloc[-1] and df['close'].iloc[-1] > df['vwap'].iloc[-1]:
            signal, trend = "BUY_CE", "BULLISH 🟢"
        elif df['ema_short'].iloc[-1] < df['ema_long'].iloc[-1] and df['close'].iloc[-1] < df['vwap'].iloc[-1]:
            signal, trend = "BUY_PE", "BEARISH 🔴"
            
        return trend, signal, df['vwap'].iloc[-1], df['ema_short'].iloc[-1], df, atr

    def apply_fvg_strategy(self, df, momentum_mult=1.5):
        if df is None or len(df) < 5: return "WAIT", "WAIT", 0, 0, df, 0
        df = df.copy()
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean() 
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        atr = self.get_atr(df)
        
        bullish_fvg = df['low'] > df['high'].shift(2)
        bearish_fvg = df['high'] < df['low'].shift(2)
        body_size = abs(df['close'].shift(1) - df['open'].shift(1))
        avg_body = abs(df['close'] - df['open']).rolling(10).mean()
        strong_disp = body_size > (avg_body * momentum_mult)
        
        signal, trend = "WAIT", "CONSOLIDATING"
        if (bullish_fvg & strong_disp).iloc[-1]: signal, trend = "BUY_CE", "FVG BULLISH 🟢"
        elif (bearish_fvg & strong_disp).iloc[-1]: signal, trend = "BUY_PE", "FVG BEARISH 🔴"
            
        return trend, signal, df['vwap'].iloc[-1], df['ema_short'].iloc[-1], df, atr

# ==========================================
# 3. CORE BOT ENGINE
# ==========================================
class SniperBot:
    def __init__(self, api_key="", client_id="", pwd="", totp_secret="", is_mock=False):
        self.api_key, self.client_id, self.pwd, self.totp_secret = api_key, client_id, pwd, totp_secret
        self.api, self.token_map, self.is_mock = None, None, is_mock
        self.analyzer = TechnicalAnalyzer()
        self.state = {
            "is_running": False, "order_in_flight": False, "active_trade": None,
            "logs": deque(maxlen=50), "current_trend": "WAIT", "current_signal": "WAIT",
            "spot": 0.0, "vwap": 0.0, "ema": 0.0, "atr": 0.0, "latest_data": None,
            "global_alerts": deque(maxlen=5), 
            "loop_count": 0
        }
        self.settings = {}

    def log(self, msg):
        ts = dt.datetime.now().strftime('%H:%M:%S')
        self.state["logs"].appendleft(f"[{ts}] {msg}")

    def login(self):
        if self.is_mock: return True
        try:
            obj = SmartConnect(api_key=self.api_key)
            token = pyotp.TOTP(self.totp_secret).now()
            res = obj.generateSession(self.client_id, self.pwd, token)
            if res['status']:
                self.api = obj
                return True
            return False
        except Exception: return False

    def fetch_master(self):
        if self.is_mock: return pd.DataFrame() 
        try:
            url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            df = pd.DataFrame(requests.get(url).json())
            df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
            df['strike'] = pd.to_numeric(df['strike'], errors='coerce') / 100 
            self.token_map = df
            return df
        except Exception: return None

    def get_live_price(self, exchange, symbol, token):
        if self.is_mock:
            if symbol in YF_TICKERS:
                try:
                    data = yf.Ticker(YF_TICKERS[symbol]).history(period="1d", interval="1m")
                    if not data.empty: return float(data['Close'].iloc[-1])
                except: pass
            return float(np.random.uniform(20000, 22100)) 
        if not self.api: return None
        try:
            res = self.api.ltpData(exchange, symbol, str(token))
            if res['status']: return float(res['data']['ltp'])
        except: return None

    def get_historical_data(self, exchange, token, symbol="NIFTY", interval="5m"):
        if self.is_mock:
            if symbol in YF_TICKERS:
                try:
                    df = yf.Ticker(YF_TICKERS[symbol]).history(period="5d", interval=interval)
                    if not df.empty:
                        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                        return df
                except: pass
            times = [dt.datetime.now() - dt.timedelta(minutes=5*i) for i in range(50)][::-1]
            base = 22000
            close_prices = base + np.random.normal(0, 10, 50).cumsum()
            df = pd.DataFrame({'timestamp': times, 'open': close_prices - np.random.uniform(0, 5, 50), 'high': close_prices + np.random.uniform(0, 10, 50), 'low': close_prices - np.random.uniform(0, 10, 50), 'close': close_prices, 'volume': np.random.randint(1000, 50000, 50)})
            df.index = df['timestamp']
            return df

        if not self.api: return None
        try:
            todate = dt.datetime.now()
            fromdate = todate - dt.timedelta(minutes=300)
            res = self.api.getCandleData({"exchange": exchange, "symboltoken": str(token), "interval": "FIVE_MINUTE", "fromdate": fromdate.strftime("%Y-%m-%d %H:%M"), "todate": todate.strftime("%Y-%m-%d %H:%M")})
            if res and res.get('status') and res.get('data'):
                df = pd.DataFrame(res['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.index = df['timestamp']
                return df
        except Exception: pass
        return None

    def place_real_order(self, symbol, token, qty, side="BUY", exchange="NFO"):
        if self.is_mock: return "MOCK_" + uuid.uuid4().hex[:6].upper()
        try:
            return self.api.placeOrder({"variety": "NORMAL", "tradingsymbol": symbol, "symboltoken": str(token), "transactiontype": side, "exchange": exchange, "ordertype": "MARKET", "producttype": "INTRADAY", "duration": "DAY", "quantity": str(qty)})
        except Exception: return None

    def get_strike(self, symbol, spot, signal, max_premium):
        if self.is_mock: 
            opt_type = "CE" if "BUY_CE" in signal else "PE"
            mock_expiry = (pd.Timestamp.today() + pd.Timedelta(days=2)).strftime('%d%b').upper()
            return f"{symbol}{mock_expiry}{int(spot)}{opt_type}", "12345", "NFO", 100.0 
            
        df = self.token_map
        if df is None or df.empty: return None, None, None, 0.0
        
        is_ce = "BUY_CE" in signal
        opt_type = "CE" if is_ce else "PE"
        exch_list, valid_instruments = ["NFO"], ['OPTIDX', 'OPTSTK']
        
        today = pd.Timestamp.today().normalize()
        mask = (df['name'] == symbol) & (df['exch_seg'].isin(exch_list)) & (df['expiry'] >= today) & (df['symbol'].str.endswith(opt_type)) & (df['instrumenttype'].isin(valid_instruments))
        subset = df[mask].copy()
        if subset.empty: return None, None, None, 0.0
        
        closest_expiry = subset['expiry'].min()
        if self.settings.get('hero_zero') and closest_expiry.date() != pd.Timestamp.today().date(): return None, None, None, 0.0
            
        subset = subset[subset['expiry'] == closest_expiry]
        candidates = subset[subset['strike'] >= spot].sort_values('strike', ascending=True) if is_ce else subset[subset['strike'] <= spot].sort_values('strike', ascending=False)
            
        actual_max_prem = self.settings.get('hz_premium', 15) if self.settings.get('hero_zero') else max_premium
        for _, row in candidates.head(15).iterrows():
            ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
            if ltp and ltp <= actual_max_prem: return row['symbol'], row['token'], row['exch_seg'], ltp
                
        return None, None, None, 0.0

    def check_global_spikes(self):
        watchlist = ["NIFTY", "BANKNIFTY", "SENSEX", "CRUDEOIL"]
        current_index = self.settings.get('index')
        for sym in watchlist:
            if sym == current_index: continue 
            df = self.get_historical_data("NFO" if sym!="CRUDEOIL" else "MCX", "12345", symbol=sym, interval="5m")
            if df is not None and not df.empty:
                trend, sig, _, _, _, _ = self.analyzer.apply_vwap_ema_strategy(df)
                if sig != "WAIT":
                    icon = "🚀" if "CE" in sig else "🩸"
                    self.state["global_alerts"].append(f"{icon} **FOMO ALERT:** {sym} is showing a {trend} signal!")

    def emergency_kill(self):
        self.state["is_running"] = False
        trade = self.state["active_trade"]
        if trade:
            if not self.settings.get("paper_mode", True) and not self.is_mock:
                self.place_real_order(trade['symbol'], trade['token'], trade['qty'], "SELL", trade['exch'])
            save_trade({"Date": dt.date.today().strftime('%Y-%m-%d'), "Time": dt.datetime.now().strftime('%H:%M:%S'), "Symbol": trade['symbol'], "Type": trade['type'], "Qty": trade['qty'], "Entry Price": trade['entry'], "Exit Price": "KILL", "PnL (₹)": 0.0})
            self.state["active_trade"] = None

    def trading_loop(self):
        self.log("Background scalping thread started.")
        while self.state["is_running"]:
            try:
                is_open, mkt_msg = get_market_status()
                if not is_open:
                    self.log(f"System paused. {mkt_msg}")
                    time.sleep(10)
                    continue

                self.state["loop_count"] += 1
                s = self.settings
                index, timeframe = s['index'], s['timeframe']
                paper = s['paper_mode']
                strategy = s['strategy']
                
                if self.state["loop_count"] % 15 == 0: self.check_global_spikes()

                cutoff_time = dt.time(15, 15)
                if self.is_mock: cutoff_time = dt.time(23, 59) 
                
                spot, base_lot_size = None, 1
                
                if not self.is_mock:
                    df_map = self.token_map
                    today = pd.Timestamp.today().normalize()
                    futs = df_map[(df_map['name'] == index) & (df_map['exch_seg'] == 'NFO') & (df_map['instrumenttype'] == 'FUTIDX') & (df_map['expiry'] >= today)]
                    if not futs.empty:
                        best_fut = futs[futs['expiry'] == futs['expiry'].min()].iloc[0]
                        spot = self.get_live_price(best_fut['exch_seg'], best_fut['symbol'], best_fut['token'])
                        df_candles = self.get_historical_data(best_fut['exch_seg'], best_fut['token'], interval=timeframe)
                        base_lot_size = int(best_fut.get('lotsize', DEFAULT_LOTS.get(index, 10)))
                else:
                    spot = self.get_live_price("NFO", index, "12345")
                    df_candles = self.get_historical_data("MOCK", "12345", symbol=index, interval=timeframe)
                    base_lot_size = DEFAULT_LOTS.get(index, 25)
                
                if spot and df_candles is not None and not df_candles.empty:
                    self.state["spot"] = spot
                    
                    if strategy == "Institutional FVG + SMC":
                        trend, signal, vwap, ema, df_chart, current_atr = self.analyzer.apply_fvg_strategy(df_candles)
                    elif strategy == "Combined (Convergence)":
                        t1, s1, vwap, ema, df_chart, current_atr = self.analyzer.apply_vwap_ema_strategy(df_candles)
                        t2, s2, _, _, _, _ = self.analyzer.apply_fvg_strategy(df_candles)
                        if s1 == s2 and s1 != "WAIT": signal, trend = s1, f"STRONG {t1} + FVG"
                        else: signal, trend = "WAIT", "MIXED SIGNALS"
                    else:
                        trend, signal, vwap, ema, df_chart, current_atr = self.analyzer.apply_vwap_ema_strategy(df_candles)

                    self.state.update({"current_trend": trend, "current_signal": signal, "vwap": vwap, "ema": ema, "atr": current_atr, "latest_data": df_chart})

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

                                    if not paper and not self.is_mock:
                                        order_id = self.place_real_order(strike_sym, strike_token, qty, "BUY", strike_exch)
                                        if order_id: 
                                            self.log(f"⚡ REAL ENTRY: {strike_sym} | Qty: {qty}")
                                            self.state["active_trade"] = {"symbol": strike_sym, "token": strike_token, "exch": strike_exch, "type": "CE" if "CE" in strike_sym else "PE", "entry": entry_ltp, "highest_price": entry_ltp, "qty": qty, "sl": dynamic_sl, "tgt": dynamic_tgt}
                                    else:
                                        self.log(f"📝 PAPER ENTRY: {strike_sym} @ {entry_ltp} | Qty: {qty}")
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
                                self.log(f"✅ AUTO EXIT {trade['symbol']} @ {ltp:.2f} | PnL: ₹{round(pnl, 2)}")
                                save_trade({"Date": dt.date.today().strftime('%Y-%m-%d'), "Time": dt.datetime.now().strftime('%H:%M:%S'), "Symbol": trade['symbol'], "Type": trade['type'], "Qty": trade['qty'], "Entry Price": trade['entry'], "Exit Price": ltp, "PnL (₹)": round(pnl, 2)})
                                self.state["active_trade"] = None
            except Exception as e:
                self.log(f"Thread Error: {str(e)}")
            time.sleep(2)

# ==========================================
# 4. STREAMLIT UI 
# ==========================================
st.set_page_config(page_title="Pro Scalper Bot", page_icon="⚡", layout="wide")
is_mkt_open, mkt_status_msg = get_market_status()

if 'bot' not in st.session_state: st.session_state.bot = None

if st.session_state.bot and st.session_state.bot.state["global_alerts"]:
    while st.session_state.bot.state["global_alerts"]:
        st.toast(st.session_state.bot.state["global_alerts"].popleft(), icon="🔥")

with st.sidebar:
    st.header("🔐 Connection Setup")
    auth_mode = st.radio("Operating Mode", ["📝 Paper Trading", "⚡ Real Trading"])
    
    if not st.session_state.bot:
        if auth_mode == "⚡ Real Trading":
            creds = load_creds()
            st.info("API Key is never saved locally.")
            # Restored inputs with 'Remember' logic
            API_KEY = st.text_input("SmartAPI Key", value=creds.get("api_key", ""), type="password")
            CLIENT_ID = st.text_input("Client ID", value=creds.get("client_id", ""))
            PIN = st.text_input("PIN", value=creds.get("pwd", ""), type="password")
            TOTP = st.text_input("TOTP secret", value=creds.get("totp_secret", ""), type="password")
            
            SAVE_CREDS = st.checkbox("Remember ID & PIN", value=True)
            
            if st.button("Connect to Live Exchange", type="primary"):
                temp_bot = SniperBot(API_KEY, CLIENT_ID, PIN, TOTP, is_mock=False)
                with st.spinner("Authenticating..."):
                    if temp_bot.login():
                        # Save credentials if checkbox is checked
                        if SAVE_CREDS: 
                            save_creds(CLIENT_ID, PIN, TOTP, API_KEY)
                        st.session_state.bot = temp_bot
                        temp_bot.fetch_master()
                        st.rerun()
                    else:
                        st.error("Login Failed. Check credentials.")
        else:
            if st.button("Start Paper Session", type="primary"):
                temp_bot = SniperBot(is_mock=True)
                temp_bot.login()
                st.session_state.bot = temp_bot
                st.rerun()
    else:
        st.success(f"Connected: {'Paper Mode' if st.session_state.bot.is_mock else st.session_state.bot.client_id}")
        if st.button("Logout & Clear"):
            st.session_state.bot.state["is_running"] = False
            st.session_state.clear()
            st.rerun()

    st.divider()
    st.header("⚙️ Strategy & Setup")
    STRATEGY = st.selectbox("Trading Strategy", ["VWAP + EMA", "Institutional FVG + SMC", "Combined (Convergence)"])
    INDEX = st.selectbox("Watchlist", list(DEFAULT_LOTS.keys()))
    TIMEFRAME = st.selectbox("Timeframe", ["1m", "3m", "5m"], index=2)
    
    st.header("🛡️ Risk Management")
    LOTS = st.number_input("Base Lots", 1, 100, 1)
    MAX_CAPITAL = st.number_input("Max Capital / Trade (₹)", 1000, 500000, 15000, step=1000)
    SL_PTS = st.number_input("Fallback Stop Loss (Pts)", 5, 200, 20)
    TSL_PTS = st.number_input("Trailing Stop Loss (Pts)", 5, 200, 15)
    TGT_PTS = st.number_input("Fallback Target (Points)", 10, 500, 40)
    PAPER = st.toggle("📝 Paper Trade Execution", True, disabled=True if (st.session_state.bot and st.session_state.bot.is_mock) else False)
    
    st.header("🚀 Hero/Zero Mode")
    HERO_ZERO = st.toggle("Enable Hero/Zero", False)
    HZ_PREMIUM = st.number_input("Max H/Z Premium (₹)", 1, 100, 15)

if not st.session_state.bot:
    st.title("Welcome to Pro Scalper Bot ⚡")
    st.warning("Please configure your connection in the sidebar to begin.")
else:
    bot = st.session_state.bot
    bot.settings = {"strategy": STRATEGY, "index": INDEX, "timeframe": TIMEFRAME, "lots": LOTS, "max_capital": MAX_CAPITAL, "sl_pts": SL_PTS, "tsl_pts": TSL_PTS, "tgt_pts": TGT_PTS, "paper_mode": PAPER, "hero_zero": HERO_ZERO, "hz_premium": HZ_PREMIUM}

    if not is_mkt_open: st.error(f"😴 {mkt_status_msg}")
        
    tab1, tab2, tab3 = st.tabs(["⚡ Live Dashboard", "🔎 Scanners", "📜 PnL Reports"])

    # ==========================================
    # TAB 1: LIVE DASHBOARD
    # ==========================================
    with tab1:
        st.subheader(f"Trading Terminal: {INDEX}")
        
        c1, c2, c3 = st.columns([1, 1, 3])
        with c1:
            if st.button("▶️ START ENGINE", use_container_width=True, type="primary"):
                if not bot.state["is_running"]:
                    bot.state["is_running"] = True
                    t = threading.Thread(target=bot.trading_loop, daemon=True)
                    add_script_run_ctx(t)
                    t.start()
                    st.rerun()
        with c2:
            if st.button("🛑 STOP ENGINE", use_container_width=True): bot.state["is_running"] = False

        if bot.state["is_running"]:
            if not is_mkt_open and bot.state["latest_data"] is None:
                df_eod = bot.get_historical_data("MOCK", "12345", symbol=INDEX, interval=TIMEFRAME)
                if df_eod is not None and not df_eod.empty:
                    bot.state["spot"] = df_eod['close'].iloc[-1]
                    t, s, v, e, df_c, atr = bot.analyzer.apply_vwap_ema_strategy(df_eod)
                    bot.state.update({"current_trend": t, "current_signal": s, "vwap": v, "ema": e, "atr": atr, "latest_data": df_c})

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"LTP", f"₹ {round(bot.state['spot'], 2)}")
            m2.metric("VWAP", f"₹ {round(bot.state['vwap'], 2)}")
            m3.metric("Volatility (ATR)", f"{round(bot.state['atr'], 2)}")
            m4.metric("Market Sentiment", bot.state["current_trend"])
            
            chart_col, trade_col = st.columns([3, 1])
            with chart_col:
                df = bot.state["latest_data"]
                if df is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
                    if 'vwap' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], name='VWAP', line=dict(color='orange', dash='dash')))
                    if 'ema_short' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['ema_short'], name='EMA 9', line=dict(color='green')))
                    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=400, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            with trade_col:
                st.markdown("### Open Position")
                if bot.state["active_trade"]:
                    t = bot.state["active_trade"]
                    pnl = t.get('floating_pnl', 0.0)
                    ltp = t.get('current_ltp', t['entry'])
                    indicator = "🟢" if pnl >= 0 else "🔴"
                    
                    st.success(f"**Strike:** `{t['symbol']}`\n\n**Entry:** `₹{t['entry']:.2f}`\n\n**LTP:** `₹{ltp:.2f}`\n\n**PnL:** {indicator} **₹{round(pnl, 2)}**")
                    st.info(f"🛑 **SL:** `₹{t['sl']:.2f}` *(PA Dynamic)*\n\n🎯 **TGT:** `₹{t['tgt']:.2f}`")
                    
                    if st.button("Close Trade Manually"):
                        if not bot.settings['paper_mode'] and not bot.is_mock: bot.place_real_order(t['symbol'], t['token'], t['qty'], "SELL", t['exch'])
                        save_trade({"Date": dt.date.today().strftime('%Y-%m-%d'), "Time": dt.datetime.now().strftime('%H:%M:%S'), "Symbol": t['symbol'], "Type": t['type'], "Qty": t['qty'], "Entry Price": t['entry'], "Exit Price": ltp, "PnL (₹)": round(pnl, 2)})
                        bot.state["active_trade"] = None
                        st.rerun()
                else:
                    st.markdown("*No active positions.*")

            if is_mkt_open: 
                time.sleep(2)
                st.rerun()
        else:
            st.warning("Engine is currently stopped. Click Start to begin tracking.")

    # ==========================================
    # TAB 2: SCANNERS & BTST SUGGESTIONS
    # ==========================================
    with tab2:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("🔥 F&O OI Spurt Scanner")
            if st.button("🔍 Run Live NSE Scan Now"):
                try:
                    from nsepython import nsefetch
                    with st.spinner("Fetching Live OI Data from NSE..."):
                        payload = nsefetch('https://www.nseindia.com/api/live-analysis-oi-spurts')
                        if payload and 'data' in payload:
                            df_oi = pd.DataFrame(payload.get('data', []))
                            df_oi['pChange'] = pd.to_numeric(df_oi['pChange'], errors='coerce')
                            df_oi['per_chnge_oi'] = pd.to_numeric(df_oi['per_chnge_oi'], errors='coerce')
                            
                            st.markdown("#### 🟢 Long Buildups")
                            longs = df_oi[(df_oi['pChange'] > 0.5) & (df_oi['per_chnge_oi'] > 2)].head(10)
                            st.dataframe(longs[['symbol', 'latest_price', 'pChange', 'per_chnge_oi']], use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Scanner Failed. Ensure 'nsepython' is installed or try during market hours.")

        with colB:
            st.subheader(f"📡 Multi-Stock BTST Scanner")
            if st.button("🔄 Scan Market"):
                with st.spinner("Analyzing Watchlist..."):
                    scan_results = []
                    for sym in DEFAULT_LOTS.keys():
                        df = bot.get_historical_data("MOCK", "12345", symbol=sym, interval="1d" if not is_mkt_open else "5m")
                        if df is not None and not df.empty:
                            trend, sig, v, e, _, _ = bot.analyzer.apply_vwap_ema_strategy(df)
                            btst_sig = check_btst_stbt(df) if not is_mkt_open else "N/A"
                            scan_results.append({"Symbol": sym, "LTP": round(df['close'].iloc[-1], 2), "Trend": trend, "BTST/STBT": btst_sig})
                    st.dataframe(pd.DataFrame(scan_results), use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 3: PNL & LOGS
    # ==========================================
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
                if st.button("🗑️ Clear Log Data"):
                    os.remove(TRADE_FILE)
                    st.rerun()
            else:
                st.info("No trade history available.")

# ==========================================
# FULL-WIDTH BOTTOM NAVIGATION DOCK
# ==========================================
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    btn1, btn2 = st.columns(2)
    if btn1.button("🏠 Home", use_container_width=True): st.rerun()
    if btn2.button("👆 Quick Login", use_container_width=True):
        if not st.session_state.bot:
            st.toast("Simulating biometric login...", icon="🔓")
            creds = load_creds()
            temp_bot = SniperBot(creds.get("api_key", ""), creds.get("client_id", ""), creds.get("pwd", ""), creds.get("totp_secret", ""), is_mock=True)
            if temp_bot.login():
                st.session_state.bot = temp_bot
                temp_bot.fetch_master()
                st.rerun()
        else:
            st.toast("Already logged in!", icon="✅")

components.html(
    """
    <script>
    const doc = window.parent.document;
    const blocks = doc.querySelectorAll('div[data-testid="stHorizontalBlock"]');
    if (blocks.length > 0) {
        const navBlock = blocks[blocks.length - 1]; 
        const wrapper = navBlock.closest('.element-container');
        if (wrapper) {
            wrapper.style.position = 'fixed';
            wrapper.style.bottom = '0px'; 
            wrapper.style.left = '0px';
            wrapper.style.width = '100%';
            wrapper.style.maxWidth = '100%';
            wrapper.style.zIndex = '999999';
            wrapper.style.backgroundColor = 'rgba(11, 15, 25, 1)';
            wrapper.style.padding = '10px 15px';
            wrapper.style.borderTop = '2px solid #38bdf8';
        }
    }
    </script>
    """,
    height=0,
)
