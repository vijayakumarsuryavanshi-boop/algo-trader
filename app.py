import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import requests
import pyotp
import json
import os
import threading
from SmartApi import SmartConnect

# ==========================================
# 1. SECURITY, CONFIG & STORAGE
# ==========================================
CRED_FILE = "secure_creds.json"
TRADE_FILE = "persistent_trades.csv"

def load_creds():
    if os.path.exists(CRED_FILE):
        try:
            with open(CRED_FILE, 'r') as f: return json.load(f)
        except Exception: pass
    return {"client_id": "", "pwd": "", "totp_secret": ""}

def save_creds(client_id, pwd, totp_secret):
    with open(CRED_FILE, 'w') as f:
        json.dump({"client_id": client_id, "pwd": pwd, "totp_secret": totp_secret}, f)

def save_trade(trade_record):
    df_new = pd.DataFrame([trade_record])
    if not os.path.exists(TRADE_FILE):
        df_new.to_csv(TRADE_FILE, index=False)
    else:
        df_new.to_csv(TRADE_FILE, mode='a', header=False, index=False)

DEFAULT_LOTS = {"NIFTY": 75, "BANKNIFTY": 30, "SENSEX": 20, "CRUDEOIL": 100, "NATURALGAS": 1250, "GOLD": 100, "SILVER": 30}

# ==========================================
# 2. TECHNICAL ANALYZER (VWAP + EMA SCALPING)
# ==========================================
class TechnicalAnalyzer:
    def calculate_scalp_signals(self, df, vol_length=20, vol_multiplier=1.5, ema_length=9):
        if df is None or len(df) < vol_length + 1: return "WAIT", "WAIT", 0, 0
            
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['cum_tp_vol'] = (df['typical_price'] * df['volume']).groupby(df['date']).cumsum()
        df['cum_vol'] = df['volume'].groupby(df['date']).cumsum()
        df['vwap'] = (df['cum_tp_vol'] / df['cum_vol']).ffill() 
        
        df['ema'] = df['close'].ewm(span=ema_length, adjust=False).mean()
        df['avg_vol'] = df['volume'].rolling(window=vol_length).mean().shift(1)
        
        current_close, prev_close = df['close'].iloc[-1], df['close'].iloc[-2]
        current_open = df['open'].iloc[-1]
        current_vol, prev_avg_vol = df['volume'].iloc[-1], df['avg_vol'].iloc[-1]
        current_vwap, current_ema = df['vwap'].iloc[-1], df['ema'].iloc[-1]

        if current_vol == 0 or pd.isna(prev_avg_vol) or prev_avg_vol == 0:
            return "WAIT", "WAIT", current_vwap, current_ema

        trend = "FLAT"
        if current_close > prev_close and current_vol > prev_avg_vol: trend = "LONG BUILDUP 🟢"
        elif current_close < prev_close and current_vol > prev_avg_vol: trend = "SHORT BUILDUP 🔴"
        elif current_close > prev_close and current_vol < prev_avg_vol: trend = "SHORT COVERING 🟡"
        elif current_close < prev_close and current_vol < prev_avg_vol: trend = "LONG UNWINDING 🟠"

        signal = "WAIT"
        if current_vol > (prev_avg_vol * vol_multiplier):
            if current_close > current_open and current_ema > current_vwap: signal = "BUY_CE"
            elif current_close < current_open and current_ema < current_vwap: signal = "BUY_PE"
                
        return trend, signal, current_vwap, current_ema

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
            "logs": [], "current_trend": "WAIT", "current_signal": "WAIT",
            "spot": 0.0, "vwap": 0.0, "ema": 0.0, "vol_ratio": "0/0"
        }
        self.settings = {}

    def log(self, msg):
        timestamp = dt.datetime.now().strftime('%H:%M:%S')
        self.state["logs"].insert(0, f"[{timestamp}] {msg}")
        if len(self.state["logs"]) > 50: self.state["logs"].pop()

    def login(self):
        if self.is_mock:
            self.log("Started in Offline Mock Mode.")
            return True
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
            df = pd.DataFrame(requests.get("https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json").json())
            df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
            df['strike'] = pd.to_numeric(df['strike'], errors='coerce') / 100 
            self.token_map = df
            return df
        except Exception: return None

    def get_live_price(self, exchange, symbol, token):
        if self.is_mock: 
            return float(np.random.uniform(100, 120)) if ("CE" in str(symbol) or "PE" in str(symbol)) else float(np.random.uniform(2000, 22100))
        if not self.api: return None
        try:
            res = self.api.ltpData(exchange, symbol, str(token))
            if res['status']: return float(res['data']['ltp'])
        except: return None

    def get_historical_data(self, exchange, token, interval="FIVE_MINUTE", minutes=300):
        if self.is_mock:
            now = dt.datetime.now()
            times = [now - dt.timedelta(minutes=5*i) for i in range(50)][::-1]
            base_price = 22000
            close_prices = base_price + np.random.normal(0, 10, 50).cumsum()
            df = pd.DataFrame({
                'timestamp': times, 'open': close_prices - np.random.uniform(0, 5, 50),
                'high': close_prices + np.random.uniform(0, 10, 50), 'low': close_prices - np.random.uniform(0, 10, 50),
                'close': close_prices, 'volume': np.random.randint(1000, 50000, 50)
            })
            if np.random.random() > 0.7: df.at[49, 'volume'] = df['volume'].mean() * 3 
            return df

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

    def place_real_order(self, symbol, token, qty, side="BUY", exchange="NFO"):
        if self.is_mock: return "MOCK_" + str(np.random.randint(1000, 9999))
        try:
            orderparams = {
                "variety": "NORMAL", "tradingsymbol": symbol, "symboltoken": str(token),
                "transactiontype": side, "exchange": exchange, "ordertype": "MARKET",
                "producttype": "INTRADAY", "duration": "DAY", "quantity": str(qty)
            }
            return self.api.placeOrder(orderparams)
        except Exception: return None

    def get_strike(self, symbol, spot, signal, max_premium):
        if self.is_mock: 
            opt_type = "CE" if "BUY_CE" in signal else "PE"
            mock_expiry = (pd.Timestamp.today() + pd.Timedelta(days=2)).strftime('%d%b').upper()
            return f"{symbol}{mock_expiry}{int(spot)}{opt_type}", "12345", "NFO"
            
        df = self.token_map
        if df is None or df.empty: return None, None, None
        
        is_ce = "BUY_CE" in signal
        opt_type = "CE" if is_ce else "PE"
        exch_list, valid_instruments = ["NFO"], ['OPTIDX', 'OPTSTK']
        
        if symbol in ["CRUDEOIL", "GOLD", "SILVER", "NATURALGAS"]: 
            exch_list, valid_instruments = ["MCX", "NCO"], ['OPTFUT', 'OPTCOM', 'OPTENR', 'OPTBLN']
        elif symbol == "SENSEX": 
            exch_list, valid_instruments = ["BFO", "BSE"], ['OPTIDX']
            
        today = pd.Timestamp.today().normalize()
        mask = (df['name'] == symbol) & (df['exch_seg'].isin(exch_list)) & (df['expiry'] >= today) & (df['symbol'].str.endswith(opt_type)) & (df['instrumenttype'].isin(valid_instruments))
        subset = df[mask].copy()
        
        if subset.empty: return None, None, None
        
        closest_expiry = subset['expiry'].min()
        
        is_hz_mode = self.settings.get('hero_zero', False)
        if is_hz_mode and closest_expiry.date() != pd.Timestamp.today().date() and not self.is_mock:
            self.log(f"Hero/Zero Block: {symbol} does not expire today.")
            return None, None, None 
            
        subset = subset[subset['expiry'] == closest_expiry]
        
        actual_max_premium = self.settings.get('hz_premium', 15) if is_hz_mode else max_premium
        candidates = subset[subset['strike'] >= spot].sort_values('strike', ascending=True) if is_ce else subset[subset['strike'] <= spot].sort_values('strike', ascending=False)
            
        for _, row in candidates.head(15).iterrows():
            ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
            if ltp and ltp <= actual_max_premium: return row['symbol'], row['token'], row['exch_seg']
        return None, None, None

    def emergency_kill(self):
        self.state["is_running"] = False
        trade = self.state["active_trade"]
        if trade:
            if not self.settings.get("paper_mode", True) and not self.is_mock:
                self.place_real_order(trade['symbol'], trade['token'], trade['qty'], "SELL", trade['exch'])
            
            self.log(f"🚨 KILL SWITCH: Force closed {trade['symbol']}")
            save_trade({
                "Date": dt.date.today().strftime('%Y-%m-%d'), "Time": dt.datetime.now().strftime('%H:%M:%S'), 
                "Symbol": trade['symbol'], "Type": trade['type'], "Qty": trade['qty'], 
                "Entry Price": trade['entry'], "Exit Price": "KILL", "PnL (₹)": 0.0
            })
            self.state["active_trade"] = None

    def trading_loop(self):
        self.log("Background scalping thread started.")
        while self.state["is_running"]:
            try:
                s = self.settings
                index, timeframe = s['index'], s['timeframe']
                vol_mult, paper = s['vol_mult'], s['paper_mode']
                
                is_commodity = index in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"]
                cutoff_time = dt.time(23, 15) if is_commodity else dt.time(15, 15)
                current_time = dt.datetime.now().time()
                
                # IGNORE TIME CUTOFF IN MOCK MODE FOR TESTING
                if self.is_mock:
                    cutoff_time = dt.time(23, 59) 
                
                spot, base_lot_size = None, 1
                
                if not self.is_mock:
                    df_map = self.token_map
                    today = pd.Timestamp.today().normalize()
                    
                    if index in ["NIFTY", "BANKNIFTY"]: mask = (df_map['name'] == index) & (df_map['exch_seg'] == 'NFO') & (df_map['instrumenttype'] == 'FUTIDX')
                    elif index == "SENSEX": mask = (df_map['name'] == index) & (df_map['exch_seg'] == 'BFO') & (df_map['instrumenttype'] == 'FUTIDX')
                    elif is_commodity: mask = (df_map['name'] == index) & (df_map['exch_seg'].isin(['MCX', 'NCO'])) & (df_map['symbol'].str.contains('FUT'))
                    else: mask = (df_map['name'] == index) & (df_map['exch_seg'] == 'NFO') & (df_map['instrumenttype'] == 'FUTSTK') 
                    
                    futs = df_map[mask]
                    futs = futs[futs['expiry'] >= today]
                    if not futs.empty:
                        best_fut = futs[futs['expiry'] == futs['expiry'].min()].iloc[0]
                        spot = self.get_live_price(best_fut['exch_seg'], best_fut['symbol'], best_fut['token'])
                        df_candles = self.get_historical_data(best_fut['exch_seg'], best_fut['token'], interval=timeframe)
                        ls = best_fut.get('lotsize', '1')
                        base_lot_size = int(ls) if str(ls).isdigit() else DEFAULT_LOTS.get(index, 10)
                else:
                    spot = self.get_live_price("NFO", index, "12345")
                    df_candles = self.get_historical_data("MOCK", "12345", interval=timeframe)
                    base_lot_size = DEFAULT_LOTS.get(index, 250)
                
                if spot and df_candles is not None and not df_candles.empty:
                    self.state["spot"] = spot
                    trend, signal, vwap, ema = self.analyzer.calculate_scalp_signals(df_candles, vol_multiplier=vol_mult)
                    
                    if s.get('hero_zero', False) and signal in ["BUY_CE", "BUY_PE"]:
                        hz_start_time = dt.time(19, 0) if is_commodity else dt.time(13, 30)
                        if current_time < hz_start_time and not self.is_mock:
                            signal = "WAIT"
                            trend = f"H/Z Locked until {hz_start_time.strftime('%I:%M %p')}"
                            
                    self.state.update({"current_trend": trend, "current_signal": signal, "vwap": vwap, "ema": ema})
                    
                    cur_vol = int(df_candles['volume'].iloc[-1])
                    avg_vol = int(df_candles['volume'].rolling(20).mean().shift(1).iloc[-1]) if len(df_candles)>20 else 0
                    self.state["vol_ratio"] = f"{cur_vol}/{avg_vol}"

                    if self.state["active_trade"] is None and signal in ["BUY_CE", "BUY_PE"] and not self.state["order_in_flight"]:
                        if current_time < cutoff_time:
                            self.state["order_in_flight"] = True 
                            try:
                                qty = s['lots'] * base_lot_size
                                max_prem = s['max_capital'] / qty if qty > 0 else 0
                                strike_sym, strike_token, strike_exch = self.get_strike(index, spot, signal, max_prem)
                                
                                if strike_sym:
                                    opt_ltp = self.get_live_price(strike_exch, strike_sym, strike_token)
                                    if opt_ltp:
                                        if not paper and not self.is_mock:
                                            order_id = self.place_real_order(strike_sym, strike_token, qty, "BUY", strike_exch)
                                            if order_id: 
                                                self.log(f"⚡ REAL ENTRY: {strike_sym} | Qty: {qty} | ID: {order_id}")
                                                self.state["active_trade"] = {"symbol": strike_sym, "token": strike_token, "exch": strike_exch, "type": "CE" if "CE" in strike_sym else "PE", "entry": opt_ltp, "qty": qty, "sl": opt_ltp - s['sl_pts'], "tgt": opt_ltp + s['tgt_pts']}
                                            else: self.log("❌ REAL ENTRY FAILED.")
                                        else:
                                            self.log(f"📝 PAPER ENTRY: {strike_sym} @ {opt_ltp} | Qty: {qty}")
                                            self.state["active_trade"] = {"symbol": strike_sym, "token": strike_token, "exch": strike_exch, "type": "CE" if "CE" in strike_sym else "PE", "entry": opt_ltp, "qty": qty, "sl": opt_ltp - s['sl_pts'], "tgt": opt_ltp + s['tgt_pts']}
                            finally:
                                self.state["order_in_flight"] = False 

                    elif self.state["active_trade"]:
                        trade = self.state["active_trade"]
                        ltp = self.get_live_price(trade['exch'], trade['symbol'], trade['token'])
                        if ltp:
                            pnl = (ltp - trade['entry']) * trade['qty'] if trade['type'] == "CE" else (trade['entry'] - ltp) * trade['qty']
                            self.state["active_trade"]["current_ltp"] = ltp
                            self.state["active_trade"]["floating_pnl"] = pnl
                            
                            if current_time >= cutoff_time or ltp >= trade['tgt'] or ltp <= trade['sl']:
                                if not paper and not self.is_mock: self.place_real_order(trade['symbol'], trade['token'], trade['qty'], "SELL", trade['exch'])
                                self.log(f"✅ AUTO EXIT {trade['symbol']} @ {ltp} | PnL: ₹{round(pnl, 2)}")
                                
                                save_trade({
                                    "Date": dt.date.today().strftime('%Y-%m-%d'), "Time": dt.datetime.now().strftime('%H:%M:%S'), 
                                    "Symbol": trade['symbol'], "Type": trade['type'], "Qty": trade['qty'], 
                                    "Entry Price": trade['entry'], "Exit Price": ltp, "PnL (₹)": round(pnl, 2)
                                })
                                self.state["active_trade"] = None
            except Exception as e:
                self.log(f"Thread Error: {str(e)}")
            time.sleep(3)
# ==========================================
# 4. STREAMLIT UI 
# ==========================================
st.set_page_config(page_title="Pro Scalper Bot", page_icon="⚡", layout="wide")

# --- Initialize Session State Keys ---
if 'bot' not in st.session_state:
    st.session_state['bot'] = None

# Your CSS/Branding cleanup code here
st.markdown("""
    <style>
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {display: none;}
    .stAppDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Now this check will work perfectly
if st.session_state.bot is None:
    # Login / Connection logic
    ...

with st.sidebar:
    st.header("🔐 Connection Setup")
    auth_mode = st.radio("Operating Mode", ["📝 Paper Trading (Offline Mock Data)", "⚡ Real Trading (Live API Data)"])
    
    if not st.session_state.bot:
        if auth_mode == "⚡ Real Trading (Live API Data)":
            creds = load_creds()
            st.info("API Key is never saved locally.")
            API_KEY = st.text_input("API Key", type="password")
            CLIENT_ID = st.text_input("Client ID", value=creds.get("client_id", ""))
            PIN = st.text_input("PIN", value=creds.get("pwd", ""), type="password")
            TOTP = st.text_input("TOTP secret", value=creds.get("totp_secret", ""), type="password")
            SAVE_CREDS = st.checkbox("Remember ID & PIN", value=True)
            
            if st.button("Connect to Live Exchange", type="primary"):
                temp_bot = SniperBot(API_KEY, CLIENT_ID, PIN, TOTP, is_mock=False)
                with st.spinner("Authenticating..."):
                    if temp_bot.login():
                        if SAVE_CREDS: save_creds(CLIENT_ID, PIN, TOTP)
                        st.session_state.bot = temp_bot
                        temp_bot.fetch_master()
                        temp_bot.log("Live System Authenticated")
                        st.rerun()
                    else: st.error("Login Failed. Check credentials.")
        else:
            st.info("No credentials needed. The bot will generate synthetic market ticks.")
            if st.button("Start Offline Demo Session", type="primary"):
                temp_bot = SniperBot(is_mock=True)
                temp_bot.login()
                st.session_state.bot = temp_bot
                st.rerun()
    else:
        st.success(f"Connected: {'Offline Mock Mode' if st.session_state.bot.is_mock else st.session_state.bot.client_id}")
        if st.button("Logout & Clear"):
            st.session_state.bot.state["is_running"] = False
            st.session_state.clear()
            st.rerun()

    st.divider()
    st.header("⚙️ Scalping Settings")
    WATCHLIST_OPTIONS = ["NIFTY", "BANKNIFTY", "SENSEX", "CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "CUSTOM_STOCK..."]
    idx_sel = st.selectbox("Watchlist", WATCHLIST_OPTIONS)
    if idx_sel == "CUSTOM_STOCK...":
        INDEX = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE, SBIN)", value="RELIANCE").upper()
    else:
        INDEX = idx_sel
        
    TIMEFRAME = st.selectbox("Timeframe", ["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE"], index=2)
    VOL_MULT = st.slider("Volume Sensitivity", 1.1, 3.0, 1.5, 0.1)
    LOTS = st.number_input("Lots", 1, 100, 1)
    MAX_CAPITAL = st.number_input("Max Capital (₹)", 1000, 500000, 10000, step=1000)
    SL_PTS = st.number_input("Stop Loss (Points)", 5, 200, 20)
    TGT_PTS = st.number_input("Target (Points)", 10, 500, 40)
    
    force_paper = True if (st.session_state.bot and st.session_state.bot.is_mock) or auth_mode == "📝 Paper Trading (Offline Mock Data)" else False
    PAPER = st.toggle("📝 Paper Trade Execution", True, disabled=force_paper)
    
    st.divider()
    st.header("🚀 Hero/Zero (Gamma Blast) Mode")
    HERO_ZERO = st.toggle("Enable Hero/Zero Mode", False, help="Strictly trades on Expiry Day after 1:30 PM (Equities) or 7:00 PM (Commodities).")
    HZ_PREMIUM = st.number_input("Max H/Z Premium (₹)", 1, 100, 15, help="Only buys cheap options under this price for the Gamma Blast.")

    st.divider()
    if st.button("🚨 KILL SWITCH (CLOSE ALL)", type="primary", use_container_width=True):
        if st.session_state.bot:
            st.session_state.bot.emergency_kill()
            st.success("Positions force closed.")
            st.rerun()

if not st.session_state.bot:
    st.title("Welcome to Pro Scalper Bot")
    st.warning("Please configure your connection in the sidebar to begin.")
    st.stop()

bot = st.session_state.bot
bot.settings = {
    "index": INDEX, "timeframe": TIMEFRAME, "vol_mult": VOL_MULT, "lots": LOTS, 
    "max_capital": MAX_CAPITAL, "sl_pts": SL_PTS, "tgt_pts": TGT_PTS, "paper_mode": PAPER or force_paper,
    "hero_zero": HERO_ZERO, "hz_premium": HZ_PREMIUM
}

tab1, tab2, tab3 = st.tabs(["⚡ Live Dashboard", "🔎 OI Scanner", "📜 PnL Reports"])

with tab1:
    c1, c2 = st.columns(2)
    if c1.button("🟢 START BACKGROUND BOT"):
        if not bot.state["is_running"]:
            bot.state["is_running"] = True
            threading.Thread(target=bot.trading_loop, daemon=True).start()
    if c2.button("🔴 STOP BOT"): bot.state["is_running"] = False

    if bot.state["is_running"]:
        st.success(f"{'🚀 Bot running in background.' if not bot.is_mock else '🧪 Bot running offline.'}")
        
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{INDEX} Spot", round(bot.state["spot"], 2))
        m2.metric("Intraday VWAP", round(bot.state["vwap"], 2))
        m3.metric("9-EMA", round(bot.state["ema"], 2))
        m4.metric("Vol/Avg Ratio", bot.state["vol_ratio"])
        
        c1, c2 = st.columns(2)
        c1.metric("Market Sentiment", bot.state["current_trend"])
        c2.metric("Algo Action", bot.state["current_signal"])

        if bot.state["active_trade"]:
            t = bot.state["active_trade"]
            pnl = t.get('floating_pnl', 0.0)
            ltp = t.get('current_ltp', t['entry'])
            indicator = "🟢" if pnl >= 0 else "🔴"
            
            st.info(f"📈 Open: **{t['symbol']}** | Entry: **{t['entry']:.2f}** | LTP: **{ltp:.2f}** | PnL: {indicator} **₹{round(pnl, 2)}**")
            st.caption(f"🛑 **Stop Loss:** {t['sl']:.2f} &nbsp; | &nbsp; 🎯 **Take Profit:** {t['tgt']:.2f}")
            
            if st.button("Close Manually"):
                if not bot.settings['paper_mode'] and not bot.is_mock: bot.place_real_order(t['symbol'], t['token'], t['qty'], "SELL", t['exch'])
                
                save_trade({
                    "Date": dt.date.today().strftime('%Y-%m-%d'), "Time": dt.datetime.now().strftime('%H:%M:%S'), 
                    "Symbol": t['symbol'], "Type": t['type'], "Qty": t['qty'], 
                    "Entry Price": t['entry'], "Exit Price": ltp, "PnL (₹)": round(pnl, 2)
                })
                bot.log(f"👋 MANUALLY CLOSED | Final PnL: ₹{round(pnl, 2)}")
                bot.state["active_trade"] = None
                st.rerun()
        time.sleep(2)
        st.rerun()
    else:
        st.warning("Bot is currently stopped.")

with tab2:
    st.subheader("🔥 F&O OI Spurt Scanner")
    st.write("Scan NSE for F&O stocks with rising Open Interest to catch momentum.")

    now = dt.datetime.now()
    is_weekend = now.weekday() >= 5
    is_outside_hours = now.time() < dt.time(9, 15) or now.time() > dt.time(15, 30)

    c1, c2 = st.columns([1, 3])
    with c1:
        if 'auto_scan_oi' not in st.session_state: st.session_state.auto_scan_oi = False
        st.session_state.auto_scan_oi = st.toggle("⏱️ Auto-Scan (3 Min)", st.session_state.auto_scan_oi)
    with c2:
        manual_scan = st.button("🔍 Run Live NSE Scan Now")

    if 'last_oi_scan' not in st.session_state: st.session_state.last_oi_scan = 0
    if 'oi_data_cache' not in st.session_state: st.session_state.oi_data_cache = None

    if is_weekend or is_outside_hours:
        st.warning("🛑 **Market is currently CLOSED.**")
    else:
        current_time_sec = time.time()
        time_since_last_scan = current_time_sec - st.session_state.last_oi_scan
        
        if manual_scan or (st.session_state.auto_scan_oi and time_since_last_scan > 180):
            try:
                from nsepython import nsefetch
                with st.spinner("Fetching Live OI Data from NSE..."):
                    payload = nsefetch('https://www.nseindia.com/api/live-analysis-oi-spurts')
                    if not payload or 'data' not in payload or len(payload['data']) == 0:
                        st.session_state.oi_data_cache = "HOLIDAY"
                    else:
                        df_oi = pd.DataFrame(payload.get('data', []))
                        df_oi['pChange'] = pd.to_numeric(df_oi['pChange'], errors='coerce')
                        df_oi['per_chnge_oi'] = pd.to_numeric(df_oi['per_chnge_oi'], errors='coerce')
                        st.session_state.oi_data_cache = df_oi
                    st.session_state.last_oi_scan = current_time_sec
                    time_since_last_scan = 0
            except Exception as e:
                st.error(f"Scanner Stopped. (System Error: {e})")

        if st.session_state.oi_data_cache is not None:
            if isinstance(st.session_state.oi_data_cache, str):
                st.info("⏸️ **Scanner Stopped:** NSE returned no data. Likely a Market Holiday.")
            elif isinstance(st.session_state.oi_data_cache, pd.DataFrame):
                df_raw = st.session_state.oi_data_cache
                scan_col1, scan_col2 = st.columns(2)
                
                with scan_col1:
                    st.markdown("### 🟢 Long Buildup (Call / CE)")
                    long_buildup = df_raw[(df_raw['pChange'] > 0.5) & (df_raw['per_chnge_oi'] > 2)].head(15)
                    if not long_buildup.empty:
                        st.dataframe(long_buildup[['symbol', 'latest_price', 'pChange', 'per_chnge_oi']].rename(columns={"symbol": "Symbol", "latest_price": "LTP", "pChange": "Price % Chg", "per_chnge_oi": "OI % Chg"}), use_container_width=True, hide_index=True)
                    else: st.warning("No strong Long Buildups.")

                with scan_col2:
                    st.markdown("### 🔴 Short Buildup (Put / PE)")
                    short_buildup = df_raw[(df_raw['pChange'] < -0.5) & (df_raw['per_chnge_oi'] > 2)].head(15)
                    if not short_buildup.empty:
                        st.dataframe(short_buildup[['symbol', 'latest_price', 'pChange', 'per_chnge_oi']].rename(columns={"symbol": "Symbol", "latest_price": "LTP", "pChange": "Price % Chg", "per_chnge_oi": "OI % Chg"}), use_container_width=True, hide_index=True)
                    else: st.warning("No strong Short Buildups.")

with tab3:
    st.subheader("📊 Export PnL Reports")
    
    if os.path.exists(TRADE_FILE):
        try:
            df_all = pd.read_csv(TRADE_FILE)
            df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
            
            today_date = pd.Timestamp(dt.date.today())
            last_week_date = today_date - dt.timedelta(days=7)
            
            df_today = df_all[df_all['Date'] == today_date]
            df_week = df_all[df_all['Date'] >= last_week_date]
            
            c1, c2 = st.columns(2)
            
            if not df_today.empty:
                today_pnl = df_today['PnL (₹)'].sum()
                c1.metric("Today's Net PnL", f"₹{round(today_pnl, 2)}")
                c1.download_button("📥 Download Today's Trades", data=df_today.to_csv(index=False).encode('utf-8'), file_name=f"Trades_{dt.date.today()}.csv", mime="text/csv")
            else:
                c1.info("No trades executed today.")
                
            if not df_week.empty:
                week_pnl = df_week['PnL (₹)'].sum()
                c2.metric("Last 7 Days Net PnL", f"₹{round(week_pnl, 2)}")
                c2.download_button("📥 Download Last 7 Days Trades", data=df_week.to_csv(index=False).encode('utf-8'), file_name=f"Trades_Week_{dt.date.today()}.csv", mime="text/csv")

            st.divider()
            st.write("**Today's Execution Ledger**")
            st.dataframe(df_today)
        except Exception as e:
            st.error(f"Error reading trade history: {e}")
    else:
        st.info("No trade history available yet.")
        
    st.divider()
    st.subheader("System Logs")
    for l in bot.state["logs"]: st.text(l)


