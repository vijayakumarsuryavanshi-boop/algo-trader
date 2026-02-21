import streamlit as st
import pandas as pd
import threading
import time
import os
import datetime
import pandas_ta as ta
import pyotp
import random
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# --- Aggressive UI Setup ---
st.set_page_config(page_title="PRO SCALPER X - Ultimate Edition", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0d1117; color: #00ffcc; border-right: 2px solid #ff003c; }
    [data-testid="stSidebar"] h1, h2, h3, h4 { color: #ff003c !important; font-family: 'Courier New', monospace; text-transform: uppercase; }
    div.stButton > button:first-child { background: linear-gradient(90deg, #ff003c 0%, #cc0033 100%); color: white; font-weight: 900; border-radius: 8px; width: 100%; text-transform: uppercase; }
    div.stButton > button:first-child:hover { background: linear-gradient(90deg, #cc0033 0%, #990022 100%); }
    .stTabs [data-baseweb="tab"] { font-weight: bold; color: #ffffff; }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid #ff003c; color: #ff003c !important; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #ff003c; }
</style>
""", unsafe_allow_html=True)

TRADE_LOG_FILE = "trade_history.csv"
MAX_CAPITAL = 10000

# --- Persistent Engine State ---
@st.cache_resource
def get_engine_state():
    """Global state that persists even if the Streamlit browser tab is closed."""
    return {
        "bot_running": False,
        "ghost_kill_lock": False,
        "live_prices": {},
        "target_token": "26000",
        "market_type": "NIFTY",
        "hero_zero_mode": False,
        "stock_scanner_enabled": False,
        "api_client": None,
        "sl_pct": 5.0,
        "tp_pct": 10.0
    }

engine_state = get_engine_state()

# Load persistent history
if os.path.exists(TRADE_LOG_FILE):
    trade_history = pd.read_csv(TRADE_LOG_FILE)
else:
    trade_history = pd.DataFrame(columns=["Date", "Time", "Symbol", "Market", "Strategy", "Entry", "SL", "TP", "Status", "PnL"])
    trade_history.to_csv(TRADE_LOG_FILE, index=False)

def log_trade(symbol, market, strategy, entry, sl, tp, status, pnl=0.0):
    now = datetime.datetime.now()
    new_trade = pd.DataFrame([{
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Symbol": symbol,
        "Market": market,
        "Strategy": strategy,
        "Entry": entry,
        "SL": sl,
        "TP": tp,
        "Status": status,
        "PnL": pnl
    }])
    new_trade.to_csv(TRADE_LOG_FILE, mode='a', header=not os.path.exists(TRADE_LOG_FILE), index=False)

# --- Integrated Strategy Logic (SMC + Multi-Market) ---
def master_analyzer(token, market_type, is_hero_zero, is_scanner_active):
    """
    Evaluates FVG/SMC across different market classes.
    Adjusts risk and signals based on Hero/Zero and Scanner parameters.
    """
    current_spot = engine_state["live_prices"].get(token, 0)
    
    # Mock Market Data for SMC
    df = pd.DataFrame({
        "Open": [100]*25, "High": [105]*25, "Low": [95]*25, "Close": [102]*25, 
        "Volume": [1000]*24 + [5000] # Simulated volume spike
    })
    df.at[df.index[-1], 'Close'] = current_spot
    
    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Inst_Vol'] = df['Volume'] > (df['Vol_MA'] * 2)
    df['Bull_FVG'] = (df['Low'] > df['High'].shift(2)) & df['Inst_Vol'].shift(1)
    df['Bear_FVG'] = (df['High'] < df['Low'].shift(2)) & df['Inst_Vol'].shift(1)
    
    last_row = df.iloc[-1]
    signal = "NEUTRAL"
    limit_price = 0
    strategy_tag = f"SMC_{market_type}"

    # Hero/Zero Expiry Logic Override (Hyper-aggressive limit fill)
    if is_hero_zero:
        strategy_tag += "_HERO_ZERO"
        # In Hero/Zero, wait for an extreme drop in premium (e.g., buying at ₹5 to ₹15)
        if current_spot > 0 and current_spot <= 20 and last_row['Inst_Vol']:
            return "BUY_LIMIT", current_spot, strategy_tag

    # Stock Scanner Logic Override (Looking for breakout momentum)
    if is_scanner_active and market_type == "EQUITY_SCANNER":
        strategy_tag = "MOMENTUM_SCANNER"
        if last_row['Inst_Vol'] and current_spot > df['High'].mean():
            return "BUY_LIMIT", current_spot, strategy_tag

    # Standard FVG Logic for Nifty/Sensex/Commodity
    if last_row['Bull_FVG'] and current_spot > 0:
        limit_price = df['High'].shift(2).iloc[-1]
        signal = "BUY_LIMIT"
    elif last_row['Bear_FVG'] and current_spot > 0:
        limit_price = df['Low'].shift(2).iloc[-1]
        signal = "SELL_LIMIT"
            
    return signal, limit_price, strategy_tag

# --- Resilient Background Daemon ---
def resilient_bot_loop():
    while True:
        if engine_state["bot_running"]:
            now = datetime.datetime.now().time()
            
            # Market hours adjust based on commodity vs equity/index
            market_type = engine_state["market_type"]
            if market_type == "COMMODITY":
                is_market_open = datetime.time(9, 0) <= now <= datetime.time(23, 30) # MCX hours
            else:
                is_market_open = datetime.time(9, 15) <= now <= datetime.time(15, 30) # NSE/BSE hours
            
            if is_market_open:
                target_token = engine_state["target_token"]
                current_spot = engine_state["live_prices"].get(target_token)
                
                if current_spot:
                    signal, limit_price, strategy_tag = master_analyzer(
                        target_token, 
                        market_type, 
                        engine_state["hero_zero_mode"], 
                        engine_state["stock_scanner_enabled"]
                    )
                    
                    if "LIMIT" in signal and not engine_state["ghost_kill_lock"]:
                        if limit_price * 50 <= MAX_CAPITAL: 
                            engine_state["ghost_kill_lock"] = True 
                            
                            sl = limit_price * (1 - (engine_state["sl_pct"] / 100))
                            tp = limit_price * (1 + (engine_state["tp_pct"] / 100))
                            
                            # Hero Zero often goes to 0, strict SL
                            if engine_state["hero_zero_mode"]:
                                sl = 0.05 # Basically let it ride to zero
                                
                            log_trade(target_token, market_type, strategy_tag, limit_price, sl, tp, "OPEN")
                            
                    elif engine_state["ghost_kill_lock"]:
                        df = pd.read_csv(TRADE_LOG_FILE)
                        open_trades = df[df['Status'] == 'OPEN']
                        
                        if not open_trades.empty:
                            entry_price = open_trades.iloc[-1]['Entry']
                            tp_price = open_trades.iloc[-1]['TP']
                            
                            if current_spot >= tp_price:
                                engine_state["ghost_kill_lock"] = False
                                pnl = (tp_price - entry_price) * 50 
                                log_trade(target_token, market_type, strategy_tag, current_spot, 0, 0, "CLOSED_TP", pnl)
            else:
                engine_state["ghost_kill_lock"] = False
                
        time.sleep(3)

@st.cache_resource
def start_daemon():
    worker = threading.Thread(target=resilient_bot_loop, daemon=True)
    worker.start()
    return worker

start_daemon()

# --- Sidebar Controls (Fully Restored) ---
with st.sidebar:
    st.markdown("### ⚡ PRO SCALPER X")
    st.caption("ULTIMATE MULTI-MARKET NODE")
    
    st.markdown("#### ⚙️ ENGINE STATUS")
    if engine_state["bot_running"]:
        st.success("🟢 ENGINE ONLINE (Headless)")
        if st.button("🛑 EMERGENCY KILL SWITCH"):
            engine_state["bot_running"] = False
            st.rerun()
    else:
        st.error("🔴 ENGINE OFFLINE")
        if st.button("▶️ IGNITE SCALPER"):
            engine_state["bot_running"] = True
            engine_state["ghost_kill_lock"] = False
            engine_state["target_token"] = st.session_state.ui_target
            engine_state["market_type"] = st.session_state.ui_market
            engine_state["hero_zero_mode"] = st.session_state.ui_hero_zero
            engine_state["stock_scanner_enabled"] = st.session_state.ui_scanner
            engine_state["sl_pct"] = st.session_state.ui_sl
            engine_state["tp_pct"] = st.session_state.ui_tp
            st.rerun()
            
    st.divider()
    
    st.markdown("#### 🎯 MARKET SELECTION")
    st.selectbox("Asset Class", ["NIFTY / BANKNIFTY", "SENSEX", "COMMODITY (MCX)", "EQUITY_SCANNER"], key="ui_market")
    st.text_input("Target Options Token", value=engine_state["target_token"], key="ui_target")
    
    st.markdown("#### 🚀 STRATEGY MODIFIERS")
    st.checkbox("🔥 Hero / Zero Mode (Expiry Day)", value=engine_state["hero_zero_mode"], key="ui_hero_zero", help="Modifies SMC logic for explosive expiry premium decay.")
    st.checkbox("📡 Enable High-Movement Stock Scanner", value=engine_state["stock_scanner_enabled"], key="ui_scanner", help="Scans equity space for massive institutional volume displacement.")
    
    st.markdown("#### 🛡️ RISK MANAGEMENT")
    st.number_input("Stop Loss (%)", value=engine_state["sl_pct"], key="ui_sl")
    st.number_input("Take Profit (%)", value=engine_state["tp_pct"], key="ui_tp")

# --- Main Dashboard ---
tab1, tab2, tab3 = st.tabs(["📊 DAILY PNL DASHBOARD", "⚙️ SYSTEM TERMINAL", "📜 TRADE LEDGER"])

with tab1:
    st.title("Daily Operations Dashboard")
    df_history = pd.read_csv(TRADE_LOG_FILE)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    daily_trades = df_history[df_history['Date'] == today_str]

    if not daily_trades.empty:
        daily_pnl = daily_trades['PnL'].sum()
        tp_hits = len(daily_trades[daily_trades['Status'] == 'CLOSED_TP'])
        winning_trades = daily_trades[daily_trades['PnL'] > 0]
        win_symbols = ", ".join(winning_trades['Symbol'].astype(str).unique()) if not winning_trades.empty else "None"
    else:
        daily_pnl = 0.0
        tp_hits = 0
        win_symbols = "Awaiting executions..."

    col1, col2, col3 = st.columns(3)
    col1.metric("Daily Net PnL", f"₹ {daily_pnl:.2f}")
    col2.metric("TP Target Hits", tp_hits)
    col3.metric("Winning Symbols", win_symbols)

    st.divider()
    st.subheader("Active Market Configuration")
    st.write(f"**Selected Market:** {engine_state['market_type']}")
    st.write(f"**Hero/Zero Mode:** {'ENABLED' if engine_state['hero_zero_mode'] else 'DISABLED'}")
    st.write(f"**Stock Scanner:** {'ACTIVE' if engine_state['stock_scanner_enabled'] else 'INACTIVE'}")

with tab2:
    st.subheader("System Terminal & Connectivity")
    st.info("The broker authentication module has been moved here to keep the sidebar clean for pure strategy execution.")
    api_key = st.text_input("🔑 API Key", type="password")
    client_code = st.text_input("👤 Client ID")
    pin = st.text_input("🔒 MPIN", type="password")
    totp_secret = st.text_input("⏱️ TOTP Secret", type="password")
    if st.button("CONNECT BROKER NODE"):
        st.success("✅ BROKER NODE CONNECTED (Simulated)")

with tab3:
    st.subheader("Live Trade Ledger")
    st.dataframe(df_history.iloc[::-1], use_container_width=True)
    if st.button("🔄 Refresh Ledger"):
        st.rerun()
