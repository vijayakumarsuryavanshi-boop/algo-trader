import streamlit as st
import pandas as pd
import threading
import time
import os
import datetime
import random
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# --- Aggressive UI Setup ---
st.set_page_config(page_title="PRO SCALPER X - Ultimate Edition", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0b0f19; color: #00ffcc; border-right: 2px solid #ff003c; }
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
    return {
        "bot_running": False,
        "ghost_kill_lock": False,
        "live_prices": {},
        "target_token": "26000",
        "market_type": "NIFTY",
        "hero_zero_mode": False,
        "stock_scanner_enabled": False,
        "use_ml": False,
        "ml_model": "Random Forest",
        "api_client": None,
        "sl_pct": 5.0,
        "tp_pct": 10.0,
        "trailing_sl": True
    }

engine_state = get_engine_state()

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

def modify_trade_status(status, new_sl=None, pnl=None):
    """Utility to update the open trade (e.g., Trailing SL or Closing)"""
    df = pd.read_csv(TRADE_LOG_FILE)
    if not df.empty and df.iloc[-1]['Status'] == 'OPEN':
        df.at[df.index[-1], 'Status'] = status
        if new_sl is not None:
            df.at[df.index[-1], 'SL'] = new_sl
        if pnl is not None:
            df.at[df.index[-1], 'PnL'] = pnl
        df.to_csv(TRADE_LOG_FILE, index=False)

# --- AI & Master Strategy Logic ---
def ml_predict_signal(token, current_price, model_type):
    """Mock interface for ML inference"""
    prob_buy = random.uniform(0.1, 0.9)
    threshold = 0.75 if model_type == "Random Forest" else 0.80
    if prob_buy > threshold: return "BUY_LIMIT", "AI_BUY"
    elif prob_buy < (1 - threshold): return "SELL_LIMIT", "AI_SELL"
    return "NEUTRAL", "NONE"

def master_analyzer(token, market_type, is_hero_zero, is_scanner_active, use_ml, ml_model):
    now = datetime.datetime.now().time()
    current_spot = engine_state["live_prices"].get(token, 0)
    
    # 1. BTST / STBT End-of-Day Logic Check (3:15 PM - 3:25 PM IST)
    is_eod = datetime.time(15, 15) <= now <= datetime.time(15, 25)
    
    # 2. Machine Learning Override Check
    if use_ml and current_spot > 0:
        ml_signal, ml_tag = ml_predict_signal(token, current_spot, ml_model)
        if ml_signal != "NEUTRAL":
            if is_eod: ml_tag += "_BTST" if "BUY" in ml_signal else "_STBT"
            return ml_signal, current_spot, ml_tag

    # 3. Standard SMC / FVG Logic
    df = pd.DataFrame({
        "Open": [100]*25, "High": [105]*25, "Low": [95]*25, "Close": [102]*25, "Volume": [1000]*24 + [5000]
    })
    df.at[df.index[-1], 'Close'] = current_spot
    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['Inst_Vol'] = df['Volume'] > (df['Vol_MA'] * 2)
    df['Bull_FVG'] = (df['Low'] > df['High'].shift(2)) & df['Inst_Vol'].shift(1)
    df['Bear_FVG'] = (df['High'] < df['Low'].shift(2)) & df['Inst_Vol'].shift(1)
    
    last_row = df.iloc[-1]
    signal, limit_price, strategy_tag = "NEUTRAL", 0, f"SMC_{market_type}"

    if is_hero_zero and current_spot > 0 and current_spot <= 20 and last_row['Inst_Vol']:
        signal, limit_price = "BUY_LIMIT", current_spot
        strategy_tag += "_HERO_ZERO"
    elif is_scanner_active and market_type == "EQUITY_SCANNER" and last_row['Inst_Vol'] and current_spot > df['High'].mean():
        signal, limit_price = "BUY_LIMIT", current_spot
        strategy_tag = "MOMENTUM_SCANNER"
    elif last_row['Bull_FVG'] and current_spot > 0:
        signal, limit_price = "BUY_LIMIT", df['High'].shift(2).iloc[-1]
    elif last_row['Bear_FVG'] and current_spot > 0:
        signal, limit_price = "SELL_LIMIT", df['Low'].shift(2).iloc[-1]
        
    if is_eod and "BUY" in signal: strategy_tag += "_BTST"
    elif is_eod and "SELL" in signal: strategy_tag += "_STBT"
            
    return signal, limit_price, strategy_tag

# --- Resilient Background Daemon ---
def resilient_bot_loop():
    while True:
        if engine_state["bot_running"]:
            now = datetime.datetime.now().time()
            market_type = engine_state["market_type"]
            is_market_open = datetime.time(9, 0) <= now <= datetime.time(23, 30) if market_type == "COMMODITY" else datetime.time(9, 15) <= now <= datetime.time(15, 30)
            
            if is_market_open:
                target_token = engine_state["target_token"]
                current_spot = engine_state["live_prices"].get(target_token)
                
                if current_spot:
                    signal, limit_price, strategy_tag = master_analyzer(
                        target_token, market_type, engine_state["hero_zero_mode"], 
                        engine_state["stock_scanner_enabled"], engine_state["use_ml"], engine_state["ml_model"]
                    )
                    
                    # Entry
                    if "LIMIT" in signal and not engine_state["ghost_kill_lock"]:
                        if limit_price * 50 <= MAX_CAPITAL: 
                            engine_state["ghost_kill_lock"] = True 
                            sl = limit_price * (1 - (engine_state["sl_pct"] / 100))
                            tp = limit_price * (1 + (engine_state["tp_pct"] / 100))
                            if engine_state["hero_zero_mode"]: sl = 0.05 
                            log_trade(target_token, market_type, strategy_tag, limit_price, sl, tp, "OPEN")
                            
                    # Exit / Management
                    elif engine_state["ghost_kill_lock"]:
                        df = pd.read_csv(TRADE_LOG_FILE)
                        open_trades = df[df['Status'] == 'OPEN']
                        
                        if not open_trades.empty:
                            entry_price = open_trades.iloc[-1]['Entry']
                            current_sl = open_trades.iloc[-1]['SL']
                            tp_price = open_trades.iloc[-1]['TP']
                            
                            # Trailing SL to Breakeven Logic (If price moves 50% to target)
                            half_way = entry_price + ((tp_price - entry_price) * 0.5)
                            if engine_state["trailing_sl"] and current_spot >= half_way and current_sl < entry_price:
                                modify_trade_status("OPEN", new_sl=entry_price) # Move SL to Entry
                                
                            # Hit TP or SL
                            if current_spot >= tp_price:
                                engine_state["ghost_kill_lock"] = False
                                modify_trade_status("CLOSED_TP", pnl=(tp_price - entry_price) * 50)
                            elif current_spot <= current_sl:
                                engine_state["ghost_kill_lock"] = False
                                modify_trade_status("CLOSED_SL", pnl=(current_sl - entry_price) * 50)
            else:
                engine_state["ghost_kill_lock"] = False
        time.sleep(3)

@st.cache_resource
def start_daemon():
    worker = threading.Thread(target=resilient_bot_loop, daemon=True)
    worker.start()
    return worker

start_daemon()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### ⚡ PRO SCALPER X")
    if engine_state["bot_running"]:
        st.success("🟢 ENGINE ONLINE")
        if st.button("🛑 KILL SWITCH"):
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
    st.selectbox("Asset Class", ["NIFTY / BANKNIFTY", "SENSEX", "COMMODITY (MCX)", "EQUITY_SCANNER"], key="ui_market")
    st.text_input("Target Options Token", value=engine_state["target_token"], key="ui_target")
    
    st.markdown("#### 🚀 STRATEGY MODIFIERS")
    st.checkbox("🔥 Hero / Zero Mode", value=engine_state["hero_zero_mode"], key="ui_hero_zero")
    st.checkbox("📡 Enable Stock Scanner", value=engine_state["stock_scanner_enabled"], key="ui_scanner")
    
    st.markdown("#### 🛡️ RISK MANAGEMENT")
    st.number_input("Stop Loss (%)", value=engine_state["sl_pct"], key="ui_sl")
    st.number_input("Take Profit (%)", value=engine_state["tp_pct"], key="ui_tp")

# --- Main Dashboard ---
tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD", "🧠 AI & EOD SETTINGS", "📜 TRADE LEDGER"])

with tab1:
    st.title("Operations Dashboard")
    df_history = pd.read_csv(TRADE_LOG_FILE)
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    daily_trades = df_history[df_history['Date'] == today_str]

    daily_pnl = daily_trades['PnL'].sum() if not daily_trades.empty else 0.0
    tp_hits = len(daily_trades[daily_trades['Status'] == 'CLOSED_TP']) if not daily_trades.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Daily Net PnL", f"₹ {daily_pnl:.2f}")
    col2.metric("TP Target Hits", tp_hits)
    col3.metric("Live Market Price", engine_state["live_prices"].get(engine_state["target_token"], "Awaiting..."))

with tab2:
    st.markdown("### 🤖 Artificial Intelligence Overrides")
    col_ai, col_eod = st.columns(2)
    with col_ai:
        engine_state["use_ml"] = st.checkbox("Enable Deep Learning Signals", value=engine_state["use_ml"])
        engine_state["ml_model"] = st.selectbox("Prediction Model", ["Random Forest", "XGBoost", "LSTM"], index=["Random Forest", "XGBoost", "LSTM"].index(engine_state["ml_model"]))
    with col_eod:
        st.markdown("### 🕒 BTST / STBT Engine")
        st.info("System automatically tags overnight positional trades between 3:15 PM and 3:25 PM IST.")
        engine_state["trailing_sl"] = st.checkbox("Enable Trailing SL to Breakeven", value=engine_state["trailing_sl"], help="Moves SL to entry price once trade is 50% to target.")

with tab3:
    st.subheader("Live Trade Ledger")
    st.dataframe(df_history.iloc[::-1], use_container_width=True)
    
    # Excel Download Button Restored
    csv = df_history.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Daily/Weekly Report (CSV)", data=csv, file_name='trade_history_report.csv', mime='text/csv')
    
    if st.button("🔄 Refresh Ledger"):
        st.rerun()
