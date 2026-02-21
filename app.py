import streamlit as st
import pandas as pd
import threading
import time
import os
import datetime
import pandas_ta as ta
import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# --- Aggressive UI Setup ---
st.set_page_config(page_title="PRO SCALPER X - SMC Edition", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0d1117; color: #00ffcc; border-right: 2px solid #ff003c; }
    [data-testid="stSidebar"] h1, h2, h3 { color: #ff003c !important; font-family: 'Courier New', monospace; text-transform: uppercase; }
    div.stButton > button:first-child { background: linear-gradient(90deg, #ff003c 0%, #cc0033 100%); color: white; font-weight: 900; border-radius: 8px; width: 100%; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #00ffcc; }
</style>
""", unsafe_allow_html=True)

TRADE_LOG_FILE = "trade_history.csv"
MAX_CAPITAL = 10000

# --- Persistent Engine State (Survives Browser Disconnects) ---
@st.cache_resource
def get_engine_state():
    """Global state that persists even if the Streamlit browser tab is closed."""
    return {
        "bot_running": False,
        "ghost_kill_lock": False,
        "live_prices": {},
        "target_token": "26000",
        "api_client": None,
        "sl_pct": 5.0,
        "tp_pct": 10.0
    }

engine_state = get_engine_state()

# Load persistent history
if os.path.exists(TRADE_LOG_FILE):
    trade_history = pd.read_csv(TRADE_LOG_FILE)
else:
    trade_history = pd.DataFrame(columns=["Date", "Time", "Symbol", "Type", "Strategy", "Entry", "SL", "TP", "Status", "PnL"])
    trade_history.to_csv(TRADE_LOG_FILE, index=False)

def log_trade(symbol, trade_type, strategy, entry, sl, tp, status, pnl=0.0):
    now = datetime.datetime.now()
    new_trade = pd.DataFrame([{
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Symbol": symbol,
        "Type": trade_type,
        "Strategy": strategy,
        "Entry": entry,
        "SL": sl,
        "TP": tp,
        "Status": status,
        "PnL": pnl
    }])
    # Append directly to CSV to ensure data isn't lost if UI drops
    new_trade.to_csv(TRADE_LOG_FILE, mode='a', header=not os.path.exists(TRADE_LOG_FILE), index=False)

# --- Smart Money Concepts (SMC) & FVG Analyzer ---
def smc_fvg_analyzer(token, api_client):
    """
    Identifies Institutional Volume Displacement and 3-Candle Fair Value Gaps.
    Returns the Limit Order Type and the Exact FVG Boundary Price.
    """
    try:
        # Mocking historic Angel One OHLCV fetch for the example
        # In production, replace with: api_client.getCandleData(historic_param)
        current_spot = engine_state["live_prices"].get(token, 0)
        
        # Simulated Data for SMC Calculation
        df = pd.DataFrame({
            "Open": [100]*25, "High": [105]*25, "Low": [95]*25, "Close": [102]*25, 
            "Volume": [1000]*24 + [5000] # Simulated institutional volume spike
        })
        df.at[df.index[-1], 'Close'] = current_spot
        
        # 1. Institutional Volume Check (Volume > 2x 20-period Moving Average)
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        df['Inst_Vol'] = df['Volume'] > (df['Vol_MA'] * 2)
        
        # 2. Fair Value Gap (3-Candle Imbalance)
        df['Bull_FVG'] = (df['Low'] > df['High'].shift(2)) & df['Inst_Vol'].shift(1)
        df['Bear_FVG'] = (df['High'] < df['Low'].shift(2)) & df['Inst_Vol'].shift(1)
        
        last_row = df.iloc[-1]
        
        if last_row['Bull_FVG'] and current_spot > 0:
            # Bullish: Place Limit Order at the top boundary of the gap
            fvg_top = df['High'].shift(2).iloc[-1]
            return "BUY_LIMIT", fvg_top, "SMC_Bull_FVG"
            
        elif last_row['Bear_FVG'] and current_spot > 0:
            # Bearish: Place Limit Order at the bottom boundary of the gap
            fvg_bottom = df['Low'].shift(2).iloc[-1]
            return "SELL_LIMIT", fvg_bottom, "SMC_Bear_FVG"
            
    except Exception as e:
        pass
        
    return "NEUTRAL", 0, "NONE"

# --- Resilient Background Daemon ---
def resilient_bot_loop():
    """Runs indefinitely in the background, fully decoupled from the browser."""
    while True:
        if engine_state["bot_running"]:
            now = datetime.datetime.now().time()
            is_market_open = datetime.time(9, 15) <= now <= datetime.time(15, 30)
            
            if is_market_open:
                target_token = engine_state["target_token"]
                current_spot = engine_state["live_prices"].get(target_token)
                
                if current_spot:
                    signal, limit_price, strategy_tag = smc_fvg_analyzer(target_token, engine_state["api_client"])
                    
                    # Entry Execution
                    if "LIMIT" in signal and not engine_state["ghost_kill_lock"]:
                        # Ensure capital safety
                        if limit_price * 50 <= MAX_CAPITAL: 
                            engine_state["ghost_kill_lock"] = True 
                            
                            sl = limit_price * (1 - (engine_state["sl_pct"] / 100))
                            tp = limit_price * (1 + (engine_state["tp_pct"] / 100))
                            
                            log_trade(target_token, signal, strategy_tag, limit_price, sl, tp, "OPEN")
                            
                    # Exit Execution (Simplified TP/SL check)
                    elif engine_state["ghost_kill_lock"]:
                        df = pd.read_csv(TRADE_LOG_FILE)
                        open_trades = df[df['Status'] == 'OPEN']
                        
                        if not open_trades.empty:
                            entry_price = open_trades.iloc[-1]['Entry']
                            tp_price = open_trades.iloc[-1]['TP']
                            
                            if current_spot >= tp_price:
                                engine_state["ghost_kill_lock"] = False
                                pnl = (tp_price - entry_price) * 50 
                                log_trade(target_token, "SELL_LIMIT", strategy_tag, current_spot, 0, 0, "CLOSED_TP", pnl)
            else:
                # Market is closed, naturally pause executions
                engine_state["ghost_kill_lock"] = False
                
        time.sleep(3)

# Start the daemon only once per server lifetime
@st.cache_resource
def start_daemon():
    worker = threading.Thread(target=resilient_bot_loop, daemon=True)
    worker.start()
    return worker

start_daemon()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### ⚡ PRO SCALPER X")
    st.caption("SMC LIMIT-ORDER ENGINE")
    
    # Engine status directly reads from the persistent global state
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
            engine_state["target_token"] = st.session_state.get('ui_target', '26000')
            engine_state["sl_pct"] = st.session_state.get('ui_sl', 5.0)
            engine_state["tp_pct"] = st.session_state.get('ui_tp', 10.0)
            st.rerun()
            
    st.divider()
    target = st.text_input("Target Options Token", value=engine_state["target_token"], key="ui_target")
    st.number_input("Stop Loss (%)", value=engine_state["sl_pct"], key="ui_sl")
    st.number_input("Take Profit (%)", value=engine_state["tp_pct"], key="ui_tp")

# --- Main Dashboard ---
st.title("📊 Daily Operations Dashboard")

# Read the freshest data directly from the CSV
df_history = pd.read_csv(TRADE_LOG_FILE)

# Calculate Daily Metrics
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
daily_trades = df_history[df_history['Date'] == today_str]

if not daily_trades.empty:
    daily_pnl = daily_trades['PnL'].sum()
    tp_hits = len(daily_trades[daily_trades['Status'] == 'CLOSED_TP'])
    
    # Extract winning symbols
    winning_trades = daily_trades[daily_trades['PnL'] > 0]
    win_symbols = ", ".join(winning_trades['Symbol'].astype(str).unique()) if not winning_trades.empty else "None yet"
else:
    daily_pnl = 0.0
    tp_hits = 0
    win_symbols = "Awaiting executions..."

col1, col2, col3 = st.columns(3)
col1.metric("Daily Net PnL", f"₹ {daily_pnl:.2f}")
col2.metric("TP Target Hits", tp_hits)
col3.metric("Winning Symbols", win_symbols)

st.divider()
st.subheader("📜 Live Trade Ledger")
st.dataframe(df_history.iloc[::-1], use_container_width=True)
if st.button("🔄 Refresh Ledger"):
    st.rerun()SocketV2(st.session_state.jwt_token, api_key, client_code, st.session_state.feed_token)
                    def on_open(wsapp):
                        sws.subscribe("stream_1", 1, [{"exchangeType": 2, "tokens": [target_token]}])
                    sws.on_open = on_open
                    sws.on_data = on_data
                    ws_thread = threading.Thread(target=sws.connect, daemon=True)
                    ws_thread.start()
                    
                    bot_thread = threading.Thread(target=bot_loop, args=(st.session_state.api_client, target_token, sl_pct, tp_pct, use_ml, ml_model))
                    bot_thread.do_run = True
                    bot_thread.add_script_run_ctx = True 
                    bot_thread.start()
                    st.rerun()

    # ---------------- TAB 2: ML & STRATEGY (BTST/STBT) ----------------
    with tab2:
        st.markdown("### 🤖 Strategy Parameters")
        
        col_ml, col_risk = st.columns(2)
        
        with col_ml:
            st.subheader("Machine Learning Interface")
            use_ml = st.checkbox("Enable AI Buy/Sell Override", key="use_ml")
            ml_model = st.selectbox("Prediction Model", ["Random Forest", "XGBoost", "LSTM (Deep Learning)"], key="ml_model")
            st.caption("Models require pre-trained .pkl or .h5 files loaded into the root directory.")
            
            st.divider()
            st.subheader("EOD Strategy (BTST/STBT)")
            st.info("🕒 System automatically monitors the 3:15 PM - 3:25 PM IST window to execute positional overnight trades based on End-of-Day technical momentum.")
            
        with col_risk:
            st.subheader("Risk Management")
            max_cap = st.number_input("Max Capital Base (₹)", value=MAX_CAPITAL, disabled=True)
            sl_pct = st.number_input("Stop Loss (%)", value=5.0, step=1.0, key="sl_pct")
            tp_pct = st.number_input("Take Profit (%)", value=10.0, step=1.0, key="tp_pct")

    # ---------------- TAB 3: TRADE LEDGER ----------------
    with tab3:
        st.markdown("### 📊 Trade History & Performance")
        
        if not st.session_state.trade_history.empty:
            total_pnl = st.session_state.trade_history['PnL'].sum()
            
            # Color code the PnL
            if total_pnl > 0:
                st.success(f"### NET PNL: ₹ {total_pnl:.2f} 🟢")
            else:
                st.error(f"### NET PNL: ₹ {total_pnl:.2f} 🔴")
                
            st.dataframe(st.session_state.trade_history.iloc[::-1], use_container_width=True)
        else:
            st.caption("No trades recorded in this session.")

else:
    st.info("Awaiting secure connection. Please initialize via the sidebar.")

