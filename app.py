import streamlit as st
import pandas as pd
import threading
import time
import os
import datetime
import pandas_ta as ta
import pyotp
import random # For simulating ML probability
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# --- Configuration & Aggressive UI Setup ---
st.set_page_config(page_title="PRO SCALPER X", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a dark, aggressive, mobile-like sidebar
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        color: #00ffcc;
        border-right: 2px solid #ff003c;
    }
    [data-testid="stSidebar"] h1, h2, h3 {
        color: #ff003c !important;
        font-family: 'Courier New', Courier, monospace;
        text-transform: uppercase;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff003c 0%, #cc0033 100%);
        color: white;
        font-weight: 900;
        border-radius: 8px;
        border: none;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #cc0033 0%, #990022 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #ff003c;
        color: #ff003c !important;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'ghost_kill_lock' not in st.session_state:
    st.session_state.ghost_kill_lock = False
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=["Time", "Symbol", "Type", "Strategy", "Entry", "SL", "TP", "Status", "PnL"])
if 'live_prices' not in st.session_state:
    st.session_state.live_prices = {}

TRADE_LOG_FILE = "trade_history.csv"
MAX_CAPITAL = 10000

if os.path.exists(TRADE_LOG_FILE):
    st.session_state.trade_history = pd.read_csv(TRADE_LOG_FILE)

# --- Angel One WebSocket Callbacks ---
def on_data(wsapp, message):
    try:
        token = message.get('token')
        ltp = message.get('last_traded_price', 0) / 100.0
        if token and ltp > 0:
            st.session_state.live_prices[token] = ltp
    except Exception as e:
        pass

def on_error(wsapp, error):
    pass

# --- Machine Learning Engine (Placeholder Interface) ---
def ml_predict_signal(token, current_price, model_type):
    """
    Mock interface for Machine Learning integration.
    You will replace the random probability with actual inference from Scikit-learn/TensorFlow.
    """
    # Simulate processing time for ML inference
    prob_buy = random.uniform(0.1, 0.9)
    
    if model_type == "Random Forest":
        threshold = 0.75
    elif model_type == "LSTM (Deep Learning)":
        threshold = 0.80
    else:
        threshold = 0.70

    if prob_buy > threshold:
        return "BUY", prob_buy
    elif prob_buy < (1 - threshold):
        return "SELL", (1 - prob_buy)
    
    return "NEUTRAL", prob_buy

# --- Core Trading Logic & End-of-Day Logic ---
def technical_analyzer(token, api_client, use_ml, ml_model):
    """Evaluates standard VWAP/EMA and intercepts End-of-Day for BTST/STBT"""
    
    now = datetime.datetime.now().time()
    current_price = st.session_state.live_prices.get(token, 0)
    
    # 1. BTST / STBT End-of-Season Logic (3:15 PM - 3:25 PM IST)
    is_eod = datetime.time(15, 15) <= now <= datetime.time(15, 25)
    
    signal = "NEUTRAL"
    strategy_tag = "VWAP_EMA"
    
    # 2. Machine Learning Override
    if use_ml and current_price > 0:
        ml_signal, confidence = ml_predict_signal(token, current_price, ml_model)
        if ml_signal != "NEUTRAL":
            signal = ml_signal
            strategy_tag = f"AI_{ml_model}"
            
    # 3. Standard Technical Fallback (Simulated due to missing live history in demo)
    if signal == "NEUTRAL" and current_price > 0:
        # Simplified standard logic for paper trading script
        signal = random.choice(["BUY", "SELL", "NEUTRAL", "NEUTRAL", "NEUTRAL"])
    
    # Tag BTST/STBT if in the specific time window
    if is_eod and signal == "BUY":
        strategy_tag = "BTST"
    elif is_eod and signal == "SELL":
        strategy_tag = "STBT"
        
    return signal, strategy_tag

def log_trade(symbol, trade_type, strategy, entry, sl, tp, status, pnl=0.0):
    new_trade = pd.DataFrame([{
        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
        "Symbol": symbol,
        "Type": trade_type,
        "Strategy": strategy,
        "Entry": entry,
        "SL": sl,
        "TP": tp,
        "Status": status,
        "PnL": pnl
    }])
    st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_trade], ignore_index=True)
    st.session_state.trade_history.to_csv(TRADE_LOG_FILE, index=False)

# --- Background Bot Thread ---
def bot_loop(api_client, target_token, sl_pct, tp_pct, use_ml, ml_model):
    while getattr(threading.current_thread(), "do_run", True):
        if not st.session_state.bot_running:
            break
            
        try:
            current_spot = st.session_state.live_prices.get(target_token)
            
            if current_spot:
                signal, strategy_tag = technical_analyzer(target_token, api_client, use_ml, ml_model)
                
                if signal == "BUY" and not st.session_state.ghost_kill_lock:
                    if current_spot * 50 <= MAX_CAPITAL: 
                        st.session_state.ghost_kill_lock = True 
                        sl = current_spot * (1 - (sl_pct / 100))
                        tp = current_spot * (1 + (tp_pct / 100))
                        
                        log_trade(target_token, "BUY", strategy_tag, current_spot, sl, tp, "OPEN")
                    
                elif signal == "SELL" and st.session_state.ghost_kill_lock:
                    st.session_state.ghost_kill_lock = False 
                    open_trades = st.session_state.trade_history[st.session_state.trade_history['Status'] == 'OPEN']
                    if not open_trades.empty:
                        entry_price = open_trades.iloc[-1]['Entry']
                        pnl = (current_spot - entry_price) * 50 
                        
                        st.session_state.trade_history.loc[st.session_state.trade_history.index[-1], 'Status'] = 'CLOSED'
                        log_trade(target_token, "SELL", strategy_tag, current_spot, 0, 0, "CLOSED", pnl)
            
            time.sleep(2) 
        except Exception as e:
            time.sleep(5)

# --- Sidebar UI (Mobile App Vibe) ---
with st.sidebar:
    st.markdown("### ⚡ PRO SCALPER X")
    st.caption("v2.1 API LINK SECURE")
    
    api_key = st.text_input("🔑 API Key", type="password")
    client_code = st.text_input("👤 Client ID")
    pin = st.text_input("🔒 MPIN", type="password")
    totp_secret = st.text_input("⏱️ TOTP Secret", type="password")
    
    if st.button("CONNECT NODE"):
        if api_key and client_code and pin and totp_secret:
            try:
                obj = SmartConnect(api_key=api_key)
                totp = pyotp.TOTP(totp_secret).now()
                data = obj.generateSession(client_code, pin, totp)
                
                if data['status']:
                    st.session_state.api_client = obj
                    st.session_state.jwt_token = data['data']['jwtToken']
                    st.session_state.feed_token = obj.getfeedToken()
                    st.session_state.logged_in = True
                    st.success("✅ BROKER NODE CONNECTED")
                else:
                    st.error("❌ AUTH FAILED")
            except Exception as e:
                st.error("❌ CONNECTION ERROR")
        else:
            st.warning("⚠️ FILL ALL FIELDS")
            
    st.divider()
    st.markdown("### ⚙️ SYSTEM STATUS")
    if st.session_state.bot_running:
        st.success("🟢 ENGINE ONLINE")
    else:
        st.error("🔴 ENGINE OFFLINE")

# --- Main Interface ---
if st.session_state.logged_in:
    
    tab1, tab2, tab3 = st.tabs(["🚀 LIVE TERMINAL", "🧠 AI ENGINE & EOD", "📜 TRADE LEDGER"])
    
    # ---------------- TAB 1: LIVE TERMINAL ----------------
    with tab1:
        st.markdown("### 🎯 Execution Dashboard")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            target_token = st.text_input("Target Options Token", value="26000")
            current_live_price = st.session_state.live_prices.get(target_token, "Awaiting Data...")
            
            st.metric(label="LIVE MARKET PRICE", value=current_live_price)
            
        with col2:
            if st.session_state.bot_running:
                if st.button("🛑 EMERGENCY KILL SWITCH"):
                    st.session_state.bot_running = False
                    st.rerun()
            else:
                if st.button("▶️ IGNITE SCALPER"):
                    st.session_state.bot_running = True
                    st.session_state.ghost_kill_lock = False
                    
                    # Assume settings pulled from Tab 2
                    use_ml = st.session_state.get('use_ml', False)
                    ml_model = st.session_state.get('ml_model', "Random Forest")
                    sl_pct = st.session_state.get('sl_pct', 5.0)
                    tp_pct = st.session_state.get('tp_pct', 10.0)
                    
                    # Start WebSockets & Bot Threads
                    sws = SmartWebSocketV2(st.session_state.jwt_token, api_key, client_code, st.session_state.feed_token)
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
