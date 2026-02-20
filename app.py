import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import requests
import pyotp
from SmartApi import SmartConnect

# ==========================================
# 1. TECHNICAL ANALYZER (VOLUME & TREND LOGIC)
# ==========================================
class TechnicalAnalyzer:
    def calculate_volume_breakout(self, df, vol_length=20, vol_multiplier=1.8):
        if df is None or len(df) < vol_length + 1: return "WAIT", "WAIT"
        df = df.copy()
        
        # Calculate the rolling average volume
        df['avg_vol'] = df['volume'].rolling(window=vol_length).mean().shift(1)
        
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        current_vol = df['volume'].iloc[-1]
        prev_avg_vol = df['avg_vol'].iloc[-1]

        # Failsafe for 0 volume ticks
        if current_vol == 0 or pd.isna(prev_avg_vol) or prev_avg_vol == 0:
            return "WAIT", "WAIT"

        # --- TREND IDENTIFICATION (Price + Volume Logic) ---
        trend = "FLAT"
        if current_close > prev_close and current_vol > prev_avg_vol:
            trend = "LONG BUILDUP 🟢"
        elif current_close < prev_close and current_vol > prev_avg_vol:
            trend = "SHORT BUILDUP 🔴"
        elif current_close > prev_close and current_vol < prev_avg_vol:
            trend = "SHORT COVERING 🟡"
        elif current_close < prev_close and current_vol < prev_avg_vol:
            trend = "LONG UNWINDING 🟠"

        # --- TRADE SIGNAL (Breakout Multiplier Logic) ---
        signal = "WAIT"
        if current_vol > (prev_avg_vol * vol_multiplier):
            if current_close > prev_close: 
                signal = "BUY_CE"
            elif current_close < prev_close: 
                signal = "BUY_PE"
                
        return trend, signal

# ==========================================
# 2. CORE BOT ENGINE
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

    def place_real_order(self, symbol, token, qty, side="BUY", exchange="NFO"):
        try:
            orderparams = {
                "variety": "NORMAL", "tradingsymbol": symbol, "symboltoken": str(token),
                "transactiontype": side, "exchange": exchange, "ordertype": "MARKET",
                "producttype": "INTRADAY", "duration": "DAY", "quantity": str(qty)
            }
            orderId = self.api.placeOrder(orderparams)
            return orderId
        except Exception as e:
            st.error(f"Order Error: {e}")
            return None

    def get_strike(self, symbol, spot, signal):
        df = self.token_map
        if df is None: return None, None, None
        opt_type = "CE" if "BUY_CE" in signal else "PE"
        name = symbol
        exch_list = ["NFO"]
        valid_instruments = ['OPTIDX', 'OPTSTK']
        
        # Commodity & Index Logic restored
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

LOT_SIZES = {
    "NIFTY": 75, "BANKNIFTY": 30, "SENSEX": 20, 
    "CRUDEOIL": 100, "NATURALGAS": 1250, "GOLD": 100, "SILVER": 30
}

# Initialize Session State Variables Safely
for key in ['auth', 'bot_active', 'logs', 'trade_history']:
    if key not in st.session_state: 
        st.session_state[key] = False if key not in ['logs', 'trade_history'] else []
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
                    st.error("Login Failed. Check credentials.")
    else:
        st.success(f"Connected: {st.session_state.bot.client_id}")
        if st.button("Logout & Clear"):
            st.session_state.clear()
            st.rerun()

    st.divider()
    st.header("⚙️ Strategy Settings")
    INDEX = st.selectbox("Watchlist", ["NIFTY", "BANKNIFTY", "SENSEX", "CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"])
    TIMEFRAME = st.selectbox("Timeframe", ["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE"], index=2)
    LOTS = st.number_input("Lots", 1, 100, 2)
    SL_PTS = st.number_input("Stop Loss (Points)", 5, 200, 20)
    TGT_PTS = st.number_input("Target (Points)", 10, 500, 40)
    PAPER = st.toggle("📝 Paper Mode (Turn OFF for Real Trading)", True)

    # --- DOWNLOAD BUTTON LOGIC (FIXED) ---
    st.divider()
    st.subheader("📊 Export Data")
    
    if len(st.session_state.trade_history) > 0:
        df_history = pd.DataFrame(st.session_state.trade_history)
    else:
        df_history = pd.DataFrame(columns=["Time", "Symbol", "Type", "Qty", "Entry Price", "Exit Price", "PnL (₹)"])
        
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Excel/CSV",
        data=csv_data,
        file_name=f"Algo_Trades_{dt.date.today()}.csv",
        mime="text/csv"
    )

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
        current_time = dt.datetime.now().time()
        
        # --- DYNAMIC MARKET TIMING ---
        if INDEX in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"]:
            cutoff_time = dt.time(23, 15)  # 11:15 PM Auto-Square-Off for MCX
            cutoff_label = "11:15 PM"
        else:
            cutoff_time = dt.time(15, 15)  # 3:15 PM Auto-Square-Off for NSE/BSE
            cutoff_label = "3:15 PM"

        st.info(f"Bot is active. Waiting for setup on {INDEX}...")
        
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
                    trend, new_signal = bot.analyzer.calculate_volume_breakout(df_candles, vol_length=20, vol_multiplier=1.8)
                    st.session_state.current_trend = trend
                    st.session_state.current_signal = new_signal

                st.metric(f"Live {INDEX} Futures Price", spot)
                st.metric("Market Sentiment (Trend)", st.session_state.current_trend)
                st.metric("Algo Action", st.session_state.current_signal)

                # SCOPING FIX: Safely pull signal from session state
                signal = st.session_state.current_signal

                # --- ENTRY LOGIC ---
                if st.session_state.active_trade is None and signal in ["BUY_CE", "BUY_PE"]:
                    # Prevent new entries if past market cutoff
                    if current_time >= cutoff_time:
                        st.warning(f"⏰ {cutoff_label} Cutoff Reached. No new trades will be initiated today.")
                    else:
                        strike_sym, strike_token, strike_exch = bot.get_strike(INDEX, spot, signal)
                        if strike_sym:
                            opt_ltp = bot.get_live_price(strike_exch, strike_sym, strike_token)
                            if opt_ltp:
                                lot_size = LOT_SIZES.get(INDEX, 10)
                                qty = LOTS * lot_size
                                
                                # REAL TRADING EXECUTION
                                if not PAPER:
                                    order_id = bot.place_real_order(strike_sym, strike_token, qty, "BUY", strike_exch)
                                    if order_id: 
                                        log(f"⚡ REAL ENTRY PLACED: {strike_sym} | Qty: {qty} | ID: {order_id}")
                                    else:
                                        log(f"❌ REAL ENTRY FAILED: Check margin/limits.")
                                else:
                                    log(f"📝 PAPER ENTRY: {strike_sym} @ {opt_ltp} | Qty: {qty}")
                                    
                                st.session_state.active_trade = {
                                    "symbol": strike_sym, 
                                    "token": strike_token, 
                                    "exch": strike_exch,
                                    "type": "CE" if "CE" in strike_sym else "PE",
                                    "entry": opt_ltp,
                                    "qty": qty,
                                    "sl": opt_ltp - SL_PTS,
                                    "tgt": opt_ltp + TGT_PTS
                                }
                
                elif st.session_state.active_trade is None and signal == "WAIT":
                    if current_time >= cutoff_time:
                        st.write("⏰ Market closed for new intraday positions.")
                    else:
                        st.write("Looking for Volume Breakouts...")

                # --- EXIT LOGIC ---
                elif st.session_state.active_trade is not None:
                    trade = st.session_state.active_trade
                    close_price = bot.get_live_price(trade['exch'], trade['symbol'], trade['token'])
                    
                    if close_price:
                        st.success(f"Open Trade: {trade['symbol']} | Entry: {trade['entry']} | LTP: {close_price}")
                        
                        exit_triggered = False
                        exit_reason = ""
                        
                        # Auto SL / TGT / Time Stop Checks
                        if current_time >= cutoff_time:
                            exit_triggered = True
                            exit_reason = f"⏰ {cutoff_label} AUTO-SQUARE-OFF @ {close_price}"
                        elif close_price >= trade['tgt']:
                            exit_triggered = True
                            exit_reason = f"🎯 TARGET HIT @ {close_price}"
                        elif close_price <= trade['sl']:
                            exit_triggered = True
                            exit_reason = f"🛑 SL HIT @ {close_price}"
                            
                        if st.button("Close Trade Manually") or exit_triggered:
                            if not exit_triggered:
                                exit_reason = f"👋 MANUALLY CLOSED @ {close_price}"
                                
                            final_pnl = (close_price - trade['entry']) * trade['qty'] if trade['type'] == "CE" else (trade['entry'] - close_price) * trade['qty']
                            
                            # REAL TRADING EXECUTION
                            if not PAPER:
                                bot.place_real_order(trade['symbol'], trade['token'], trade['qty'], "SELL", trade['exch'])
                                log(f"⚡ REAL EXIT PLACED: {trade['symbol']} | Qty: {trade['qty']}")
                                
                            st.session_state.trade_history.append({
                                "Time": dt.datetime.now().strftime('%H:%M:%S'),
                                "Symbol": trade['symbol'],
                                "Type": trade['type'],
                                "Qty": trade['qty'],
                                "Entry Price": trade['entry'],
                                "Exit Price": close_price,
                                "PnL (₹)": round(final_pnl, 2)
                            })
                            
                            log(f"{exit_reason} | PnL: ₹{round(final_pnl, 2)}")
                            st.session_state.active_trade = None
                        else:
                            st.info(f"Tracking SL: {trade['sl']} | Target: {trade['tgt']}")
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

        BASKET = ["HDFCBANK", "RELIANCE", "ICICIBANK", "INFY", "TCS", "SBIN"]
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
                        if signal != "WAIT" or "BUILDUP" in trend:
                            suggestions.append({"Stock": stock, "Trend": trend, "Action": signal})
                time.sleep(0.4)
                progress_bar.progress((i + 1) / len(BASKET))
            
            if suggestions:
                st.dataframe(pd.DataFrame(suggestions))
            else:
                st.info("No volume breakouts or buildups found right now.")
        
        if was_active: 
            st.session_state.bot_active = True
            st.success("Scan complete. Live dashboard resumed.")
            def send_whatsapp_alert(message):
    # Replace with your actual number and the API key you just received
    phone = "+919876543210" 
    api_key = "123456"
    
    # URL encode the message to handle spaces and symbols
    safe_message = requests.utils.quote(message)
    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={safe_message}&apikey={api_key}"
    
    try:
        requests.get(url, timeout=5)
    except Exception:
        pass # Fail silently so it doesn't crash your trading loop

