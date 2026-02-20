import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import requests
import pyotp
from SmartApi import SmartConnect

# ==========================================
# 1. TECHNICAL ANALYZER (VWAP + EMA SCALPING)
# ==========================================
class TechnicalAnalyzer:
    def calculate_scalp_signals(self, df, vol_length=20, vol_multiplier=1.5, ema_length=9):
        if df is None or len(df) < vol_length + 1: 
            return "WAIT", "WAIT", 0, 0
            
        df = df.copy()
        
        # 1. Parse Datetime for Intraday VWAP calculation
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # 2. Calculate VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['cum_tp_vol'] = (df['typical_price'] * df['volume']).groupby(df['date']).cumsum()
        df['cum_vol'] = df['volume'].groupby(df['date']).cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        df['vwap'] = df['vwap'].ffill() 
        
        # 3. Calculate Fast EMA
        df['ema'] = df['close'].ewm(span=ema_length, adjust=False).mean()
        
        # 4. Calculate Volume Average
        df['avg_vol'] = df['volume'].rolling(window=vol_length).mean().shift(1)
        
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        current_open = df['open'].iloc[-1]
        current_vol = df['volume'].iloc[-1]
        prev_avg_vol = df['avg_vol'].iloc[-1]
        current_vwap = df['vwap'].iloc[-1]
        current_ema = df['ema'].iloc[-1]

        if current_vol == 0 or pd.isna(prev_avg_vol) or prev_avg_vol == 0:
            return "WAIT", "WAIT", current_vwap, current_ema

        # --- TREND IDENTIFICATION ---
        trend = "FLAT"
        if current_close > prev_close and current_vol > prev_avg_vol:
            trend = "LONG BUILDUP 🟢"
        elif current_close < prev_close and current_vol > prev_avg_vol:
            trend = "SHORT BUILDUP 🔴"
        elif current_close > prev_close and current_vol < prev_avg_vol:
            trend = "SHORT COVERING 🟡"
        elif current_close < prev_close and current_vol < prev_avg_vol:
            trend = "LONG UNWINDING 🟠"

        # --- VWAP SCALPING SIGNAL LOGIC ---
        signal = "WAIT"
        if current_vol > (prev_avg_vol * vol_multiplier):
            if current_close > current_open and current_ema > current_vwap: 
                signal = "BUY_CE"
            elif current_close < current_open and current_ema < current_vwap: 
                signal = "BUY_PE"
                
        return trend, signal, current_vwap, current_ema

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

    def get_strike(self, symbol, spot, signal, max_premium):
        df = self.token_map
        if df is None: return None, None, None
        
        is_ce = "BUY_CE" in signal
        opt_type = "CE" if is_ce else "PE"
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
        
        # Filter for closest expiry
        subset = subset[subset['expiry'] == subset['expiry'].min()]
        
        # Sort strikes to prioritize ATM first, then move OTM to find cheaper options
        if is_ce:
            # For CE, we look at strikes >= spot (moving OTM)
            candidates = subset[subset['strike'] >= spot].sort_values('strike', ascending=True)
        else:
            # For PE, we look at strikes <= spot (moving OTM)
            candidates = subset[subset['strike'] <= spot].sort_values('strike', ascending=False)
            
        # Limit to checking 15 strikes to avoid API rate limits
        for _, row in candidates.head(15).iterrows():
            ltp = self.get_live_price(row['exch_seg'], row['symbol'], row['token'])
            # Check if we can afford this strike
            if ltp and ltp <= max_premium:
                return row['symbol'], row['token'], row['exch_seg']
                
        return None, None, None

# ==========================================
# 3. STREAMLIT UI & SECURITY
# ==========================================
st.set_page_config(page_title="Pro Scalper Bot", page_icon="⚡", layout="wide")

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
    st.header("⚙️ Scalping Settings")
    INDEX = st.selectbox("Watchlist", ["NIFTY", "BANKNIFTY", "SENSEX", "CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"])
    TIMEFRAME = st.selectbox("Timeframe", ["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE"], index=2)
    VOL_MULT = st.slider("Volume Sensitivity", 1.1, 3.0, 1.5, 0.1, help="1.5 = Needs 50% more volume than average.")
    
    LOTS = st.number_input("Lots", 1, 100, 2)
    MAX_CAPITAL = st.number_input("Max Capital (₹)", 1000, 500000, 10000, step=1000, help="Bot will only buy options that fit this budget.")
    
    SL_PTS = st.number_input("Stop Loss (Points)", 5, 200, 20)
    TGT_PTS = st.number_input("Target (Points)", 10, 500, 40)
    PAPER = st.toggle("📝 Paper Mode (Turn OFF for Real Trading)", True)

    # --- DOWNLOAD BUTTON LOGIC ---
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
    st.title("Welcome to Pro Scalper Bot")
    st.warning("Please connect using the sidebar with your Angel One details to begin.")
    st.stop()

tab1, tab2 = st.tabs(["⚡ Live VWAP Scalper", "🔎 Equity Scanner"])
bot = st.session_state.bot

with tab1:
    col1, col2 = st.columns(2)
    if col1.button("🟢 START BOT"): st.session_state.bot_active = True
    if col2.button("🔴 STOP BOT"): st.session_state.bot_active = False

    if st.session_state.bot_active:
        current_time = dt.datetime.now().time()
        
        # --- DYNAMIC MARKET TIMING ---
        if INDEX in ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"]:
            cutoff_time = dt.time(23, 15)
            cutoff_label = "11:15 PM"
        else:
            cutoff_time = dt.time(15, 15)
            cutoff_label = "3:15 PM"

        # --- TOTAL DAILY PNL DISPLAY ---
        total_pnl = sum([t.get("PnL (₹)", 0.0) for t in st.session_state.trade_history])
        pnl_color = "normal" if total_pnl == 0 else ("inverse" if total_pnl < 0 else "normal") # Visual cue
        st.metric("💰 Total Daily Realized PnL", f"₹{round(total_pnl, 2)}")
        st.divider()
        
        st.info(f"Bot is active. Waiting for VWAP crossover on {INDEX}...")
        
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
                
                # --- LIVE METRICS DASHBOARD ---
                if df_candles is not None and not df_candles.empty:
                    trend, new_signal, vwap_val, ema_val = bot.analyzer.calculate_scalp_signals(df_candles, vol_length=20, vol_multiplier=VOL_MULT, ema_length=9)
                    st.session_state.current_trend = trend
                    st.session_state.current_signal = new_signal
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"Live {INDEX} Futures", spot)
                    m2.metric("Intraday VWAP", round(vwap_val, 2))
                    m3.metric("9-EMA", round(ema_val, 2))
                    
                    current_vol = int(df_candles['volume'].iloc[-1])
                    avg_vol = int(df_candles['volume'].rolling(window=20).mean().shift(1).iloc[-1]) if len(df_candles) > 20 else 0
                    m4.metric(f"Volume vs Avg", f"{current_vol} / {avg_vol}")
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Market Sentiment", st.session_state.current_trend)
                    c2.metric("Algo Action", st.session_state.current_signal)
                else:
                    st.error("⚠️ API is returning empty historical data for this asset right now.")

                signal = st.session_state.current_signal

                # --- ENTRY LOGIC ---
                if st.session_state.active_trade is None and signal in ["BUY_CE", "BUY_PE"]:
                    if current_time >= cutoff_time:
                        st.warning(f"⏰ {cutoff_label} Cutoff Reached. No new trades will be initiated.")
                    else:
                        lot_size = LOT_SIZES.get(INDEX, 10)
                        qty = LOTS * lot_size
                        max_premium_allowed = MAX_CAPITAL / qty
                        
                        strike_sym, strike_token, strike_exch = bot.get_strike(INDEX, spot, signal, max_premium_allowed)
                        
                        if strike_sym:
                            opt_ltp = bot.get_live_price(strike_exch, strike_sym, strike_token)
                            if opt_ltp:
                                if not PAPER:
                                    order_id = bot.place_real_order(strike_sym, strike_token, qty, "BUY", strike_exch)
                                    if order_id: 
                                        log(f"⚡ REAL ENTRY: {strike_sym} | Qty: {qty} | ID: {order_id}")
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
                        else:
                            st.warning(f"⚠️ No strike found within budget of ₹{MAX_CAPITAL} (Max premium limit: ₹{max_premium_allowed:.2f}).")
                
                elif st.session_state.active_trade is None and signal == "WAIT":
                    if current_time >= cutoff_time:
                        st.write(f"⏰ Market closed for new intraday positions (Past {cutoff_label}).")
                    else:
                        st.write("Looking for VWAP Crossover & Volume Spikes...")

                # --- EXIT LOGIC & FLOATING PNL ---
                elif st.session_state.active_trade is not None:
                    trade = st.session_state.active_trade
                    close_price = bot.get_live_price(trade['exch'], trade['symbol'], trade['token'])
                    
                    if close_price:
                        # Calculate Live Floating PnL
                        floating_pnl = (close_price - trade['entry']) * trade['qty'] if trade['type'] == "CE" else (trade['entry'] - close_price) * trade['qty']
                        pnl_indicator = "🟢" if floating_pnl >= 0 else "🔴"
                        
                        st.info(f"📈 Open Trade: **{trade['symbol']}** | Entry: **{trade['entry']}** | LTP: **{close_price}** | Live PnL: {pnl_indicator} **₹{round(floating_pnl, 2)}**")
                        
                        exit_triggered = False
                        exit_reason = ""
                        
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
                                "PnL (₹)": round(floating_pnl, 2)
                            })
                            
                            log(f"{exit_reason} | Final PnL: ₹{round(floating_pnl, 2)}")
                            st.session_state.active_trade = None
                        else:
                            st.caption(f"Tracking SL: {trade['sl']} | Target: {trade['tgt']}")
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
                        trend, signal, v, e = bot.analyzer.calculate_scalp_signals(hist, vol_length=20, vol_multiplier=VOL_MULT, ema_length=9)
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
