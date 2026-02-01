import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math
import calendar
import pytz
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# --- 1. CORE SYSTEM & THEME ---
st.set_page_config(page_title="Apex Sovereign v170.0", layout="wide", page_icon="🏛️")
st_autorefresh(interval=1 * 60 * 1000, key="apex_refresher")

def get_last_tuesday(dt):
    # Ensure dt is IST-aware
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    last_date = datetime(dt.year, dt.month, last_day, tzinfo=ist)
    offset = (last_date.weekday() - 1) % 7
    res = last_date - timedelta(days=offset)
    
    # Check against current IST time
    if res.date() < dt.date():
        # If last Thursday of this month passed, get next month's
        next_month = dt.replace(day=28) + timedelta(days=5)
        return get_last_tuesday(next_month)
    return res
    
# --- 2. DATA UTILITIES ---
@st.cache_data(ttl=60)
def get_pulse():
    # Corrected Tickers: Use .NS for indices where ^ is unreliable
    idx = {
        "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", 
        "FinNifty": "NIFTY_FIN_SERVICE.NS", 
        "Midcap Select": "^NIFTYMDCPSEL",
        "SENSEX": "^BSESN", 
        "VIX": "^INDIAVIX"
    }
    res = {}
    
    # Download all at once for speed and reliability
    tickers_list = list(idx.values())
    try:
        data = yf.download(tickers_list, period="5d", interval="1d", progress=False)
        
        # Flatten MultiIndex if necessary
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close']
        else:
            close_data = data[['Close']]

        for name, ticker in idx.items():
            if ticker in close_data.columns:
                # Get the last two valid non-NaN prices
                series = close_data[ticker].dropna()
                if len(series) >= 2:
                    current_price = series.iloc[-1]
                    prev_price = series.iloc[-2]
                    change_pct = ((current_price / prev_price) - 1) * 100
                    res[name] = (current_price, change_pct)
                else:
                    res[name] = (0, 0)
            else:
                res[name] = (0, 0)
    except Exception as e:
        st.sidebar.error(f"Pulse Error: {e}")
        # Fallback to zeros
        for name in idx.keys(): res[name] = (0, 0)
        
    return res
# --- 3. TOP BANNER ---
pulse = get_pulse(); 
b = st.columns(len(pulse))
for i, (name, (v, c)) in enumerate(pulse.items()):
    b[i].metric(name, f"{v:,.0f}" if "VIX" not in name else f"{v:.2f}", f"{c:+.2f}%")
st.divider()

if 'master_df' not in st.session_state: st.session_state.master_df = None
if 'watchlist' not in st.session_state: st.session_state.watchlist = []

# --- 4. SCANNER ENGINE ---
SECTOR_MAP = {
    "AUTO": [
        "APOLLOTYRE", "ASHOKLEY", "BAJAJ-AUTO", "BALKRISIND", "BHARATFORG", 
        "EICHERMOT", "HEROMOTOCO", "M&M", "MARUTI", "MOTHERSON", "MRF", 
        "SONACOMS", "TATAMOTORS", "TIINDIA", "TINDIA", "TVSMOTOR", "UNOMINDA"
    ],
    "BANKING": [
        "AUBANK", "AXISBANK", "BANDHANBNK", "BANKBARODA", "BANKINDIA", 
        "CANBK", "CENTRALBK", "FEDERALBNK", "HDFCBANK", "ICICIBANK", 
        "IDBI", "IDFCFIRSTB", "INDUSINDBK", "IOB", "KOTAKBANK", "PNB", 
        "SBIN", "UCOBANK"
    ],
    "CHEMICALS": [
        "ATUL", "COROMANDEL", "DEEPAKNTR", "GUJGASLTD", "PIIND", "SRF", "TATACHEM"
    ],
    "CONSUMER": [
        "BALRAMCHIN", "BRITANNIA", "COLPAL", "DABUR", "GODREJCP", "HINDUNILVR", 
        "IRCTC", "ITC", "JUBLFOOD", "KALYANKJIL", "MARICO", "METROBRAND", 
        "NESTLEIND", "PAGEIND", "TATACONSUM", "TITAN", "TRENT", "UPL", "VBL"
    ],
    "DEFENSE/INFRA": [
        "ABB", "ACC", "ADANIENT", "ADANIPORTS", "AMBUJACEM", "BDL", "BEL", 
        "BHEL", "COCHINSHIP", "CONCOR", "CUMMINSIND", "GRASIM", "HAL", 
        "HAVELLS", "LT", "MAZDOCK", "POLYCAB", "RVNL", "SIEMENS", "SOLARINDS", 
        "ULTRACEMCO"
    ],
    "ENERGY/OIL": [
        "ADANIGREEN", "ADANIENSOL", "BPCL", "CESC", "GAIL", "HINDPETRO", 
        "IGL", "IREDA", "JSWENERGY", "MGL", "NHPC", "NTPC", "OIL", "ONGC", 
        "PETRONET", "POWERGRID", "RELIANCE", "SJVN", "TATAPOWER", "TORNTPOWER"
    ],
    "FINANCE": [
        "ABCAPITAL", "ANGELONE", "BAJFINANCE", "BAJAJFINSV", "BSE", "CAMS", 
        "CDSL", "CHOLAFIN", "HDFCLIFE", "HUDCO", "ICICIGI", "ICICIPRULI", 
        "IIFL", "IRFC", "JIOFIN", "LICI", "M&MFIN", "MUTHOOTFIN", "PFC", 
        "POONAWALLA", "RECLTD", "SAMMAANCAP", "SBILIFE", "SHRIRAMFIN"
    ],
    "HEALTHCARE": [
        "ALKEM", "APOLLOHOSP", "AUROPHARMA", "BIOCON", "CIPLA", "DIVISLAB", 
        "DRREDDY", "GLAND", "GLENMARK", "GRANULES", "LUPIN", "MAXHEALTH", 
        "PPLPHARMA", "SUNPHARMA", "TORNTPHARM", "ZYDUSLIFE"
    ],
    "IT": [
        "BSOFT", "COFORGE", "CYIENT", "HCLTECH", "INFY", "KFINTECH", 
        "KPITTECH", "LTIM", "LTTS", "MPHASIS", "ORACLE", "PERSISTENT", 
        "TATAELXSI", "TATATECH", "TCS", "TECHM", "WIPRO"
    ],
    "METALS": [
        "COALINDIA", "HINDALCO", "HINDZINC", "JINDALSTEL", "JSL", "JSWSTEEL", 
        "NATIONALUM", "NMDC", "SAIL", "TATASTEEL", "VEDL"
    ],
    "PLATFORMS/MISC": [
        "ASTRAL", "DELHIVERY", "DIXON", "EXIDEIND", "IDEA", "INDIGO", 
        "INDIAMART", "MCX", "NYKAA", "PAYTM", "PBSTECH", "SWIGGY", "ZOMATO"
    ],
    "REALTY": [
        "DLF", "GODREJPROP", "IRB", "LODHA", "NBCC", "OBERREALTY", "PRESTIGE"
    ]
}

ist = pytz.timezone('Asia/Kolkata')
today = datetime.now(ist)
exp_dt = get_last_tuesday(today) # Assuming your helper function is defined

with st.sidebar:
    st.header("⚙️ APEX COMMAND")
    # Clock: Displays full Date and 24hr Time
    st.metric(
        label="🕒 CLOCK", 
        value=today.strftime('%d %b %Y, %H:%M:%S').upper()
    )
    # Expiry: Displays Date with Days Remaining as the 'delta'
    days_to_expiry = (exp_dt.date() - today.date()).days
    st.metric(
        label="📅 EXPIRY", 
        value=exp_dt.strftime('%d %b').upper(), 
        delta=f"{days_to_expiry} days"
    )
    st.divider()
    # --- FIX STARTS HERE ---
    if st.session_state.master_df is not None:
        # Check if master_df is populated and columns exist
        if not st.session_state.master_df.empty and 'SCORE' in st.session_state.master_df.columns:
            
            # Filter for elite stocks with score of 100
            elite_df = st.session_state.master_df[st.session_state.master_df['SCORE'] >= 100].copy()
            
            if not elite_df.empty:
                st.subheader("🏆 SCORE 100 WALL")
                
                # Selecting and Renaming for a professional look
                display_df = elite_df[['Symbol', 'Sector', 'CONTRIB']].rename(
                    columns={'CONTRIB': 'Logic Breakdown'}
                )
                
                # Displaying as a clean table
                st.dataframe(
                    display_df, 
                    hide_index=True, 
                    use_container_width=True
                )
                st.caption("✨ Logic: Trend | Momentum | Volatility | Volume")
                st.divider()
    # --- FIX ENDS HERE --- 
    sel_sec = st.multiselect("Sectors", list(SECTOR_MAP.keys()), default=list(SECTOR_MAP.keys()))
    if st.button("🚀 EXECUTE FULL SCAN", use_container_width=True):
        results = []
        targets = [(s, sec) for sec in sel_sec for s in SECTOR_MAP[sec]]
        p_txt = st.empty()
        for i, (s, sec) in enumerate(targets):
            p_txt.info(f"[{i+1}/{len(targets)}] | `{s}` | {sec}")
            d = yf.download(f"{s}.NS", period="1y", interval="1d", progress=False)
            if not d.empty and len(d) > 100:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume']
                
                # Indicators
                ma20, ma50, ma200 = ta.sma(c, 20), ta.sma(c, 50), ta.sma(c, 200)
                rsi = ta.rsi(c, 14)
                atr = ta.atr(h, l, c, 14)
                vwap = ( ( (h + l + c) / 3 ) * v).rolling(14).sum() / v.rolling(14).sum()
                adx = ta.adx(h, l, c)
                st_df = ta.supertrend(h, l, c, 7, 3)
                
                # CPR
                ph, pl, pc = h.iloc[-2], l.iloc[-2], c.iloc[-2]
                pivot = (ph + pl + pc) / 3
                bc = (ph + pl) / 2
                tc = (pivot - bc) + pivot
                
                # --- IMPROVED MULTI-FACTOR SCORING ENGINE ---
                curr_c = c.iloc[-1]
                
                # 1. Trend Factor (Max 40 pts)
                # Reward based on proximity to MA50 and MA200
                s1 = 0
                if curr_c > ma20.iloc[-1]: s1 += 10
                if curr_c > ma50.iloc[-1]: s1 += 10
                if curr_c > ma200.iloc[-1]: s1 += 10
                if ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]: s1 += 10 # Golden Alignment bonus
                s1 = min(s1, 40) # Cap at 40

                # 2. Momentum & Overbought Protection (Max 30 pts)
                s2 = 0
                curr_rsi = rsi.iloc[-1]
                if 55 <= curr_rsi <= 70: 
                    s2 = 30  # "Sweet Spot" Momentum
                elif curr_rsi > 75:
                    s2 = 15  # Caution: Overbought territory
                elif 45 <= curr_rsi < 55:
                    s2 = 15  # Building strength
                
                # 3. Trend Intensity & Volatility (Max 30 pts)
                s3 = 0
                curr_adx = adx.iloc[-1, 0]
                is_bull_st = (st_df is not None and st_df.iloc[-1, 1] > 0)
                
                if curr_c > vwap.iloc[-1]: s3 += 5
                if is_bull_st: s3 += 5
                if curr_adx > 25: s3 += 15 # Strong Trend Confirmation
                elif curr_adx > 40: s3 += 5 # Exhaustion risk but high strength

                # 4. Volume Surge (Bonus 10 pts)
                # Compare current volume to 20-day average
                vol_ma20 = v.rolling(20).mean().iloc[-1]
                s4 = 10 if v.iloc[-1] > (vol_ma20 * 1.5) else 0

                final_score = min(s1 + s2 + s3 + s4, 100)
                # Create detailed contribution string for Deep Dive
                contrib_msg = f"Golden-cross:{s1} | RelativeStrength:{s2} | TrendIntensity:{s3} | V-Surge:{s4}"
                # --- END SCORING ENGINE ---
                
                results.append({
                    'Symbol': s, 'Sector': sec, 'SCORE': final_score, 'LTP': c.iloc[-1],
                    'CHG': round(((c.iloc[-1]/c.iloc[-2])-1)*100,2), 'Gap_Pct': round(((d['Open'].iloc[-1]/c.iloc[-2])-1)*100,2),
                    'RSI': round(rsi.iloc[-1],2), 'ATR': round(atr.iloc[-1],2), 'ADX': round(adx.iloc[-1, 0],2),
                    'MA20': round(ma20.iloc[-1],2), 'MA50': round(ma50.iloc[-1],2), 'MA200': round(ma200.iloc[-1],2),
                    'VWAP': round(vwap.iloc[-1],2), 'Pivot': round(pivot,2), 'TC': round(tc,2), 'BC': round(bc,2),
                    'ST_Dir': "BULL" if is_bull_st else "BEAR", 'VFI': round((v.iloc[-1]/v.rolling(20).mean().iloc[-1]),2),
                    'CONTRIB': contrib_msg
                })
        st.session_state.master_df = pd.DataFrame(results).fillna(0)
        st.rerun()

# --- 5. THE 18-TAB SUITE ---
df = st.session_state.master_df
if df is not None:
    t = st.tabs(["🔍 Scan", "📊 Tactical", "🎯 Range", "🔭 Search", "🤖 Verdict", "🎯 CPR Hub", "🛡️ Risk", "🏗️ Rotation", "📈 Vol", "⭐ Watch", "📉 Flows", "🔭 Deep Dive", "📉 Backtest", "🔥 Heatmap", "🔔 Alerts", "📊 Sector", "☁️ Cloud IQ", "📄 Export"])

    with t[0]: # SCAN
        st.dataframe(df.sort_values('SCORE', ascending=False), use_container_width=True, hide_index=True)

    with t[1]: # TAB 2: TACTICAL (High-Frequency Analysis)
        st.header("📊 Tactical Execution Hub")
        
        # 1. Selection & Refresh Logic
        col_a, col_b = st.columns([1, 3])
        with col_a:
            t_sel = st.selectbox("🎯 Target Symbol", df['Symbol'].unique(), key="tactical_sel")
            t_data = df[df['Symbol'] == t_sel].iloc[0]
            
            st.metric("LTP", f"₹{t_data['LTP']:.2f}", f"{t_data['CHG']:.2f}%")
            st.write(f"**Trend Status:** {t_data['ST_Dir']}")
            st.write(f"**ADX Strength:** {t_data['ADX']:.2f}")
            
            # Actionable Signal Gauge
            if t_data['SCORE'] >= 80:
                st.success("🔥 SIGNAL: STRONG BUY")
            elif t_data['SCORE'] >= 50:
                st.info("⚡ SIGNAL: ACCUMULATE")
            else:
                st.warning("⚠️ SIGNAL: NEUTRAL/WATCH")
        
        with col_b:
            # 2. Fetch Intraday Data
            with st.spinner(f"Loading {t_sel} Intraday Profile..."):
                h_df = yf.download(f"{t_sel}.NS", period="5d", interval="15m", progress=False)
                if isinstance(h_df.columns, pd.MultiIndex): h_df.columns = h_df.columns.get_level_values(0)
                
                # Calculate Intraday EMA for the chart
                h_df['EMA20'] = ta.ema(h_df['Close'], length=20)
                
            # 3. Plotly Candlestick with Overlays
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=h_df.index, open=h_df['Open'], high=h_df['High'], 
                low=h_df['Low'], close=h_df['Close'], name="Price"
            ))

            # EMA 20 Overlay
            fig.add_trace(go.Scatter(
                x=h_df.index, y=h_df['EMA20'], 
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5), 
                name="EMA 20"
            ))

            fig.update_layout(
                height=500,
                template="plotly_dark",
                title=f"{t_sel} - 15m Intraday Structure",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
     

        # 4. Support & Resistance Quick-View
        st.subheader("🛡️ Tactical Levels (Daily)")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Pivot", f"{t_data['Pivot']:.2f}")
        l2.metric("BC (Support)", f"{t_data['BC']:.2f}")
        l3.metric("TC (Resistance)", f"{t_data['TC']:.2f}")
        l4.metric("ATR (Volatility)", f"{t_data['ATR']:.2f}")
        
    with t[2]: # TAB 3: RANGE
        vix = pulse.get("VIX", (15, 0))[0]
        st.subheader("🏛️ Global Index Probability Matrix")
        st.write(f"Current VIX: **{vix:.2f}** | Probability: **68% (1-Std Dev)**")
        
        idx_m = []
        # Mapping the pulse keys to readable names
        for n in ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]:
            if n in pulse:
                curr = pulse[n][0]
                # Standard Deviation Range Formula: Price * (VIX/100) * SQRT(Days/365)
                def calc_range(days):
                    move = curr * (vix/100) * math.sqrt(days/365)
                    return f"{curr-move:,.0f} - {curr+move:,.0f}"
                
                idx_m.append({
                    "Market Index": n,
                    "LTP": f"{curr:,.2f}",
                    "Daily Range": calc_range(1),
                    "Weekly Range": calc_range(7),
                    "Monthly Range": calc_range(30)
                })
        st.table(pd.DataFrame(idx_m))
        st.divider()
        # --- Symbol Specific Range Calculator ---
        st.subheader("🎯 Symbol-Specific Range Forecast")
        s_range_sel = st.selectbox("Select Symbol for Range Projection", df['Symbol'].unique(), key="range_calc_sel")
        
        s_data = df[df['Symbol'] == s_range_sel].iloc[0]
        s_price = s_data['LTP']
        # Individual stocks usually have higher volatility than indices. 
        # We use a 25% Annual Volatility constant as a conservative estimate for equities.
        def calc_sym_range(days):
            s_vol = 0.25 # 25% Annual Vol
            s_move = s_price * s_vol * math.sqrt(days/365)
            return f"{s_price-s_move:,.2f} - {s_price+s_move:,.2f}"
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric(f"{s_range_sel} Daily", calc_sym_range(1))
        r_col2.metric(f"{s_range_sel} Weekly", calc_sym_range(7))
        r_col3.metric(f"{s_range_sel} Monthly", calc_sym_range(30))      
        st.info("💡 **Institutional Insight:** The Daily Range (1-SD) represents the boundary within which the price is expected to stay 68% of the time based on current implied volatility.")
    
    with t[3]: # SEARCH
        sq = st.text_input("🔭 Search Ticker", placeholder="e.g. RELIANCE")
        if sq: st.dataframe(df[df['Symbol'].str.contains(sq.upper())])

    with t[4]: # VERDICT
        st.header("🤖 Apex Institutional Verdict")
        v1, v2 = st.columns(2)
        v1.success("### 🚀 TOP 10 BULLS")
        v1.table(df.nlargest(10, 'SCORE')[['Symbol', 'SCORE', 'LTP', 'CHG']])
        v2.error("### 📉 TOP 10 BEARS")
        v2.table(df.nsmallest(10, 'SCORE')[['Symbol', 'SCORE', 'LTP', 'CHG']])
        

    with t[5]: # CPR HUB
        st.subheader("🎯 Central Pivot Range Analysis")
        df['CPR_W'] = abs(df['TC'] - df['BC']) / df['Pivot'] * 100
        st.dataframe(df.sort_values('CPR_W')[['Symbol', 'CPR_W', 'TC', 'Pivot', 'BC', 'ST_Dir']], hide_index=True)
        

    with t[6]: # RISK
        cap = st.number_input("Capital", value=100000)
        risk_df = df.copy()
        risk_df['Qty'] = (cap * 0.01) / risk_df['ATR']
        st.dataframe(risk_df[['Symbol', 'LTP', 'ATR', 'Qty']], hide_index=True)

    with t[7]: # ROTATION
        st.plotly_chart(px.scatter(df, x="RSI", y="ADX", color="Sector", size="SCORE", text="Symbol"))

    with t[8]: # VOL
        st.plotly_chart(px.bar(df, x="Symbol", y="ATR", color="Sector"))

    with t[9]: # WATCHLIST
        w_sel = st.multiselect("Watchlist", df['Symbol'].unique(), default=st.session_state.watchlist)
        st.session_state.watchlist = w_sel
        if w_sel: st.dataframe(df[df['Symbol'].isin(w_sel)])

    with t[10]: # FLOWS
        st.plotly_chart(px.bar(df, x="Symbol", y="VFI", color="VFI", color_continuous_scale="RdYlGn"))

    with t[11]: # DEEP DIVE (Fundamentals + Peer Comparison)
        dd_sel = st.selectbox("Deep Dive Target", df['Symbol'].unique())
        
        # Define a cached function to fetch info and handle rate limits
        @st.cache_data(ttl=3600) # Cache for 1 hour to prevent re-triggering limits
        def fetch_ticker_info(symbol):
            try:
                tick = yf.Ticker(f"{symbol}.NS")
                return tick.info
            except Exception as e:
                return {"error": str(e)}

        inf = fetch_ticker_info(dd_sel)
        
        if "error" in inf or not inf:
            st.error("⚠️ Yahoo Finance Rate Limit reached. Fundamental data is temporarily unavailable.")
            st.info("Try again in a few minutes. Technical scanning and charts will still work.")
        else:
            d1, d2 = st.columns([1, 2])
            with d1:
                st.metric("Market Cap", f"₹{inf.get('marketCap', 0)//10**7:,.0f} Cr")
                st.metric("P/E Ratio", f"{inf.get('trailingPE', 'N/A')}")
                st.metric("Beta", f"{inf.get('beta', 'N/A')}")
            with d2:
                st.subheader(inf.get('longName', dd_sel))
                st.write(inf.get('longBusinessSummary', 'N/A')[:600] + "...")
                st.write("---")
                st.subheader("👥 Sector Peer Comparison")
                sector_name = df[df['Symbol'] == dd_sel]['Sector'].values[0]
                peers = df[df['Sector'] == sector_name]
                st.dataframe(peers[['Symbol', 'SCORE', 'LTP', 'CHG', 'RSI']], hide_index=True)
    
    with t[12]: # BACKTEST
        st.info("Strategy: SMA50 Trend Following")
        st.dataframe(df[['Symbol', 'ST_Dir', 'MA50', 'MA200']])

    with t[13]: # HEATMAP
        st.plotly_chart(px.treemap(df, path=['Sector', 'Symbol'], values='SCORE', color='CHG', color_continuous_scale='RdYlGn'))

    with t[14]: # ALERTS
        st.subheader("🚨 High-Conviction Bullish Alerts")
        # Multi-factor logic: RSI > 50, Price > MA20, Golden Cross (MA50 > MA200), Price > VWAP, and SuperTrend Bullish
        alerts_df = df[(df['ADX'] > 30) & (df['RSI'] > 55) & (df['LTP'] > df['MA20']) & (df['MA20'] > df['MA50']) & (df['MA50'] > df['MA200']) & (df['LTP'] > df['Pivot']) & (df['LTP'] > df['VWAP']) & (df['ST_Dir'] == "BULL")]
        
        if not alerts_df.empty:
            st.success(f"🔥 Found {len(alerts_df)} stocks meeting all technical criteria")
            st.dataframe(alerts_df[['Symbol', 'LTP', 'VWAP', 'Pivot', 'SCORE', 'RSI', 'ADX', 'ST_Dir']], use_container_width=True, hide_index=True)
        else:
            st.info("No stocks currently meet the combined multi-factor criteria.")

    with t[15]: # SECTOR
        st.plotly_chart(px.sunburst(df, path=['Sector', 'Symbol'], values='SCORE'))

    with t[16]: # CLOUD IQ
        st.write("#### ☁️ Pivot & VWAP Convergence")
        st.dataframe(df[['Symbol', 'LTP', 'Pivot', 'VWAP', 'MA20']])

    with t[17]: # EXPORT
        st.download_button("📥 Export Report", df.to_csv(index=False), "Apex_Full_Report.csv")

else:
    st.info("System Standby. Execute Market Scan to activate modules.")
    











