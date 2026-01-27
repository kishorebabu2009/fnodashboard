import yfinance as yf
import pandas as pd
import pandas_ta as ta
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Use the exact SECTOR_MAP from your original code
SECTOR_MAP = {
    "BANKING": [
        "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "BANKBARODA", 
        "PNB", "AUBANK", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "INDUSINDBK", 
        "BANKINDIA", "CANBK", "IDBI", "CENTRALBK", "IOB", "UCOBANK"
    ],
    "IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", "TECHM", "COFORGE", 
        "PERSISTENT", "MPHASIS", "KPITTECH", "TATAELXSI", "LTTS", "BSOFT", 
        "CYIENT", "TATATECH", "KFINTECH", "ORACLE"
    ],
    "FINANCE": [
        "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "RECLTD", "PFC", "MUTHOOTFIN", 
        "SHRIRAMFIN", "M&MFIN", "LICI", "HDFCLIFE", "SBILIFE", "ICICIPRULI", 
        "ICICIGI", "ABCAPITAL", "JIOFIN", "ANGELONE", "CDSL", "BSE", "CAMS", 
        "POONAWALLA", "SAMMAANCAP", "IIFL", "HUDCO"
    ],
    "ENERGY/OIL": [
        "RELIANCE", "NTPC", "ONGC", "POWERGRID", "BPCL", "HINDPETRO", "GAIL", 
        "TATAPOWER", "JSWENERGY", "ADANIGREEN", "ADANIENSOL", "OIL", "PETRONET", 
        "IGL", "MGL", "NHPC", "SJVN", "IREDA", "TORNTPOWER", "CESC"
    ],
    "AUTO": [
        "TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR", 
        "EICHERMOT", "ASHOKLEY", "BHARATFORG", "BALKRISIND", "APOLLOTYRE", 
        "MRF", "MOTHERSON", "SONACOMS", "UNOMINDA", "TINDIA"
    ],
    "CONSUMER": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "VBL", 
        "COLPAL", "DABUR", "GODREJCP", "MARICO", "UPL", "BALRAMCHIN", 
        "JUBLFOOD", "KALYANKJIL", "TITAN", "TRENT", "PAGEIND", "METROBRAND"
    ],
    "METALS": [
        "TATASTEEL", "JINDALSTEL", "JSWSTEEL", "HINDALCO", "VEDL", "NMDC", 
        "NATIONALUM", "SAIL", "HINDZINC", "COALINDIA", "HINDCOPPER", "JSL"
    ],
    "HEALTHCARE": [
        "SUNPHARMA", "CIPLA", "DRREDDY", "APOLLOHOSP", "DIVISLAB", "MAXHEALTH", 
        "ZYDUSLIFE", "LUPIN", "AUROPHARMA", "ALKEM", "BIOCON", "GLENMARK", 
        "GRANULES", "TORNTPHARM", "PPLPHARMA", "GLAND"
    ],
    "DEFENSE/INFRA": [
        "LT", "ADANIENT", "ADANIPORTS", "AMBUJACEM", "ACC", "ULTRACEMCO", 
        "BEL", "HAL", "BDL", "GRASIM", "CUMMINSIND", "ABB", "SIEMENS", 
        "POLYCAB", "HAVELLS", "CONCOR", "MAZDOCK", "COCHINSHIP", "SOLARINDS"
    ],
    "REALTY": [
        "DLF", "GODREJPROP", "LODHA", "PRESTIGE", "OBERREALTY", "IRB", "NBCC"
    ],
    "PLATFORMS/MISC": [
        "ZOMATO", "SWIGGY", "NYKAA", "PAYTM", "PBSTECH", "DELHIVERY", 
        "INDIAMART", "DIXON", "INDIGO", "IDEA", "MCX", "EXIDEIND", "ASTRAL"
    ],
    "CHEMICALS": [
        "PIIND", "SRF", "DEEPAKNTR", "TATACHEM", "ATUL", "GUJGASLTD", "COROMANDEL"
    ]
}

def run_scan():
    results = []
    # Loop through symbols just like your 'EXECUTE FULL SCAN' logic
    for sec, symbols in SECTOR_MAP.items():
        for s in symbols:
            try:
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
            except: continue

    df = pd.DataFrame(results)
    
    # Filter for your specific alerts
    score_100 = df[df['SCORE'] >= 100][['Symbol', 'Sector', 'LTP', 'SCORE','CONTRIB']]
    high_conviction = df[(df['ADX'] > 30) & (df['RSI'] > 55) & (df['LTP'] > df['MA20']) & (df['MA50'] > df['MA200']) & (df['LTP'] > df['Pivot']) & (df['LTP'] > df['VWAP']) & (df['ST_Dir'] == "BULL")]
    
    if not score_100.empty or not high_conviction.empty:
        send_email(score_100, high_conviction)

def send_email(df1, df2):
    sender = os.environ.get('EMAIL_SENDER')
    receiver = os.environ.get('EMAIL_RECEIVER')
    password = os.environ.get('EMAIL_PASSWORD') # Gmail App Password

    msg = MIMEMultipart()
    msg['Subject'] = f"🚀 APEX ALERTS: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    
    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #FF4B4B;">🏆 SCORE 100 WALL</h2>
        {df1.to_html(index=False, border=0, classes='table')}
        <hr>
        <h2 style="color: #00CC66;">🚨 HIGH CONVICTION ALERTS</h2>
        <p>Criteria: RSI>50, Price>MA20, Golden Cross, Price>VWAP, SuperTrend Bullish</p>
        {df2[['Symbol', 'LTP', 'RSI']].to_html(index=False, border=0)}
      </body>
    </html>
    """
    msg.attach(MIMEText(html, 'html'))
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())

if __name__ == "__main__":
    run_scan()
