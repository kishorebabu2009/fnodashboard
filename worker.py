import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
import pytz

# --- NSE HOLIDAY LIST 2026 ---
# Format: YYYY-MM-DD
NSE_HOLIDAYS = [
    "2026-01-26", # Republic Day
    "2026-03-06", # Holi
    "2026-03-27", # Ram Navami
    "2026-04-02", # Mahavir Jayanti
    "2026-04-03", # Good Friday
    "2026-04-14", # Dr. Ambedkar Jayanti
    "2026-05-01", # Maharashtra Day
    "2026-08-15", # Independence Day
    "2026-10-02", # Gandhi Jayanti
    "2026-10-21", # Dussehra
    "2026-11-09", # Diwali-Laxmi Pujan
    "2026-12-25", # Christmas
]

def is_market_open():
    """Checks if today is a weekday and not a NSE holiday"""
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(tz)
    date_str = now.strftime('%Y-%m-%d')
    
    # Check Weekend (5 = Saturday, 6 = Sunday)
    if now.weekday() >= 5:
        return False, "Weekend"
    
    # Check Holiday List
    if date_str in NSE_HOLIDAYS:
        return False, f"NSE Holiday: {date_str}"
    
    return True, "Market Open"


# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def calculate_indicators(df):
    """Manual technical indicator math (No-Dependency Mode)"""
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

    # Price Change % calculation
    prev_close = c.iloc[-2]
    curr_close = c.iloc[-1]
    price_chg_pct = ((curr_close - prev_close) / prev_close) * 100
    
    # Moving Averages
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()
    
    # RSI
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    
    # VWAP
    vwap = (((h + l + c) / 3) * v).rolling(14).sum() / v.rolling(14).sum()
    
    # ADX (Simplified Directional Movement)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = (h.diff().where((h.diff() > l.diff().abs()) & (h.diff() > 0), 0)).rolling(14).mean()
    minus_dm = (l.diff().abs().where((l.diff().abs() > h.diff()) & (l.diff().abs() > 0), 0)).rolling(14).mean()
    adx = 100 * (plus_dm - minus_dm).abs() / (plus_dm + minus_dm)
    
    # SuperTrend
    hl2 = (h + l) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    st_dir = [True] * len(df)
    for i in range(1, len(df)):
        if c.iloc[i] > upper_band.iloc[i-1]: st_dir[i] = True
        elif c.iloc[i] < lower_band.iloc[i-1]: st_dir[i] = False
        else: st_dir[i] = st_dir[i-1]
        
    # CPR Logic
    ph, pl, pc = h.iloc[-2], l.iloc[-2], c.iloc[-2]
    pivot = (ph + pl + pc) / 3
    
    return {
        'ma20': ma20.iloc[-1], 'ma50': ma50.iloc[-1], 'ma200': ma200.iloc[-1],
        'rsi': rsi.iloc[-1], 'vwap': vwap.iloc[-1], 'adx': adx.iloc[-1],
        'st_bull': st_dir[-1], 'pivot': pivot, 'ltp': c.iloc[-1],'chg_pct': price_chg_pct
    }

# --- SECTOR MAP ---
SECTOR_MAP = {
    "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "BANKBARODA", "PNB", "AUBANK", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "INDUSINDBK", "BANKINDIA", "CANBK", "IDBI", "CENTRALBK", "IOB", "UCOBANK"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", "TECHM", "COFORGE", "PERSISTENT", "MPHASIS", "KPITTECH", "TATAELXSI", "LTTS", "BSOFT", "CYIENT", "TATATECH", "KFINTECH", "ORACLE"],
    "FINANCE": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "RECLTD", "PFC", "MUTHOOTFIN", "SHRIRAMFIN", "M&MFIN", "LICI", "HDFCLIFE", "SBILIFE", "ICICIPRULI", "ICICIGI", "ABCAPITAL", "JIOFIN", "ANGELONE", "CDSL", "BSE", "CAMS", "POONAWALLA", "SAMMAANCAP", "IIFL", "HUDCO"],
    "ENERGY/OIL": ["RELIANCE", "NTPC", "ONGC", "POWERGRID", "BPCL", "HINDPETRO", "GAIL", "TATAPOWER", "JSWENERGY", "ADANIGREEN", "ADANIENSOL", "OIL", "PETRONET", "IGL", "MGL", "NHPC", "SJVN", "IREDA", "TORNTPOWER", "CESC"],
    "AUTO": ["TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR", "EICHERMOT", "ASHOKLEY", "BHARATFORG", "BALKRISIND", "APOLLOTYRE", "MRF", "MOTHERSON", "SONACOMS", "UNOMINDA", "TINDIA"],
    "CONSUMER": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "VBL", "COLPAL", "DABUR", "GODREJCP", "MARICO", "UPL", "BALRAMCHIN", "JUBLFOOD", "KALYANKJIL", "TITAN", "TRENT", "PAGEIND", "METROBRAND"],
    "METALS": ["TATASTEEL", "JINDALSTEL", "JSWSTEEL", "HINDALCO", "VEDL", "NMDC", "NATIONALUM", "SAIL", "HINDZINC", "COALINDIA", "HINDCOPPER", "JSL"],
    "HEALTHCARE": ["SUNPHARMA", "CIPLA", "DRREDDY", "APOLLOHOSP", "DIVISLAB", "MAXHEALTH", "ZYDUSLIFE", "LUPIN", "AUROPHARMA", "ALKEM", "BIOCON", "GLENMARK", "GRANULES", "TORNTPHARM", "PPLPHARMA", "GLAND"],
    "DEFENSE/INFRA": ["LT", "ADANIENT", "ADANIPORTS", "AMBUJACEM", "ACC", "ULTRACEMCO", "BEL", "HAL", "BDL", "GRASIM", "CUMMINSIND", "ABB", "SIEMENS", "POLYCAB", "HAVELLS", "CONCOR", "MAZDOCK", "COCHINSHIP", "SOLARINDS"],
    "REALTY": ["DLF", "GODREJPROP", "LODHA", "PRESTIGE", "OBERREALTY", "IRB", "NBCC"],
    "PLATFORMS/MISC": ["ZOMATO", "SWIGGY", "NYKAA", "PAYTM", "PBSTECH", "DELHIVERY", "INDIAMART", "DIXON", "INDIGO", "IDEA", "MCX", "EXIDEIND", "ASTRAL"],
    "CHEMICALS": ["PIIND", "SRF", "DEEPAKNTR", "TATACHEM", "ATUL", "GUJGASLTD", "COROMANDEL"]
}

def run_scan():
    logger.info("Executing Full Multi-Table Scan...")
    scan_results = []
    open_status, reason = is_market_open()
    if not open_status:
        logger.info(f"Scan aborted: {reason}")
        return # Stops the script here

    logger.info("Starting Apex Sovereign Hourly Scan...")
    for sec, symbols in SECTOR_MAP.items():
        for s in symbols:
            try:
                d = yf.download(f"{s}.NS", period="1y", interval="1d", progress=False)
                if d.empty or len(d) < 150: continue
                
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                
                ind = calculate_indicators(d)
                
                # --- SCORING ---
                s1 = 0
                if ind['ltp'] > ind['ma20']: s1 += 10
                if ind['ltp'] > ind['ma50']: s1 += 5
                if ind['ltp'] > ind['ma200']: s1 += 5
                if ind['ma20'] > ind['ma50'] > ind['ma200']: s1 += 20
                
                s2 = 20 if (55 <= ind['rsi'] <= 70) else 10
                s3 = 20 if (ind['st_bull'] and ind['ltp'] > ind['vwap'] and ind['ltp'] > ind['pivot']) else 0
                s4 = 10 if (ind['adx'] > 30) else 0
                
                final_score = min(s1 + s2 + s3 + s4, 100)
                
                scan_results.append({
                    'Symbol': s, 'Sector': sec, 'LTP': round(ind['ltp'], 2), 'Change%': round(ind['chg_pct'],2),
                    'SCORE': final_score, 'RSI': round(ind['rsi'], 2),
                    'ADX': round(ind['adx'], 2), 'ST_Dir': "BULL" if ind['st_bull'] else "BEAR",
                    'VWAP': round(ind['vwap'],2),'Pivot': round(ind['pivot'],2),
                    'Above_Pivot': ind['ltp'] > ind['pivot'], 'Above_MA20': ind['ltp'] > ind['ma20']
                })
            except Exception as e:
                logger.debug(f"Error {s}: {e}")

    df = pd.DataFrame(scan_results)
    if not df.empty:
        # Table 1: Score 100 Wall
        score_100 = df[df['SCORE'] >= 85][['Symbol', 'Sector', 'LTP', 'Change%','SCORE','RSI','ADX','ST_Dir','VWAP','Pivot','Above_Pivot']]
        
        # Table 2: High Conviction (Strict Filters)
        high_conviction = df[
            (df['ADX'] > 30) & (df['RSI'] > 60) & 
            (df['Above_MA20'] == True) & (df['ST_Dir'] == "BULL") & 
            (df['Above_Pivot'] == True)
        ][['Symbol', 'Sector', 'LTP', 'Change%', 'SCORE','RSI','ADX','ST_Dir','VWAP','Pivot','Above_Pivot']]

        # Table 3: 
        # --- GET THE BEST STOCK ---
        # Sort by Score first, then by Change %
        best_stock = df.sort_values(by=['SCORE', 'Change%'], ascending=[False, False]).iloc[0]
        
        # Get Top 5 Runners up
        runners_up = df[df['Symbol'] != best_stock['Symbol']].sort_values(by='SCORE', ascending=False).head(5)
        
        # Table 4: 
        if not score_100.empty or not high_conviction.empty:
            send_email(score_100, high_conviction,best_stock,runners_up)

def send_email(df1, df2, df3, df4):
    try:
        sender, receiver, password = os.environ.get('EMAIL_SENDER'), os.environ.get('EMAIL_RECEIVER'), os.environ.get('EMAIL_PASSWORD')
        msg = MIMEMultipart()
        msg['Subject'] = f"🚀 APEX ALERTS: {pd.Timestamp.now('Asia/Kolkata').strftime('%Y-%m-%d %H:%M')}"
        
        html = f"""
        <html>
          <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #FF4B4B;">🏆 Top SCORE Symbols</h2>
            {df1.to_html(index=False) if not df1.empty else "<p>No Score 100 stocks.</p>"}
            <hr>
            <h2 style="color: #00CC66;">🚨 HIGH CONVICTION ALERTS</h2>
            {df2.to_html(index=False) if not df2.empty else "<p>No High Conviction alerts.</p>"}
            <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h2 style="margin: 0;">Sovereign Winner</h2>
                <h1 style="font-size: 40px; margin: 10px 0;">{df3['Symbol']}</h1>
                <p style="font-size: 18px;">Score: <b>{df3['SCORE']}</b> | LTP: ₹{df3['LTP']} ({df3['Change%']}%)</p>
            </div>
            
            <h3 style="margin-top: 25px;">📈 Top Runners Up</h3>
            <div style="width: 100%; overflow-x: auto;">
                {df4.to_html(index=False, border=0) if not df4.empty else "<p>No runners up.</p>"}
            </div>
          </body>
        </html>
        """
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        logger.info("Email sent!")
    except Exception as e:
        logger.error(f"Failed: {e}")

if __name__ == "__main__":
    run_scan()

