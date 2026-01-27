import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def calculate_indicators(df):
    """Manual technical indicator math (No-Dependency Mode)"""
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    
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
        'st_bull': st_dir[-1], 'pivot': pivot, 'ltp': c.iloc[-1]
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
                if ind['ltp'] > ind['ma50']: s1 += 10
                if ind['ltp'] > ind['ma200']: s1 += 10
                if ind['ma20'] > ind['ma50'] > ind['ma200']: s1 += 10
                
                s2 = 30 if (55 <= ind['rsi'] <= 70) else 15
                s3 = 20 if (ind['st_bull'] and ind['ltp'] > ind['vwap']) else 0
                s4 = 10 if (ind['adx'] > 25) else 0
                
                final_score = min(s1 + s2 + s3 + s4, 100)
                
                scan_results.append({
                    'Symbol': s, 'Sector': sec, 'LTP': round(ind['ltp'], 2),
                    'SCORE': final_score, 'RSI': round(ind['rsi'], 2),
                    'ADX': round(ind['adx'], 2), 'ST_Dir': "BULL" if ind['st_bull'] else "BEAR",
                    'Above_Pivot': ind['ltp'] > ind['pivot'], 'Above_MA20': ind['ltp'] > ind['ma20']
                })
            except Exception as e:
                logger.debug(f"Error {s}: {e}")

    df = pd.DataFrame(scan_results)
    if not df.empty:
        # Table 1: Score 100 Wall
        score_100 = df[df['SCORE'] >= 100][['Symbol', 'Sector', 'LTP', 'SCORE']]
        
        # Table 2: High Conviction (Strict Filters)
        high_conviction = df[
            (df['ADX'] > 25) & (df['RSI'] > 50) & 
            (df['Above_MA20'] == True) & (df['ST_Dir'] == "BULL") & 
            (df['Above_Pivot'] == True)
        ][['Symbol', 'LTP', 'RSI', 'ADX']]
        
        if not score_100.empty or not high_conviction.empty:
            send_email(score_100, high_conviction)

def send_email(df1, df2):
    try:
        sender, receiver, password = os.environ.get('EMAIL_SENDER'), os.environ.get('EMAIL_RECEIVER'), os.environ.get('EMAIL_PASSWORD')
        msg = MIMEMultipart()
        msg['Subject'] = f"🚀 APEX ALERTS: {pd.Timestamp.now('Asia/Kolkata').strftime('%Y-%m-%d %H:%M')}"
        
        html = f"""
        <html>
          <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #FF4B4B;">🏆 SCORE 100 WALL</h2>
            {df1.to_html(index=False) if not df1.empty else "<p>No Score 100 stocks.</p>"}
            <hr>
            <h2 style="color: #00CC66;">🚨 HIGH CONVICTION ALERTS</h2>
            {df2.to_html(index=False) if not df2.empty else "<p>No High Conviction alerts.</p>"}
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

