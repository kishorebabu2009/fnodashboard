import yfinance as yf
import pandas as pd
import pandas_ta as ta
import smtplib
import os
import logging  # Added missing import
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

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
    logger.info("Starting Apex Sovereign Hourly Scan...")
    results = []
    for sec, symbols in SECTOR_MAP.items():
        for s in symbols:
            try:
                d = yf.download(f"{s}.NS", period="1y", interval="1d", progress=False)
                if not d.empty and len(d) > 100:
                    # Fix for MultiIndex columns in newer yfinance versions
                    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                    
                    c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume']
                    
                    # Indicators
                    ma20, ma50, ma200 = ta.sma(c, 20), ta.sma(c, 50), ta.sma(c, 200)
                    rsi = ta.rsi(c, 14)
                    atr = ta.atr(h, l, c, 14)
                    vwap = (((h + l + c) / 3) * v).rolling(14).sum() / v.rolling(14).sum()
                    adx = ta.adx(h, l, c)
                    st_df = ta.supertrend(h, l, c, 7, 3)
                    
                    # CPR Logic
                    ph, pl, pc = h.iloc[-2], l.iloc[-2], c.iloc[-2]
                    pivot = (ph + pl + pc) / 3
                    bc = (ph + pl) / 2
                    tc = (pivot - bc) + pivot
                    
                    # Scoring
                    curr_c = float(c.iloc[-1])
                    s1 = 0
                    if curr_c > ma20.iloc[-1]: s1 += 10
                    if curr_c > ma50.iloc[-1]: s1 += 10
                    if curr_c > ma200.iloc[-1]: s1 += 10
                    if ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]: s1 += 10
                    
                    s2 = 0
                    curr_rsi = rsi.iloc[-1]
                    if 55 <= curr_rsi <= 70: s2 = 30
                    elif curr_rsi > 75: s2 = 15
                    elif 45 <= curr_rsi < 55: s2 = 15
                    
                    s3 = 0
                    curr_adx = adx.iloc[-1, 0]
                    is_bull_st = (st_df is not None and st_df.iloc[-1, 1] > 0)
                    if curr_c > vwap.iloc[-1]: s3 += 5
                    if is_bull_st: s3 += 5
                    if curr_adx > 25: s3 += 15
                    
                    vol_ma20 = v.rolling(20).mean().iloc[-1]
                    s4 = 10 if v.iloc[-1] > (vol_ma20 * 1.5) else 0

                    final_score = min(s1 + s2 + s3 + s4, 100)
                    contrib_msg = f"Trend:{s1}|RSI:{s2}|Intensity:{s3}|Vol:{s4}"
                    
                    results.append({
                        'Symbol': s, 'Sector': sec, 'SCORE': final_score, 'LTP': round(curr_c, 2),
                        'CHG': round(((c.iloc[-1]/c.iloc[-2])-1)*100,2), 'RSI': round(curr_rsi,2), 
                        'ADX': round(curr_adx,2), 'ST_Dir': "BULL" if is_bull_st else "BEAR",
                        'Pivot': round(pivot,2), 'VWAP': round(vwap.iloc[-1],2), 'CONTRIB': contrib_msg
                    })
            except Exception as e:
                logger.error(f"Error scanning {s}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        score_100 = df[df['SCORE'] >= 100]
        high_conviction = df[(df['ADX'] > 30) & (df['RSI'] > 55) & (df['ST_Dir'] == "BULL")]
        
        if not score_100.empty or not high_conviction.empty:
            logger.info(f"Criteria met. Score 100: {len(score_100)}, High Conviction: {len(high_conviction)}")
            send_email(score_100, high_conviction)
        else:
            logger.info("No stocks met high-score criteria this hour.")

def send_email(df1, df2):
    try:
        sender = os.environ.get('EMAIL_SENDER')
        receiver = os.environ.get('EMAIL_RECEIVER')
        password = os.environ.get('EMAIL_PASSWORD')

        msg = MIMEMultipart()
        msg['Subject'] = f"🚀 APEX ALERTS: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
        
        html = f"""
        <html>
          <body>
            <h2 style="color: #FF4B4B;">🏆 SCORE 100 WALL</h2>
            {df1[['Symbol', 'Sector', 'LTP', 'SCORE', 'CONTRIB']].to_html(index=False)}
            <hr>
            <h2 style="color: #00CC66;">🚨 HIGH CONVICTION ALERTS</h2>
            {df2[['Symbol', 'LTP', 'RSI', 'ST_Dir']].to_html(index=False)}
          </body>
        </html>
        """
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        logger.info("Email sent successfully!")
    except Exception as e:
        logger.error(f"Email failed: {e}")

if __name__ == "__main__":
    run_scan()
