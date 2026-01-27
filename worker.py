import yfinance as yf
import pandas as pd
import pandas_ta as ta
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Use the exact SECTOR_MAP from your original code
SECTOR_MAP = {
    "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK"],
    # ... (Include all other sectors from your file)
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
                    vwap = ((((h + l + c) / 3) * v).rolling(14).sum() / v.rolling(14).sum())
                    st_df = ta.supertrend(h, l, c, 7, 3)
                    is_bull_st = (st_df is not None and st_df.iloc[-1, 1] > 0)

                    # Scoring Logic
                    # ... (Include your full scoring logic here to calculate 'final_score')
                    
                    results.append({
                        'Symbol': s, 'SCORE': final_score, 'LTP': c.iloc[-1],
                        'RSI': rsi.iloc[-1], 'MA20': ma20.iloc[-1], 
                        'MA50': ma50.iloc[-1], 'MA200': ma200.iloc[-1],
                        'VWAP': vwap.iloc[-1], 'ST_Dir': "BULL" if is_bull_st else "BEAR"
                    })
            except: continue

    df = pd.DataFrame(results)
    
    # Filter for your specific alerts
    score_100 = df[df['SCORE'] >= 100][['Symbol', 'LTP', 'SCORE']]
    high_conviction = df[(df['RSI'] > 50) & (df['LTP'] > df['MA20']) & (df['MA50'] > df['MA200']) & (df['LTP'] > df['VWAP']) & (df['ST_Dir'] == "BULL")]
    
    if not score_100.empty or not high_conviction.empty:
        send_email(score_100, high_conviction)

def send_email(df1, df2):
    sender = os.environ.get('EMAIL_SENDER')
    receiver = os.environ.get('EMAIL_RECEIVER')
    password = os.environ.get('ggvm pabz fheh jpyq') # Gmail App Password

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
