import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import smtplib
import os
import logging
import datetime
import pytz
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- NSE HOLIDAY LIST 2026 ---
NSE_HOLIDAYS = ["2026-01-26", "2026-03-06", "2026-03-27", "2026-04-02", "2026-04-03", 
                "2026-04-14", "2026-05-01", "2026-08-15", "2026-10-02", "2026-10-21", 
                "2026-11-09", "2026-12-25"]

# --- SECTOR MAP ---
SECTOR_MAP = {
    "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "PNB"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "AUTO": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR"],
    "ENERGY": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL", "TATAPOWER"],
    "FINANCE": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "RECLTD", "PFC", "MUTHOOTFIN"],
    "METALS": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "COALINDIA", "VEDL"],
    "MISC": ["LT", "TITAN", "ADANIENT", "BHARTIARTL", "SUNPHARMA", "ZOMATO"]
}

# --- CORE LOGIC FUNCTIONS ---

def is_market_open():
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(tz)
    if now.weekday() >= 5: return False, "Weekend"
    if now.strftime('%Y-%m-%d') in NSE_HOLIDAYS: return False, "NSE Holiday"
    return True, "Market Open"

def get_market_sentiment():
    """Fetches NIFTY 50 to confirm overall market trend"""
    try:
        nifty = yf.download("^NSEI", period="2d", interval="15m", progress=False)
        if nifty.empty: return "SIDEWAYS"
        last_price = nifty['Close'].iloc[-1]
        day_open = nifty['Open'].iloc[0]
        if last_price > day_open * 1.002: return "BULLISH"
        elif last_price < day_open * 0.998: return "BEARISH"
        return "SIDEWAYS"
    except: return "SIDEWAYS"

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes Greek Engine"""
    if T <= 0 or sigma <= 0: return {k: 0 for k in ['Delta', 'Gamma', 'Theta', 'Vega']}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = si.norm.cdf(d1) if option_type == 'call' else si.norm.cdf(d1) - 1
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = (S * si.norm.pdf(d1) * np.sqrt(T)) / 100
    theta_term1 = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        theta = (theta_term1 - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365
    else:
        theta = (theta_term1 + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365
    return {'Delta': round(delta, 3), 'Gamma': round(gamma, 4), 'Theta': round(theta, 2), 'Vega': round(vega, 2)}

def get_option_analytics(symbol, side="CALL"):
    """ATM Greeks & Straddle Data"""
    ticker = yf.Ticker(f"{symbol}.NS")
    try:
        expirations = ticker.options
        if not expirations: return None
        expiry = expirations[0]
        opt = ticker.option_chain(expiry)
        ltp = ticker.history(period="1d")['Close'].iloc[-1]
        
        chain = opt.calls if side == "CALL" else opt.puts
        atm_idx = (chain['strike'] - ltp).abs().idxmin()
        strike = chain.loc[atm_idx, 'strike']
        iv = chain.loc[atm_idx, 'impliedVolatility']
        
        c_price = opt.calls.loc[atm_idx, 'lastPrice']
        p_price = opt.puts[opt.puts['strike'] == strike]['lastPrice'].iloc[0]
        
        T = max((datetime.datetime.strptime(expiry, '%Y-%m-%d') - datetime.datetime.now()).days, 1) / 365.0
        greeks = calculate_greeks(ltp, strike, T, 0.07, iv, option_type=side.lower())
        
        return {
            'Strike': strike, 'Expiry': expiry, 'IV': f"{round(iv*100, 1)}%", 
            'Straddle': round(c_price + p_price, 2), **greeks
        }
    except: return None

def calculate_indicators(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    # Wilder's RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    vwap = (((h + l + c) / 3) * v).rolling(14).sum() / v.rolling(14).sum()
    ma20, ma50, ma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    
    return {
        'ltp': c.iloc[-1], 'rsi': rsi.iloc[-1], 'atr': atr.iloc[-1], 'vwap': vwap.iloc[-1],
        'ma20': ma20.iloc[-1], 'ma50': ma50.iloc[-1], 'ma200': ma200.iloc[-1],
        'vol_ratio': v.iloc[-1] / v.rolling(20).mean().iloc[-1],
        'chg_pct': ((c.iloc[-1] - c.iloc[-2])/c.iloc[-2])*100,
        'pivot': (h.iloc[-2] + l.iloc[-2] + c.iloc[-2]) / 3
    }

def run_scan():
    is_open, _ = is_market_open()
    if not is_open: return 
    
    sentiment = get_market_sentiment()
    all_syms = [f"{s}.NS" for sub in SECTOR_MAP.values() for s in sub]
    data = yf.download(all_syms, period="1y", group_by='ticker', progress=False)
    
    results = []
    for sector, symbols in SECTOR_MAP.items():
        for s in symbols:
            try:
                df = data[f"{s}.NS"].dropna()
                if len(df) < 150: continue
                ind = calculate_indicators(df)
                
                score = 0
                if ind['ltp'] > ind['ma20']: score += 20
                if 55 <= ind['rsi'] <= 70: score += 20
                if ind['vol_ratio'] > 1.2: score += 20
                if ind['ltp'] > ind['vwap']: score += 20
                if ind['ltp'] > ind['pivot']: score += 20

                # Trend Filter
                trade_type = "NONE"
                if sentiment == "BULLISH" and score >= 75: trade_type = "CALL"
                elif sentiment == "BEARISH" and score <= 35: trade_type = "PUT"

                results.append({
                    'Symbol': s, 'Sector': sector, 'LTP': round(ind['ltp'], 2), 
                    'Score': score, 'RSI': round(ind['rsi'], 1), 'Vol': f"{round(ind['vol_ratio'], 1)}x",
                    'Trade': trade_type, 'StopLoss': round(ind['ltp'] - (1.5 * ind['atr']), 2),
                    'Chg%': round(ind['chg_pct'], 2)
                })
            except: continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # High Conviction Table
        high_conv = res_df[res_df['Score'] >= 80].sort_values(by='Score', ascending=False)
        # Winner for Greeks
        best = res_df.sort_values(by='Score', ascending=False).iloc[0]
        greeks = get_option_analytics(best['Symbol'], side="CALL" if best['Score'] > 50 else "PUT")
        send_email(high_conv, best, greeks, sentiment)

def send_email(top_df, winner, greeks, sentiment):
    try:
        sender, pwd = os.environ.get('EMAIL_SENDER'), os.environ.get('EMAIL_PASSWORD')
        receiver = os.environ.get('EMAIL_RECEIVER').split(',')
        msg = MIMEMultipart()
        msg['Subject'] = f"🚀 APEX {sentiment}: {winner['Symbol']} ({winner['Score']} pts)"
        
        greek_table = f"""
        <table border="1" style="width:100%; text-align:center; border-collapse:collapse; background-color:#f4f4f4;">
            <tr style="background-color:#333; color:white;">
                <th>Strike</th><th>Expiry</th><th>IV</th><th>Delta</th><th>Theta</th><th>Vega</th><th>Straddle</th>
            </tr>
            <tr>
                <td>{greeks['Strike']}</td><td>{greeks['Expiry']}</td><td>{greeks['IV']}</td>
                <td>{greeks['Delta']}</td><td>{greeks['Theta']}</td><td>{greeks['Vega']}</td><td>₹{greeks['Straddle']}</td>
            </tr>
        </table>""" if greeks else "<p>Greeks Unavailable</p>"

        html = f"""
        <html><body style="font-family: Arial;">
            <h2 style="color:#2E86C1;">Market Sentiment: {sentiment}</h2>
            <div style="border: 2px solid #2E86C1; padding: 15px; border-radius: 10px;">
                <h3>👑 Sovereign Winner: {winner['Symbol']}</h3>
                <p>LTP: <b>₹{winner['LTP']}</b> | Stop Loss: <b>₹{winner['StopLoss']}</b> | Vol: <b>{winner['Vol']}</b></p>
                {greek_table}
            </div>
            <br>
            <h3>🔥 High Score Watchlist</h3>
            {top_df.to_html(index=False, border=1)}
        </body></html>
        """
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(sender, pwd)
            s.sendmail(sender, receiver, msg.as_string())
        logger.info("Email sent.")
    except Exception as e: logger.error(f"Failed: {e}")

if __name__ == "__main__":
    run_scan()
