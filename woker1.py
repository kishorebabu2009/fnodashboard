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

# --- CONFIG & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- NSE HOLIDAY LIST 2026 ---
NSE_HOLIDAYS = ["2026-01-26", "2026-03-06", "2026-03-27", "2026-04-02", "2026-04-03", 
                "2026-04-14", "2026-05-01", "2026-08-15", "2026-10-02", "2026-10-21", 
                "2026-11-09", "2026-12-25"]

# --- SECTOR MAP ---
SECTOR_MAP = {
    "BANKING": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "AUTO": ["TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT"],
    "ENERGY": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL", "ADANIGREEN"],
    "FINANCE": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "RECLTD", "PFC", "IRFC"],
    "METALS": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "COALINDIA", "VEDL"]
}

# --- HELPER FUNCTIONS ---

def get_market_sentiment():
    """Checks NIFTY 50 status to confirm market trend"""
    nifty = yf.download("^NSEI", period="2d", interval="15m", progress=False)
    if nifty.empty: return "NEUTRAL"
    
    last_price = nifty['Close'].iloc[-1]
    # Simple logic: If price is above the opening price of the day
    day_open = nifty.iloc[0]['Open']
    
    if last_price > day_open * 1.002: # 0.2% buffer
        return "BULLISH"
    elif last_price < day_open * 0.998:
        return "BEARISH"
    else:
        return "SIDEWAYS"

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return {k: 0 for k in ['delta', 'gamma', 'theta', 'vega']}
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
    ticker = yf.Ticker(f"{symbol}.NS")
    try:
        expirations = ticker.options
        if not expirations: return None
        expiry = expirations[0]
        opt = ticker.option_chain(expiry)
        ltp = ticker.history(period="1d")['Close'].iloc[-1]
        
        # Determine ATM Strike
        chain = opt.calls if side == "CALL" else opt.puts
        atm_idx = (chain['strike'] - ltp).abs().idxmin()
        strike = chain.loc[atm_idx, 'strike']
        iv = chain.loc[atm_idx, 'impliedVolatility']
        
        # Straddle value for context
        straddle = chain.loc[atm_idx, 'lastPrice'] + opt.puts[opt.puts['strike'] == strike]['lastPrice'].iloc[0]
        
        T = max((datetime.datetime.strptime(expiry, '%Y-%m-%d') - datetime.datetime.now()).days, 1) / 365.0
        greeks = calculate_greeks(ltp, strike, T, 0.07, iv, option_type=side.lower())
        return {'Strike': strike, 'Expiry': expiry, 'IV': f"{round(iv*100, 2)}%", 'Straddle': round(straddle, 2), **greeks}
    except: return None

def calculate_indicators(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    delta = c.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, min_periods=14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    vwap = (((h + l + c) / 3) * v).rolling(14).sum() / v.rolling(14).sum()
    ma20, ma50 = c.rolling(20).mean(), c.rolling(50).mean()
    
    return {
        'ltp': c.iloc[-1], 'rsi': rsi.iloc[-1], 'atr': atr.iloc[-1], 
        'vwap': vwap.iloc[-1], 'ma20': ma20.iloc[-1], 'ma50': ma50.iloc[-1],
        'vol_ratio': v.iloc[-1] / v.rolling(20).mean().iloc[-1],
        'chg_pct': ((c.iloc[-1] - c.iloc[-2])/c.iloc[-2])*100
    }

def run_scan():
    # 1. Market Trend Check
    sentiment = get_market_sentiment()
    logger.info(f"Market Sentiment: {sentiment}")

    all_syms = [f"{s}.NS" for sub in SECTOR_MAP.values() for s in sub]
    data = yf.download(all_syms, period="1y", group_by='ticker', progress=False)
    
    results = []
    for sector, symbols in SECTOR_MAP.items():
        for s in symbols:
            try:
                df = data[f"{s}.NS"].dropna()
                ind = calculate_indicators(df)
                
                # Scoring Logic
                score = 0
                if ind['ltp'] > ind['ma20']: score += 20
                if ind['rsi'] > 50: score += 20
                if ind['vol_ratio'] > 1.1: score += 20
                if ind['ltp'] > ind['vwap']: score += 20
                if ind['ma20'] > ind['ma50']: score += 20

                # Trend Confirmation Filter
                trade_type = "NONE"
                if sentiment == "BULLISH" and score >= 75: trade_type = "CALL"
                elif sentiment == "BEARISH" and score <= 35: trade_type = "PUT"

                results.append({
                    'Symbol': s, 'Sector': sector, 'LTP': round(ind['ltp'], 2), 
                    'Score': score, 'RSI': round(ind['rsi'], 1), 'Vol': f"{round(ind['vol_ratio'], 1)}x",
                    'Type': trade_type, 'SL': round(ind['ltp'] - (1.5 * ind['atr']), 2)
                })
            except: continue

    res_df = pd.DataFrame(results)
    trades = res_df[res_df['Type'] != "NONE"]
    
    if not trades.empty:
        best = trades.sort_values(by='Score', ascending=(sentiment == "BULLISH")).iloc[0]
        greeks = get_option_analytics(best['Symbol'], side=best['Type'])
        send_email(trades, best, greeks, sentiment)

def send_email(top_df, winner, greeks):
    try:
        sender = os.environ.get('EMAIL_SENDER')
        pwd = os.environ.get('EMAIL_PASSWORD')
        receiver = os.environ.get('EMAIL_RECEIVER').split(',')
        
        msg = MIMEMultipart()
        msg['Subject'] = f"🚀 APEX OPTIONS: {winner['Symbol']} @ {winner['Score']}pts"
        
        greek_html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-left: 5px solid #007bff;">
            <h3>💎 ATM Option Greeks: {winner['Symbol']}</h3>
            <table border="1" style="border-collapse: collapse; width: 100%; text-align: center;">
                <tr style="background-color: #007bff; color: white;">
                    <th>Strike</th><th>Expiry</th><th>IV</th><th>Straddle</th><th>Delta</th><th>Theta</th><th>Vega</th>
                </tr>
                <tr>
                    <td>{greeks['Strike']}</td><td>{greeks['Expiry']}</td><td>{greeks['IV']}</td>
                    <td>₹{greeks['Straddle']}</td><td>{greeks['Delta']}</td><td>{greeks['Theta']}</td><td>{greeks['Vega']}</td>
                </tr>
            </table>
        </div>
        """ if greeks else "<p>Greeks Unavailable for this symbol.</p>"

        html = f"""
        <html><body>
            <h2 style="color: #28a745;">🏆 Sovereign Winner: {winner['Symbol']}</h2>
            <p>LTP: <b>₹{winner['LTP']}</b> | Exit SL: <b>₹{winner['StopLoss']}</b> | Vol: <b>{winner['Vol']}</b></p>
            {greek_html}
            <br>
            <h3>📈 Market High Scorers (>= 80)</h3>
            {top_df.to_html(index=False, classes='table')}
        </body></html>
        """
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, pwd)
            server.sendmail(sender, receiver, msg.as_string())
        logger.info("Email sent successfully.")
    except Exception as e: logger.error(f"Email failed: {e}")

if __name__ == "__main__":
    run_scan()
