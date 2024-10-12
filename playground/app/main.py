# main.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import aiohttp
from datetime import datetime, timedelta

app = FastAPI()

async def fetch_data(symbol, interval, days=None, limit=1):
    # Map interval to Binance API intervals
    binance_interval = interval
    if days is not None:
        start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binance_interval}&startTime={start_time}"
    else:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binance_interval}&limit={limit}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
    
    # Convert data to DataFrame
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
               'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
               'Taker buy quote asset volume', 'Ignore']
    df = pd.DataFrame(data, columns=columns)
    df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def calculate_support_resistance(df):
    # Identify pivot points
    pivots = []
    for i in range(2, df.shape[0] - 2):
        max_price = max(df['High'][i - 2:i + 3])
        min_price = min(df['Low'][i - 2:i + 3])
        current_max = df['High'][i]
        current_min = df['Low'][i]
        if current_max == max_price:
            pivots.append({'Date': df['Date'][i], 'Value': current_max, 'Type': 'Resistance'})
        if current_min == min_price:
            pivots.append({'Date': df['Date'][i], 'Value': current_min, 'Type': 'Support'})
    
    # Filter significant levels
    levels = []
    for pivot in pivots:
        level = pivot['Value']
        if any([abs(level - x['Value']) < (df['High'].max() - df['Low'].min()) * 0.005 for x in levels]):
            continue
        levels.append(pivot)
    return levels

def calculate_indicators(df):
    # Calculate EMA (e.g., 20-period EMA)
    ema_period = 20
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Calculate VWAP
    df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['Typical Price'] * df['Volume']
    df['Cumulative TPV'] = df['TPV'].cumsum()
    df['Cumulative Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative TPV'] / df['Cumulative Volume']
    
    # Clean up intermediate columns
    df.drop(['Typical Price', 'TPV', 'Cumulative TPV', 'Cumulative Volume'], axis=1, inplace=True)
    
    return df

@app.get("/chart-data")
async def get_chart_data(interval: str = '1d'):
    if interval not in ['4h', '1d']:
        return JSONResponse(content={"error": "Invalid interval"}, status_code=400)

    df = await fetch_data(interval)
    df = calculate_indicators(df)
    levels = calculate_support_resistance(df)

    # Convert 'Date' fields in 'levels' to strings
    for level in levels:
        level['Date'] = str(level['Date'])

    # Prepare data for Plotly
    data = {
        'dates': df['Date'].astype(str).tolist(),
        'open': df['Open'].tolist(),
        'high': df['High'].tolist(),
        'low': df['Low'].tolist(),
        'close': df['Close'].tolist(),
        'volume': df['Volume'].tolist(),
        'ema': df['EMA'].tolist(),
        'vwap': df['VWAP'].tolist(),
        'levels': levels
    }

    return JSONResponse(content=data)

@app.get("/")
async def read_index():
    with open("index.html", 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
