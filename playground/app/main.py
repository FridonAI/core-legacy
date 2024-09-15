# main.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import aiohttp
import asyncio
import os
from datetime import datetime, timedelta

app = FastAPI()

# Ensure the 'static' directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

async def fetch_data(interval):
    # Map interval to Binance API intervals
    if interval == '4h':
        binance_interval = '4h'
        start_time = int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)
    else:  # '1d'
        binance_interval = '1d'
        start_time = int((datetime.utcnow() - timedelta(days=30)).timestamp() * 1000)
    
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={binance_interval}&startTime={start_time}"
    
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
            pivots.append((df['Date'][i], current_max))
        if current_min == min_price:
            pivots.append((df['Date'][i], current_min))

    # Filter significant levels
    levels = []
    for pivot in pivots:
        level = pivot[1]
        if any([abs(level - x) < (df['High'].max() - df['Low'].min()) * 0.005 for x in levels]):
            continue
        levels.append(level)
    return levels

def create_chart(df, levels, interval):
    # Create candlestick chart with Plotly
    candlestick = go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )

    volume = go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.5
    )

    data = [candlestick, volume]

    # Add support and resistance lines
    shapes = []
    annotations = []
    for level in levels:
        shapes.append({
            'type': 'line',
            'x0': df['Date'].min(),
            'y0': level,
            'x1': df['Date'].max(),
            'y1': level,
            'line': {
                'color': 'green',
                'width': 1,
                'dash': 'dash',
            },
        })
        annotations.append({
            'x': df['Date'].max(),
            'y': level,
            'xref': 'x',
            'yref': 'y',
            'text': f'{level:.2f}',
            'showarrow': False,
            'xanchor': 'left',
            'yanchor': 'middle',
            'font': {'color': 'green'},
        })

    layout = go.Layout(
        title=f'Bitcoin Price Chart ({interval.upper()})',
        xaxis=dict(
            rangeslider=dict(visible=False),
            title='Date'
        ),
        yaxis=dict(
            title='Price (USD)',
            domain=[0.3, 1]
        ),
        yaxis2=dict(
            title='Volume',
            domain=[0, 0.2],
            anchor='x'
        ),
        shapes=shapes,
        annotations=annotations,
        legend=dict(orientation='h', x=0, y=1.1),
        height=600,
    )

    fig = go.Figure(data=data, layout=layout)

    # Save the chart as an HTML file
    filename = f"chart_{interval}.html"
    filepath = os.path.join('static', filename)
    pyo.plot(fig, filename=filepath, auto_open=False, include_plotlyjs='cdn')

    return filename

@app.get("/chart")
async def get_chart(interval: str = '1d'):
    if interval not in ['4h', '1d']:
        return JSONResponse(content={"error": "Invalid interval"}, status_code=400)

    df = await fetch_data(interval)
    levels = calculate_support_resistance(df)
    filename = create_chart(df, levels, interval)

    # Return the URL to the chart
    url = f"/static/{filename}"
    return {"chart_url": url}

@app.get("/")
async def read_index():
    with open("index.html", 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
