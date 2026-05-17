"""
Day 1: Pull BTC daily data from Binance and explore basics.
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize Binance (public data, no API key needed)
exchange = ccxt.binance()

# Fetch last 365 days of BTC/USDT daily candles
ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=365)

# Convert to DataFrame
df = pd.DataFrame(
    ohlcv,
    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# --- Basic exploration ---
print("=== Last 5 days of BTC/USDT ===")
print(df.tail())

print("\n=== Summary statistics ===")
print(df['close'].describe())

# --- Calculate returns ---
df['daily_return'] = df['close'].pct_change()
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

print("\n=== Return statistics ===")
print(f"1-year return:        {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")
print(f"Average daily return: {df['daily_return'].mean() * 100:.3f}%")
print(f"Daily volatility:     {df['daily_return'].std() * 100:.2f}%")
print(f"Annualized vol:       {df['daily_return'].std() * np.sqrt(365) * 100:.2f}%")
print(f"Best day:             {df['daily_return'].max() * 100:.2f}%")
print(f"Worst day:            {df['daily_return'].min() * 100:.2f}%")

# --- Plot price and returns ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(df.index, df['close'], color='orange', linewidth=1.5)
axes[0].set_title('BTC/USDT — Daily Close (Last 365 days)')
axes[0].set_ylabel('Price (USDT)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index, df['daily_return'] * 100, color='steelblue', linewidth=0.8)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_title('BTC/USDT — Daily Returns (%)')
axes[1].set_ylabel('Return (%)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notes/day1_btc_chart.png', dpi=100)
plt.show()

print("\nChart saved to notes/day1_btc_chart.png")