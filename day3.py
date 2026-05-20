"""
Day 3: Stationarity and autocorrelation.
Question: Is there exploitable structure, or is crypto a random walk?
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

exchange = ccxt.binance()


def fetch_history(symbol, days):
    all_ohlcv = []
    since = exchange.parse8601(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    )
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=1000)
        if not chunk:
            break
        all_ohlcv += chunk
        since = chunk[-1][0] + 86400_000
        if len(chunk) < 1000:
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').set_index('timestamp')
    return df


def adf_test(series, name):
    """Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    print(f"\n--- ADF Test: {name} ---")
    print(f"ADF Statistic:  {result[0]:.4f}")
    print(f"p-value:        {result[1]:.4f}")
    print(f"Critical values: 1%={result[4]['1%']:.2f}, 5%={result[4]['5%']:.2f}")
    if result[1] < 0.05:
        print("=> p < 0.05: STATIONARY (reject random walk / unit root)")
    else:
        print("=> p > 0.05: NON-STATIONARY (likely has a trend / unit root)")


# --- Fetch BTC ---
df = fetch_history('BTC/USDT', days=1825)
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# --- 1. Stationarity tests ---
print("=" * 60)
print("  STATIONARITY TESTS")
print("=" * 60)
adf_test(df['close'], "BTC close PRICE")        # expect non-stationary
adf_test(df['log_return'], "BTC log RETURNS")   # expect stationary

# --- 2. Autocorrelation of returns ---
print("\n" + "=" * 60)
print("  AUTOCORRELATION OF RETURNS")
print("=" * 60)
returns = df['log_return'].dropna()
acf_values = acf(returns, nlags=10)
print("\nLag | Autocorrelation")
for lag, val in enumerate(acf_values):
    marker = "  <-- notable" if abs(val) > 0.05 and lag > 0 else ""
    print(f"{lag:3d} | {val:+.4f}{marker}")

# Rough significance threshold: ±1.96/sqrt(N)
threshold = 1.96 / np.sqrt(len(returns))
print(f"\nSignificance threshold (95%): +/-{threshold:.4f}")
print("Autocorrelations beyond this are statistically notable.")

# --- 3. Autocorrelation of ABSOLUTE returns (volatility clustering) ---
print("\n" + "=" * 60)
print("  VOLATILITY CLUSTERING (autocorr of |returns|)")
print("=" * 60)
abs_acf = acf(returns.abs(), nlags=10)
print("\nLag | Autocorrelation of |returns|")
for lag, val in enumerate(abs_acf):
    marker = "  <-- strong" if val > 0.1 and lag > 0 else ""
    print(f"{lag:3d} | {val:+.4f}{marker}")

# --- 4. Plots ---
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price (non-stationary - trends)
axes[0, 0].plot(df.index, df['close'], color='orange')
axes[0, 0].set_title('BTC Price (non-stationary - drifts over time)')
axes[0, 0].grid(True, alpha=0.3)

# Returns (stationary - hovers around 0)
axes[0, 1].plot(df.index, df['log_return'], color='steelblue', linewidth=0.6)
axes[0, 1].axhline(0, color='black', linewidth=0.5)
axes[0, 1].set_title('BTC Returns (stationary - mean ~0)')
axes[0, 1].grid(True, alpha=0.3)

# ACF of returns
plot_acf(returns, lags=30, ax=axes[1, 0])
axes[1, 0].set_title('ACF of Returns (is there momentum/reversion?)')

# ACF of absolute returns
plot_acf(returns.abs(), lags=30, ax=axes[1, 1])
axes[1, 1].set_title('ACF of |Returns| (volatility clustering)')

plt.tight_layout()
plt.savefig('notes/day3_stationarity.png', dpi=100)
plt.show()

print("\nCharts saved to notes/day3_stationarity.png")