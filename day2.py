"""
Day 2: Return distributions, fat tails, and multi-year comparison.
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

exchange = ccxt.binance()

def fetch_history(symbol, days):
    """Fetch up to `days` of daily candles by paginating backwards."""
    all_ohlcv = []
    since = exchange.parse8601(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    )
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=1000)
        if not chunk:
            break
        all_ohlcv += chunk
        since = chunk[-1][0] + 86400_000  # next day in ms
        if len(chunk) < 1000:
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').set_index('timestamp')
    return df


def analyze(symbol, days=1825):  # ~5 years
    print(f"\n{'=' * 60}")
    print(f"  {symbol}  —  last {days} days")
    print('=' * 60)

    df = fetch_history(symbol, days)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    returns = df['log_return'].dropna()

    # --- Stats ---
    mean = returns.mean()
    std = returns.std()
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # excess kurtosis (normal = 0)

    print(f"Sample size:        {len(returns)} days")
    print(f"Mean daily return:  {mean * 100:.3f}%")
    print(f"Daily volatility:   {std * 100:.2f}%")
    print(f"Annualized vol:     {std * np.sqrt(365) * 100:.2f}%")
    print(f"Skewness:           {skew:.3f}")
    print(f"Excess kurtosis:    {kurt:.3f}   (normal = 0)")

    # --- Expected vs actual extreme days ---
    # Under normal distribution, P(|return| > 3*std) ≈ 0.27%
    threshold = 3 * std
    actual_extreme = (returns.abs() > threshold).sum()
    expected_extreme = len(returns) * 0.0027
    print(f"\nExpected 3-sigma days (normal): {expected_extreme:.1f}")
    print(f"Actual 3-sigma days:            {actual_extreme}")
    print(f"Ratio:                          {actual_extreme / max(expected_extreme, 0.01):.1f}x more than normal predicts")

    return df, returns


def plot_distribution(returns, symbol, ax):
    """Histogram of returns vs fitted normal distribution."""
    returns_pct = returns * 100
    ax.hist(returns_pct, bins=60, density=True, alpha=0.6, color='steelblue', label='Actual returns')

    # Overlay normal distribution with same mean and std
    x = np.linspace(returns_pct.min(), returns_pct.max(), 200)
    mu, sigma = returns_pct.mean(), returns_pct.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal distribution')

    ax.set_title(f'{symbol} — Return distribution')
    ax.set_xlabel('Daily return (%)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)


# --- Run for BTC, ETH, SOL ---
btc_df, btc_ret = analyze('BTC/USDT', days=1825)
eth_df, eth_ret = analyze('ETH/USDT', days=1825)
sol_df, sol_ret = analyze('SOL/USDT', days=1825)

# --- Plot all three side by side ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_distribution(btc_ret, 'BTC/USDT', axes[0])
plot_distribution(eth_ret, 'ETH/USDT', axes[1])
plot_distribution(sol_ret, 'SOL/USDT', axes[2])
plt.tight_layout()
plt.savefig('notes/day2_distributions.png', dpi=100)
plt.show()

# --- Q-Q plot: another way to see fat tails ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (returns, name) in zip(axes, [(btc_ret, 'BTC'), (eth_ret, 'ETH'), (sol_ret, 'SOL')]):
    stats.probplot(returns, dist='norm', plot=ax)
    ax.set_title(f'{name} — Q-Q plot vs Normal')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notes/day2_qq_plots.png', dpi=100)
plt.show()

print("\nCharts saved to notes/")