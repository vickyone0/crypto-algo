"""
Day 4: Moving Average Crossover strategy with HONEST backtesting.
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def backtest_ma_crossover(df, fast=20, slow=50, fee=0.001):
    df = df.copy()

    # 1. Calculate the two moving averages
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()

    # 2. Generate signal: 1 = want to be in the market, 0 = want to be out
    #    (fast above slow = uptrend = hold BTC)
    df['signal'] = (df['ma_fast'] > df['ma_slow']).astype(int)

    # 3. CRITICAL: shift signal by 1 day to avoid look-ahead bias.
    #    We see the signal at today's close, but can only act tomorrow.
    df['position'] = df['signal'].shift(1).fillna(0)

    # 4. Daily returns of the asset
    df['market_return'] = df['close'].pct_change()

    # 5. Strategy return = market return ONLY on days we held a position
    df['strategy_return'] = df['position'] * df['market_return']

    # 6. Subtract fees whenever the position CHANGES (a trade happens)
    df['trade'] = df['position'].diff().abs()
    df['strategy_return'] -= df['trade'] * fee

    # 7. Build equity curves (cumulative growth of $1)
    df['market_equity'] = (1 + df['market_return']).cumprod()
    df['strategy_equity'] = (1 + df['strategy_return']).cumprod()

    return df


def performance_stats(df, name):
    strat = df['strategy_return'].dropna()
    mkt = df['market_return'].dropna()

    def annualized_sharpe(returns):
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(365)

    def max_drawdown(equity):
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return dd.min()

    total_return = df['strategy_equity'].iloc[-1] - 1
    buy_hold_return = df['market_equity'].iloc[-1] - 1
    num_trades = int(df['trade'].sum())

    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print('=' * 55)
    print(f"Strategy total return:   {total_return * 100:+.2f}%")
    print(f"Buy & hold return:       {buy_hold_return * 100:+.2f}%")
    print(f"Strategy Sharpe:         {annualized_sharpe(strat):.2f}")
    print(f"Buy & hold Sharpe:       {annualized_sharpe(mkt):.2f}")
    print(f"Strategy max drawdown:   {max_drawdown(df['strategy_equity']) * 100:.2f}%")
    print(f"Buy & hold max drawdown: {max_drawdown(df['market_equity']) * 100:.2f}%")
    print(f"Number of trades:        {num_trades}")
    print(f"Days in market:          {(df['position'] == 1).sum()} / {len(df)}")


# --- Run it ---
df = fetch_history('BTC/USDT', days=1825)
result = backtest_ma_crossover(df, fast=20, slow=50, fee=0.001)
performance_stats(result, "BTC MA Crossover (20/50)")

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})

axes[0].plot(result.index, result['close'], color='lightgray', label='BTC Price', linewidth=1)
axes[0].plot(result.index, result['ma_fast'], color='blue', label='20-day MA', linewidth=1)
axes[0].plot(result.index, result['ma_slow'], color='red', label='50-day MA', linewidth=1)
# Shade the periods we're holding
axes[0].fill_between(result.index, result['close'].min(), result['close'].max(),
                     where=result['position'] == 1, alpha=0.1, color='green', label='In market')
axes[0].set_title('BTC Price with Moving Averages (green = holding)')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

axes[1].plot(result.index, result['strategy_equity'], color='green', label='Strategy', linewidth=1.5)
axes[1].plot(result.index, result['market_equity'], color='gray', label='Buy & Hold', linewidth=1.5)
axes[1].axhline(1, color='black', linewidth=0.5)
axes[1].set_title('Equity Curve — Growth of $1')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notes/day4_ma_crossover.png', dpi=100)
plt.show()

print("\nChart saved to notes/day4_ma_crossover.png")