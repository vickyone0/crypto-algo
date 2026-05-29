"""
Day 5: Grid search and the overfitting trap.
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


def backtest(df, fast, slow, fee=0.001):
    """Returns (total_return, sharpe, max_dd, num_trades)."""
    df = df.copy()
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()
    df['signal'] = (df['ma_fast'] > df['ma_slow']).astype(int)
    df['position'] = df['signal'].shift(1).fillna(0)
    df['market_return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'] * df['market_return']
    df['trade'] = df['position'].diff().abs()
    df['strategy_return'] -= df['trade'] * fee

    strat = df['strategy_return'].dropna()
    equity = (1 + df['strategy_return']).cumprod()

    total_return = equity.iloc[-1] - 1
    sharpe = (strat.mean() / strat.std()) * np.sqrt(365) if strat.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    num_trades = int(df['trade'].sum())

    return total_return, sharpe, max_dd, num_trades


# --- Fetch data ---
df = fetch_history('BTC/USDT', days=1825)

# --- EXPERIMENT 1: Grid search on ALL 5 years ---
print("=" * 60)
print("  EXPERIMENT 1: Grid search on full 5 years")
print("=" * 60)

fast_range = [5, 10, 15, 20, 25, 30, 40, 50]
slow_range = [30, 50, 75, 100, 150, 200]

results = []
for fast in fast_range:
    for slow in slow_range:
        if fast >= slow:
            continue
        ret, sharpe, dd, trades = backtest(df, fast, slow)
        results.append({
            'fast': fast, 'slow': slow,
            'return': ret, 'sharpe': sharpe, 'max_dd': dd, 'trades': trades
        })

results_df = pd.DataFrame(results)
print(f"\nTested {len(results_df)} parameter combinations.")
print("\nTop 5 by total return:")
print(results_df.nlargest(5, 'return').to_string(index=False))
print("\nTop 5 by Sharpe:")
print(results_df.nlargest(5, 'sharpe').to_string(index=False))
print("\nBottom 5 (the cherry-picking warning):")
print(results_df.nsmallest(5, 'return').to_string(index=False))

best_full = results_df.loc[results_df['return'].idxmax()]
print(f"\n>>> 'Best' parameters on full data: fast={int(best_full['fast'])}, slow={int(best_full['slow'])}")
print(f"    Return: {best_full['return']*100:+.2f}%, Sharpe: {best_full['sharpe']:.2f}")

# --- EXPERIMENT 2: Heatmap of returns by parameter combo ---
pivot = results_df.pivot(index='fast', columns='slow', values='return') * 100

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel('Slow MA')
ax.set_ylabel('Fast MA')
ax.set_title('Total Return (%) by MA parameters — FULL DATA\n(see how unstable nearby cells are)')
plt.colorbar(im, ax=ax, label='Return %')

# annotate each cell
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('notes/day5_heatmap.png', dpi=100)
plt.show()

# --- EXPERIMENT 3: THE TRAP — In-sample vs out-of-sample ---
print("\n" + "=" * 60)
print("  EXPERIMENT 3: The Overfitting Trap")
print("  Optimize on first half, test on second half")
print("=" * 60)

split = len(df) // 2
df_train = df.iloc[:split].copy()
df_test  = df.iloc[split:].copy()

print(f"\nTrain period: {df_train.index[0].date()} to {df_train.index[-1].date()}")
print(f"Test  period: {df_test.index[0].date()} to {df_test.index[-1].date()}")

# Find best parameters on TRAIN only
train_results = []
for fast in fast_range:
    for slow in slow_range:
        if fast >= slow:
            continue
        ret, sharpe, dd, trades = backtest(df_train, fast, slow)
        train_results.append({'fast': fast, 'slow': slow, 'return': ret, 'sharpe': sharpe})

train_df = pd.DataFrame(train_results)
best_train = train_df.loc[train_df['return'].idxmax()]
best_fast, best_slow = int(best_train['fast']), int(best_train['slow'])

print(f"\nBest params on TRAIN: fast={best_fast}, slow={best_slow}")
print(f"  Train return: {best_train['return']*100:+.2f}%")

# Apply those "best" params to TEST data (data the optimizer never saw)
test_ret, test_sharpe, test_dd, test_trades = backtest(df_test, best_fast, best_slow)
print(f"\nApplying those same params to TEST (unseen) data:")
print(f"  Test  return: {test_ret*100:+.2f}%")
print(f"  Test  Sharpe: {test_sharpe:.2f}")
print(f"  Test  max DD: {test_dd*100:.2f}%")

# Compare to buy & hold over test period
bh_ret = df_test['close'].iloc[-1] / df_test['close'].iloc[0] - 1
print(f"\nBuy & hold over test period: {bh_ret*100:+.2f}%")

# --- Final visualization: how unstable is "best"? ---
print("\n" + "=" * 60)
print("  How stable is the 'best' result?")
print("=" * 60)
print("Returns of parameters NEAR the optimum (full data):")
near = results_df[
    (results_df['fast'].between(best_fast-5, best_fast+5)) &
    (results_df['slow'].between(best_slow-25, best_slow+25))
]
print(near.to_string(index=False))