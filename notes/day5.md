# Day 5 — Overfitting Trap

## Grid search results
- Combinations tested: 44
- Best return: fast=25, slow=30, return = +247.37%
- Worst return: fast=15, slow=50, return = +8.77%
- Spread (best - worst): 238.6 percentage points

## Heatmap observations
- PATCHY — the top result is +247% but immediate neighbors are +14% to +36%
- This means: the "best" is a lucky island, not a robust edge

## The train/test betrayal (partial)
- Best params on TRAIN: fast=25, slow=30, train return = +73.04%
- Same params on TEST:  test return = +78.18%
- Buy & hold on test: +94.81%
- Did "best" beat buy & hold on unseen data? NO (78% vs 95%)
- But: strategy DD only -25% vs buy & hold's much larger DD

## Stability check — the smoking gun
- 25/30 returned +247%
- 20/30 returned +35%   (one step away → 7x collapse)
- 25/50 returned +36%   (other step away → 7x collapse)
- 30/50 returned +36%
- Conclusion: NOT STABLE. The "best" is an isolated lucky cell.

## The lesson
- A real edge looks like a smooth hill across parameters.
- A lucky overfit looks like a lonely spike.
- +247% on past data ≠ +247% in the future when neighbors return +35%.
- Even when params "survive" train→test, they still lose to buy & hold.

## Tomorrow's goal
- Try a different STRATEGY TYPE (mean reversion with Bollinger Bands)
  and see if it has a smoother stability profile.