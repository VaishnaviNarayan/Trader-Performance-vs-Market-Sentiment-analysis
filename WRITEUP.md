# Trader Performance vs Market Sentiment — Write-Up

## Methodology

### Data Sources
1. **Bitcoin Fear & Greed Index** — 2,644 daily records (2018-02-01 to 2025-05-02) with sentiment classifications: Extreme Fear, Fear, Neutral, Greed, Extreme Greed
2. **Hyperliquid Trader Data** — 211,224 trade records from 32 unique accounts across 246 coins (2023-03-28 to 2025-06-15)

### Approach
1. **Data Preparation**: Loaded both datasets (no missing values, no duplicates). Converted epoch timestamps to datetime, aligned on daily level. Engineered metrics: daily PnL, win rate, trade size, long/short ratio, volume.
2. **Merge**: Inner join on date produced 77 daily account-level observations across Fear (32), Greed (37), and Neutral (8) days.
3. **Analysis**: Compared performance and behavior across sentiment classes, segmented traders into 3 dimensions, and cross-analyzed segments × sentiment.
4. **Modeling**: Random Forest classifier to predict profitability bucket (Profit/Loss/Breakeven). K-Means clustering to identify behavioral archetypes.

---

## Key Insights

### Insight 1: Traders Are Actually More Profitable on Fear Days
- **Fear days** average PnL: **$209,373** vs Greed days: **$90,989** (2.3× higher)
- Win rate is also higher in Fear (41.6% vs 36.9%)
- **Implication**: Contrarian opportunities are more lucrative; Fear = volatility = larger directional moves for skilled traders.

### Insight 2: Trading Volume and Frequency Spike During Fear
- Fear days see **3.6× more trades** (4,184 vs 1,169 per account-day)
- Long/Short ratio is **balanced at ~1.0 during Fear** but **skews heavily long (10.0) during Greed**
- **Implication**: During Greed, traders pile into long positions (crowding risk). During Fear, traders are more balanced and active — potentially capturing more mean-reversion opportunities.

### Insight 3: High-Size Traders Outperform But With Much Higher Variance
- High-size traders: avg PnL $386K but lower win rate (34%) — high risk, high reward
- Low-size traders: avg PnL $253K with higher win rate (42%) — more consistent
- Frequent traders earn more in total ($486K vs $153K) but success is concentration-dependent

---

## Strategy Recommendations

### Strategy 1 — Reduce Position Sizes for High-Risk Traders During Fear
High-size traders earn $278K on Fear days vs $85K on Greed days. While profitable on average, the extreme PnL variance ($380K std dev) creates dangerous drawdown risk.

**Rule**: During Fear days, high-size traders should cap position sizes at 70% of their normal allocation to reduce tail risk while still capturing the higher average returns.

### Strategy 2 — Selective Trading by Frequency Segment
Frequent traders earn significantly more on Fear days ($324K vs $142K on Greed). Infrequent traders see the same directional pattern but at lower magnitude.

**Rule**: During Fear days, adopt a selective approach — make fewer but higher-conviction trades. During Greed days, momentum-following (frequent trading) is acceptable, but avoid crowded long positions (L/S ratio >5:1).

---

## Model Performance
- **Random Forest Classifier**: 81.25% accuracy (5-fold CV: 84.33% ± 3.6%)
- **Top features**: win_rate, avg_trade_size_usd, sentiment value, num_trades
- **Clustering**: 3 archetypes identified via K-Means — Active Winners, Active Losers, Passive Traders
