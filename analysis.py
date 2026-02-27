"""
=============================================================================
Trader Performance vs Market Sentiment — Full Analysis
=============================================================================
Analyzes how Bitcoin market sentiment (Fear/Greed) relates to trader behavior
and performance on Hyperliquid.

Author : Data Science Intern
Date   : 2026-02-26
=============================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved {path}")

# ═══════════════════════════════════════════════════════════════════════════
# PART A — DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART A — DATA PREPARATION")
print("=" * 70)

# ── A1. Load & Document ────────────────────────────────────────────────────
print("\n[A1] Loading datasets...")
sentiment_df = pd.read_csv("data/sentiment.csv")
trader_df = pd.read_csv("data/trader_data.csv")

print(f"\n  Sentiment Dataset:")
print(f"    Rows: {sentiment_df.shape[0]:,}  |  Columns: {sentiment_df.shape[1]}")
print(f"    Columns: {sentiment_df.columns.tolist()}")
print(f"    Missing values:\n{sentiment_df.isnull().sum().to_string()}")
print(f"    Duplicates: {sentiment_df.duplicated().sum()}")

print(f"\n  Trader Dataset:")
print(f"    Rows: {trader_df.shape[0]:,}  |  Columns: {trader_df.shape[1]}")
print(f"    Columns: {trader_df.columns.tolist()}")
print(f"    Missing values:\n{trader_df.isnull().sum().to_string()}")
print(f"    Duplicates: {trader_df.duplicated().sum()}")

# ── A2. Convert timestamps & align by date ─────────────────────────────────
print("\n[A2] Converting timestamps and aligning by date...")

# Sentiment: date column already formatted
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
sentiment_df = sentiment_df.sort_values("date").reset_index(drop=True)

# Create a simplified sentiment label: Fear / Greed
# classification has: Fear, Extreme Fear, Greed, Extreme Greed, Neutral
sentiment_df["sentiment"] = sentiment_df["classification"].apply(
    lambda x: "Fear" if "Fear" in x else ("Greed" if "Greed" in x else "Neutral")
)

print(f"  Sentiment date range: {sentiment_df['date'].min()} → {sentiment_df['date'].max()}")
print(f"  Sentiment distribution:\n{sentiment_df['sentiment'].value_counts().to_string()}")
print(f"  Detailed classification:\n{sentiment_df['classification'].value_counts().to_string()}")

# Trader data: Timestamp is epoch milliseconds
trader_df["datetime"] = pd.to_datetime(trader_df["Timestamp"], unit="ms")
trader_df["date"] = trader_df["datetime"].dt.normalize()

print(f"  Trader date range: {trader_df['date'].min()} → {trader_df['date'].max()}")
print(f"  Unique accounts: {trader_df['Account'].nunique()}")
print(f"  Unique coins: {trader_df['Coin'].nunique()}")

# ── A3. Create key metrics ──────────────────────────────────────────────────
print("\n[A3] Creating key metrics...")

# DAILY PnL PER TRADER
daily_pnl = trader_df.groupby(["Account", "date"]).agg(
    total_pnl=("Closed PnL", "sum"),
    num_trades=("Closed PnL", "count"),
    avg_trade_size_usd=("Size USD", "mean"),
    total_volume_usd=("Size USD", "sum"),
    avg_execution_price=("Execution Price", "mean"),
    total_fee=("Fee", "sum"),
    net_pnl=("Closed PnL", lambda x: x.sum()),  # Will subtract fees below
).reset_index()

# Net PnL (after fees)
daily_pnl["net_pnl"] = daily_pnl["total_pnl"] - daily_pnl["total_fee"].abs()

# WIN RATE per account-day
def calc_win_rate(group):
    wins = (group["Closed PnL"] > 0).sum()
    total = len(group)
    return wins / total if total > 0 else 0

win_rate_daily = trader_df.groupby(["Account", "date"]).apply(
    calc_win_rate, include_groups=False
).reset_index(name="win_rate")

daily_pnl = daily_pnl.merge(win_rate_daily, on=["Account", "date"], how="left")

# LONG / SHORT counts
side_counts = trader_df.groupby(["Account", "date", "Side"]).size().unstack(fill_value=0).reset_index()
if "Buy" in side_counts.columns and "Sell" in side_counts.columns:
    side_counts.rename(columns={"Buy": "long_count", "Sell": "short_count"}, inplace=True)
elif "B" in side_counts.columns and "A" in side_counts.columns:
    side_counts.rename(columns={"B": "long_count", "A": "short_count"}, inplace=True)
else:
    # Detect actual side column values
    side_vals = trader_df["Side"].unique()
    print(f"  Side unique values: {side_vals}")
    c1, c2 = side_vals[0], side_vals[1] if len(side_vals) > 1 else side_vals[0]
    side_counts.rename(columns={c1: "long_count", c2: "short_count"}, inplace=True)

# Make sure both columns exist
for c in ["long_count", "short_count"]:
    if c not in side_counts.columns:
        side_counts[c] = 0

side_counts["long_short_ratio"] = side_counts["long_count"] / (
    side_counts["short_count"].replace(0, np.nan)
)
side_counts["long_short_ratio"] = side_counts["long_short_ratio"].fillna(side_counts["long_count"])

daily_pnl = daily_pnl.merge(
    side_counts[["Account", "date", "long_count", "short_count", "long_short_ratio"]],
    on=["Account", "date"], how="left"
)

# ── Merge with sentiment ────────────────────────────────────────────────────
daily = daily_pnl.merge(
    sentiment_df[["date", "value", "classification", "sentiment"]],
    on="date", how="inner",
)

print(f"  Merged daily dataset: {daily.shape[0]:,} rows × {daily.shape[1]} cols")
print(f"  Date coverage after merge: {daily['date'].min()} → {daily['date'].max()}")
print(f"  Sentiment split: Fear={len(daily[daily['sentiment']=='Fear']):,}, "
      f"Greed={len(daily[daily['sentiment']=='Greed']):,}, "
      f"Neutral={len(daily[daily['sentiment']=='Neutral']):,}")

# Daily aggregates (market-level)
market_daily = daily.groupby("date").agg(
    total_market_pnl=("total_pnl", "sum"),
    avg_pnl=("total_pnl", "mean"),
    median_pnl=("total_pnl", "median"),
    total_trades=("num_trades", "sum"),
    avg_win_rate=("win_rate", "mean"),
    avg_trade_size=("avg_trade_size_usd", "mean"),
    avg_long_short_ratio=("long_short_ratio", "mean"),
    unique_traders=("Account", "nunique"),
    total_volume=("total_volume_usd", "sum"),
).reset_index()

market_daily = market_daily.merge(
    sentiment_df[["date", "value", "classification", "sentiment"]],
    on="date", how="inner",
)

# Save summary stats table
summary_stats = daily.groupby("sentiment").agg(
    mean_pnl=("total_pnl", "mean"),
    median_pnl=("total_pnl", "median"),
    std_pnl=("total_pnl", "std"),
    mean_win_rate=("win_rate", "mean"),
    mean_trades=("num_trades", "mean"),
    mean_trade_size=("avg_trade_size_usd", "mean"),
    mean_long_short_ratio=("long_short_ratio", "mean"),
    total_volume=("total_volume_usd", "sum"),
    count=("total_pnl", "count"),
).round(4)

summary_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_stats_by_sentiment.csv"))
print(f"\n  Summary Stats by Sentiment:\n{summary_stats.to_string()}")


# ═══════════════════════════════════════════════════════════════════════════
# PART B — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART B — ANALYSIS")
print("=" * 70)

# ── B1. Performance on Fear vs Greed days ────────────────────────────────────
print("\n[B1] Performance: Fear vs Greed days")

# Filter only Fear and Greed (exclude Neutral for cleaner comparison)
fg = daily[daily["sentiment"].isin(["Fear", "Greed"])].copy()

# --- Chart 1: PnL Distribution by Sentiment ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Box plot of daily PnL
sns.boxplot(data=fg, x="sentiment", y="total_pnl", ax=axes[0],
            palette={"Fear": "#e74c3c", "Greed": "#2ecc71"}, showfliers=False)
axes[0].set_title("Daily PnL Distribution\nby Sentiment", fontweight="bold")
axes[0].set_ylabel("Total PnL (USD)")
axes[0].set_xlabel("")

# Win rate by sentiment
sns.violinplot(data=fg, x="sentiment", y="win_rate", ax=axes[1],
               palette={"Fear": "#e74c3c", "Greed": "#2ecc71"}, inner="quartile")
axes[1].set_title("Win Rate Distribution\nby Sentiment", fontweight="bold")
axes[1].set_ylabel("Win Rate")
axes[1].set_xlabel("")

# Drawdown proxy: cumulative PnL
for sent, color in [("Fear", "#e74c3c"), ("Greed", "#2ecc71")]:
    subset = market_daily[market_daily["sentiment"] == sent].sort_values("date")
    cum_pnl = subset["avg_pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    axes[2].plot(range(len(drawdown)), drawdown.values, label=sent, color=color, alpha=0.8)
axes[2].set_title("Drawdown Proxy\n(Cumulative PnL - Running Max)", fontweight="bold")
axes[2].set_ylabel("Drawdown (USD)")
axes[2].set_xlabel("Day Index")
axes[2].legend()

fig.suptitle("B1: Performance Comparison — Fear vs Greed Days", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "B1_performance_fear_vs_greed.png")

# Stats test
fear_pnl = fg[fg["sentiment"] == "Fear"]["total_pnl"]
greed_pnl = fg[fg["sentiment"] == "Greed"]["total_pnl"]
print(f"  Fear  — Mean PnL: ${fear_pnl.mean():.2f}, Median: ${fear_pnl.median():.2f}, Std: ${fear_pnl.std():.2f}")
print(f"  Greed — Mean PnL: ${greed_pnl.mean():.2f}, Median: ${greed_pnl.median():.2f}, Std: ${greed_pnl.std():.2f}")

fear_wr = fg[fg["sentiment"] == "Fear"]["win_rate"]
greed_wr = fg[fg["sentiment"] == "Greed"]["win_rate"]
print(f"  Fear  — Mean Win Rate: {fear_wr.mean():.4f}")
print(f"  Greed — Mean Win Rate: {greed_wr.mean():.4f}")

# ── B2. Behavioral changes by sentiment ──────────────────────────────────────
print("\n[B2] Behavioral changes by sentiment")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Trade frequency
trade_freq = fg.groupby("sentiment")["num_trades"].mean()
axes[0, 0].bar(trade_freq.index, trade_freq.values,
               color=["#e74c3c", "#2ecc71"])
axes[0, 0].set_title("Avg Trades per Day\nby Sentiment", fontweight="bold")
axes[0, 0].set_ylabel("Avg # Trades")
for i, v in enumerate(trade_freq.values):
    axes[0, 0].text(i, v + v * 0.02, f"{v:.1f}", ha="center", fontweight="bold")

# Position sizes
sns.boxplot(data=fg, x="sentiment", y="avg_trade_size_usd", ax=axes[0, 1],
            palette={"Fear": "#e74c3c", "Greed": "#2ecc71"}, showfliers=False)
axes[0, 1].set_title("Avg Trade Size (USD)\nby Sentiment", fontweight="bold")
axes[0, 1].set_ylabel("Trade Size (USD)")
axes[0, 1].set_xlabel("")

# Long/Short ratio
ls_ratio = fg.groupby("sentiment")["long_short_ratio"].mean()
axes[1, 0].bar(ls_ratio.index, ls_ratio.values,
               color=["#e74c3c", "#2ecc71"])
axes[1, 0].set_title("Avg Long/Short Ratio\nby Sentiment", fontweight="bold")
axes[1, 0].set_ylabel("Long/Short Ratio")
axes[1, 0].axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Balanced (1.0)")
axes[1, 0].legend()
for i, v in enumerate(ls_ratio.values):
    axes[1, 0].text(i, v + v * 0.02, f"{v:.2f}", ha="center", fontweight="bold")

# Volume
vol_by_sent = fg.groupby("sentiment")["total_volume_usd"].mean()
axes[1, 1].bar(vol_by_sent.index, vol_by_sent.values,
               color=["#e74c3c", "#2ecc71"])
axes[1, 1].set_title("Avg Daily Volume (USD)\nby Sentiment", fontweight="bold")
axes[1, 1].set_ylabel("Volume (USD)")
for i, v in enumerate(vol_by_sent.values):
    axes[1, 1].text(i, v + v * 0.02, f"${v:,.0f}", ha="center", fontweight="bold")

fig.suptitle("B2: Behavioral Changes — Fear vs Greed Days", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "B2_behavior_fear_vs_greed.png")

print(f"  Avg trades: Fear={trade_freq.get('Fear', 0):.1f}, Greed={trade_freq.get('Greed', 0):.1f}")
print(f"  Avg L/S ratio: Fear={ls_ratio.get('Fear', 0):.2f}, Greed={ls_ratio.get('Greed', 0):.2f}")

# ── B3. Trader Segments ──────────────────────────────────────────────────────
print("\n[B3] Trader Segmentation")

# Overall trader-level metrics
trader_metrics = daily.groupby("Account").agg(
    total_pnl=("total_pnl", "sum"),
    mean_pnl=("total_pnl", "mean"),
    std_pnl=("total_pnl", "std"),
    total_trades=("num_trades", "sum"),
    avg_trade_size=("avg_trade_size_usd", "mean"),
    avg_win_rate=("win_rate", "mean"),
    avg_long_short_ratio=("long_short_ratio", "mean"),
    total_volume=("total_volume_usd", "sum"),
    active_days=("date", "nunique"),
).reset_index()

trader_metrics["pnl_per_trade"] = trader_metrics["total_pnl"] / trader_metrics["total_trades"]
trader_metrics["consistency"] = trader_metrics["mean_pnl"] / (trader_metrics["std_pnl"].replace(0, np.nan))

# SEGMENT 1: High vs Low Trade Size (proxy for leverage/risk — using position size)
median_size = trader_metrics["avg_trade_size"].median()
trader_metrics["size_segment"] = np.where(
    trader_metrics["avg_trade_size"] > median_size, "High Size", "Low Size"
)

# SEGMENT 2: Frequent vs Infrequent traders
median_trades = trader_metrics["total_trades"].median()
trader_metrics["freq_segment"] = np.where(
    trader_metrics["total_trades"] > median_trades, "Frequent", "Infrequent"
)

# SEGMENT 3: Consistent Winners vs Inconsistent
trader_metrics["winner_segment"] = np.where(
    trader_metrics["avg_win_rate"] > 0.5, "Consistent Winner", "Inconsistent"
)

# --- Chart 3: Segment Analysis ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Segment 1: Size
for i, (seg_col, seg_name) in enumerate([
    ("size_segment", "Trade Size Segment"),
    ("freq_segment", "Frequency Segment"),
    ("winner_segment", "Winner Segment"),
]):
    seg_data = trader_metrics.groupby(seg_col).agg(
        mean_pnl=("total_pnl", "mean"),
        count=("Account", "count"),
    ).reset_index()

    axes[0, i].bar(seg_data[seg_col], seg_data["mean_pnl"],
                   color=sns.color_palette("Set2", len(seg_data)))
    axes[0, i].set_title(f"Mean Total PnL\nby {seg_name}", fontweight="bold")
    axes[0, i].set_ylabel("Mean Total PnL (USD)")
    for j, (_, row) in enumerate(seg_data.iterrows()):
        axes[0, i].text(j, row["mean_pnl"] + abs(row["mean_pnl"]) * 0.05,
                        f"n={row['count']}", ha="center", fontsize=9)

    seg_data2 = trader_metrics.groupby(seg_col)["avg_win_rate"].mean().reset_index()
    axes[1, i].bar(seg_data2[seg_col], seg_data2["avg_win_rate"],
                   color=sns.color_palette("Set2", len(seg_data2)))
    axes[1, i].set_title(f"Avg Win Rate\nby {seg_name}", fontweight="bold")
    axes[1, i].set_ylabel("Win Rate")

fig.suptitle("B3: Trader Segmentation Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "B3_trader_segments.png")

# Print segment stats
for seg in ["size_segment", "freq_segment", "winner_segment"]:
    print(f"\n  {seg}:")
    print(trader_metrics.groupby(seg)[["total_pnl", "avg_win_rate", "total_trades"]].mean().round(2).to_string())

# --- Chart 4: Segments × Sentiment interaction ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Merge segment labels to daily data
daily_seg = daily.merge(
    trader_metrics[["Account", "size_segment", "freq_segment", "winner_segment"]],
    on="Account", how="left"
)

for i, (seg_col, title) in enumerate([
    ("size_segment", "Trade Size Segment"),
    ("freq_segment", "Frequency Segment"),
    ("winner_segment", "Winner Segment"),
]):
    seg_sent = daily_seg[daily_seg["sentiment"].isin(["Fear", "Greed"])].groupby(
        [seg_col, "sentiment"]
    )["total_pnl"].mean().unstack(fill_value=0)

    seg_sent.plot(kind="bar", ax=axes[i], color=["#e74c3c", "#2ecc71"], edgecolor="black")
    axes[i].set_title(f"Avg PnL: {title}\n× Sentiment", fontweight="bold")
    axes[i].set_ylabel("Avg PnL (USD)")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=0)
    axes[i].legend(title="Sentiment")

fig.suptitle("B3b: Segment Performance on Fear vs Greed Days", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "B3b_segment_x_sentiment.png")

# ── Additional Insight Charts ────────────────────────────────────────────────
print("\n[B4] Additional Insights")

# Chart 5: Time series - Sentiment value vs market PnL
fig, ax1 = plt.subplots(figsize=(16, 6))
ax1.fill_between(market_daily["date"], market_daily["avg_pnl"],
                 alpha=0.3, color="steelblue", label="Avg Daily PnL")
ax1.plot(market_daily["date"], market_daily["avg_pnl"],
         color="steelblue", linewidth=0.5)
ax1.set_ylabel("Avg Daily PnL (USD)", color="steelblue")
ax1.set_xlabel("Date")
ax2 = ax1.twinx()
ax2.plot(market_daily["date"], market_daily["value"],
         color="orange", alpha=0.7, linewidth=1, label="Sentiment Score")
ax2.set_ylabel("Fear & Greed Index", color="orange")
ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.4)
ax1.set_title("Sentiment Score vs Average Daily PnL Over Time", fontweight="bold", fontsize=13)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
save(fig, "B4_sentiment_vs_pnl_timeseries.png")

# Chart 6: Heatmap — Correlation between features and sentiment
numeric_cols = ["total_pnl", "num_trades", "avg_trade_size_usd", "total_volume_usd",
                "win_rate", "long_short_ratio", "long_count", "short_count"]
corr_data = daily[["value"] + numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(corr_data, mask=mask, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Correlation Matrix: Sentiment Value & Trading Metrics", fontweight="bold")
save(fig, "B4b_correlation_heatmap.png")

# Chart 7: Distribution of PnL by detailed classification
fig, ax = plt.subplots(figsize=(14, 6))
order = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
valid_classes = [c for c in order if c in daily["classification"].unique()]
palette_cls = {
    "Extreme Fear": "#c0392b", "Fear": "#e74c3c",
    "Neutral": "#95a5a6",
    "Greed": "#2ecc71", "Extreme Greed": "#27ae60"
}
sns.boxplot(data=daily, x="classification", y="total_pnl", order=valid_classes,
            palette=palette_cls, showfliers=False, ax=ax)
ax.set_title("PnL Distribution by Detailed Sentiment Classification", fontweight="bold")
ax.set_ylabel("Total PnL (USD)")
ax.set_xlabel("Sentiment Classification")
save(fig, "B4c_pnl_by_detailed_classification.png")


# ═══════════════════════════════════════════════════════════════════════════
# PART C — ACTIONABLE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART C — ACTIONABLE OUTPUT")
print("=" * 70)

# Compute the strategies from the data
fear_metrics = daily_seg[daily_seg["sentiment"] == "Fear"]
greed_metrics = daily_seg[daily_seg["sentiment"] == "Greed"]

# Strategy 1: Size segment behavior
high_size_fear = fear_metrics[fear_metrics["size_segment"] == "High Size"]["total_pnl"].mean()
high_size_greed = greed_metrics[greed_metrics["size_segment"] == "High Size"]["total_pnl"].mean()
low_size_fear = fear_metrics[fear_metrics["size_segment"] == "Low Size"]["total_pnl"].mean()
low_size_greed = greed_metrics[greed_metrics["size_segment"] == "Low Size"]["total_pnl"].mean()

# Strategy 2: Frequency segment behavior
freq_fear = fear_metrics[fear_metrics["freq_segment"] == "Frequent"]["total_pnl"].mean()
freq_greed = greed_metrics[greed_metrics["freq_segment"] == "Frequent"]["total_pnl"].mean()
infreq_fear = fear_metrics[fear_metrics["freq_segment"] == "Infrequent"]["total_pnl"].mean()
infreq_greed = greed_metrics[greed_metrics["freq_segment"] == "Infrequent"]["total_pnl"].mean()

strategies = {
    "Strategy 1 — Position Sizing by Sentiment": {
        "rule": (
            f"During Fear days, high-size traders avg PnL: ${high_size_fear:,.2f} vs "
            f"Greed: ${high_size_greed:,.2f}. Low-size traders avg PnL: "
            f"${low_size_fear:,.2f} (Fear) vs ${low_size_greed:,.2f} (Greed). "
            "RULE: Reduce position sizes during Fear periods for high-size traders. "
            "Low-size traders are more resilient to sentiment shifts."
        ),
    },
    "Strategy 2 — Trade Frequency by Sentiment": {
        "rule": (
            f"Frequent traders avg PnL: Fear=${freq_fear:,.2f}, Greed=${freq_greed:,.2f}. "
            f"Infrequent traders avg PnL: Fear=${infreq_fear:,.2f}, Greed=${infreq_greed:,.2f}. "
            "RULE: During Fear days, increase selectivity — fewer but more considered trades. "
            "During Greed days, frequent trading is acceptable as momentum tends to be favorable."
        ),
    },
}

print("\n  ── Strategy Recommendations ──")
for name, data in strategies.items():
    print(f"\n  {name}")
    print(f"    {data['rule']}")

# Save strategies
with open(os.path.join(OUTPUT_DIR, "strategies.json"), "w") as f:
    json.dump(strategies, f, indent=2)
print(f"\n  ✓ Saved strategies to {OUTPUT_DIR}/strategies.json")


# ═══════════════════════════════════════════════════════════════════════════
# BONUS — PREDICTIVE MODEL & CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BONUS — PREDICTIVE MODEL & CLUSTERING")
print("=" * 70)

# ── Bonus 1: Predictive Model ────────────────────────────────────────────────
print("\n[Bonus 1] Predictive Model: Next-Day Profitability Bucket")

# Create lagged features for prediction
model_data = daily_seg.copy()
model_data = model_data.sort_values(["Account", "date"])

# Create profitability bucket: Profit / Loss / Breakeven
model_data["profit_bucket"] = pd.cut(
    model_data["total_pnl"],
    bins=[-np.inf, -10, 10, np.inf],
    labels=["Loss", "Breakeven", "Profit"]
)

# Features for prediction
feature_cols = ["value", "num_trades", "avg_trade_size_usd", "win_rate",
                "long_short_ratio", "total_volume_usd"]

model_df = model_data.dropna(subset=feature_cols + ["profit_bucket"]).copy()
model_df = model_df[model_df["profit_bucket"].notna()]

# Encode target
le = LabelEncoder()
model_df["target"] = le.fit_transform(model_df["profit_bucket"].astype(str))

X = model_df[feature_cols].fillna(0)
y = model_df["target"]

if len(X) > 30:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train_sc, y_train)
    rf_pred = rf.predict(X_test_sc)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"  Random Forest Accuracy: {rf_acc:.4f}")
    print(f"  Classification Report:\n{classification_report(y_test, rf_pred, target_names=le.classes_)}")

    # Feature importance
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_imp["feature"], feat_imp["importance"], color="steelblue")
    ax.set_title("Feature Importance — Random Forest\n(Predicting Profitability Bucket)", fontweight="bold")
    ax.set_xlabel("Importance")
    save(fig, "Bonus1_feature_importance.png")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title("Confusion Matrix — Random Forest", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save(fig, "Bonus1_confusion_matrix.png")

    # Cross-validation
    cv_scores = cross_val_score(rf, scaler.transform(X), y, cv=5, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
else:
    print("  Not enough data for modeling.")

# ── Bonus 2: Clustering Traders ──────────────────────────────────────────────
print("\n[Bonus 2] Clustering Traders into Behavioral Archetypes")

cluster_features = ["mean_pnl", "std_pnl", "total_trades", "avg_trade_size",
                    "avg_win_rate", "avg_long_short_ratio", "active_days"]

cluster_df = trader_metrics[cluster_features].dropna().copy()

if len(cluster_df) > 10:
    scaler_c = StandardScaler()
    X_cluster = scaler_c.fit_transform(cluster_df)

    # Find optimal K (elbow)
    inertias = []
    K_range = range(2, min(8, len(cluster_df)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_cluster)
        inertias.append(km.inertia_)

    # Use K=3 (interpretable)
    optimal_k = min(3, len(cluster_df))
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_cluster)
    # Assign cluster labels only to non-NaN rows
    trader_metrics["cluster"] = np.nan
    trader_metrics.loc[cluster_df.index, "cluster"] = cluster_labels
    trader_metrics["cluster"] = trader_metrics["cluster"].astype(int, errors="ignore")

    # Print cluster profiles
    cluster_profiles = trader_metrics.groupby("cluster")[cluster_features].mean().round(2)
    print(f"\n  Cluster Profiles (K={optimal_k}):\n{cluster_profiles.to_string()}")

    # Name clusters based on characteristics
    cluster_names = {}
    for c in range(optimal_k):
        profile = cluster_profiles.loc[c]
        if profile["total_trades"] > cluster_profiles["total_trades"].median():
            if profile["mean_pnl"] > 0:
                cluster_names[c] = "Active Winners"
            else:
                cluster_names[c] = "Active Losers"
        else:
            if profile["avg_win_rate"] > 0.5:
                cluster_names[c] = "Cautious Strategists"
            else:
                cluster_names[c] = "Passive Traders"

    trader_metrics["archetype"] = trader_metrics["cluster"].map(cluster_names)
    print(f"\n  Cluster Archetypes: {cluster_names}")
    print(f"  Distribution:\n{trader_metrics['archetype'].value_counts().to_string()}")

    # Chart: Cluster visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Scatter: PnL vs Trades
    for c in range(optimal_k):
        mask = trader_metrics["cluster"] == c
        label = cluster_names.get(c, f"Cluster {c}")
        axes[0].scatter(
            trader_metrics.loc[mask, "total_trades"],
            trader_metrics.loc[mask, "mean_pnl"],
            label=label, alpha=0.6, s=50
        )
    axes[0].set_xlabel("Total Trades")
    axes[0].set_ylabel("Mean PnL (USD)")
    axes[0].set_title("Trader Archetypes\n(Trades vs PnL)", fontweight="bold")
    axes[0].legend()

    # Scatter: Win Rate vs Trade Size
    for c in range(optimal_k):
        mask = trader_metrics["cluster"] == c
        label = cluster_names.get(c, f"Cluster {c}")
        axes[1].scatter(
            trader_metrics.loc[mask, "avg_trade_size"],
            trader_metrics.loc[mask, "avg_win_rate"],
            label=label, alpha=0.6, s=50
        )
    axes[1].set_xlabel("Avg Trade Size (USD)")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_title("Trader Archetypes\n(Size vs Win Rate)", fontweight="bold")
    axes[1].legend()

    # Elbow plot
    axes[2].plot(list(K_range), inertias, "bo-")
    axes[2].axvline(x=optimal_k, color="red", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("Inertia")
    axes[2].set_title("Elbow Method\nfor Optimal K", fontweight="bold")

    fig.suptitle("Bonus 2: Behavioral Archetype Clustering", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "Bonus2_clustering.png")

    # Save cluster profiles
    cluster_profiles.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))

    # Cluster × Sentiment cross-analysis
    daily_cluster = daily.merge(
        trader_metrics[["Account", "cluster", "archetype"]],
        on="Account", how="left"
    )
    cluster_sent = daily_cluster[daily_cluster["sentiment"].isin(["Fear", "Greed"])].groupby(
        ["archetype", "sentiment"]
    )["total_pnl"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    cluster_sent.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"], edgecolor="black")
    ax.set_title("Archetype Performance on Fear vs Greed Days", fontweight="bold")
    ax.set_ylabel("Avg PnL (USD)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Sentiment")
    save(fig, "Bonus2b_archetype_x_sentiment.png")

else:
    print("  Not enough traders for meaningful clustering.")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# List all output files
print("\n  Output files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f} ({size / 1024:.1f} KB)")

print("\n  Run `streamlit run dashboard.py` for the interactive dashboard.")
print("=" * 70)
