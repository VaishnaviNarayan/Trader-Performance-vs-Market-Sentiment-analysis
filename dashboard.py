"""
=============================================================================
Streamlit Dashboard — Trader Performance vs Market Sentiment
=============================================================================
Launch:  streamlit run dashboard.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, json

st.set_page_config(
    page_title="Trader × Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    sentiment = pd.read_csv("data/sentiment.csv")
    trader = pd.read_csv("data/trader_data.csv")

    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment["sentiment"] = sentiment["classification"].apply(
        lambda x: "Fear" if "Fear" in x else ("Greed" if "Greed" in x else "Neutral")
    )

    trader["datetime"] = pd.to_datetime(trader["Timestamp"], unit="ms")
    trader["date"] = trader["datetime"].dt.normalize()

    daily_pnl = trader.groupby(["Account", "date"]).agg(
        total_pnl=("Closed PnL", "sum"),
        num_trades=("Closed PnL", "count"),
        avg_trade_size_usd=("Size USD", "mean"),
        total_volume_usd=("Size USD", "sum"),
        total_fee=("Fee", "sum"),
    ).reset_index()
    daily_pnl["net_pnl"] = daily_pnl["total_pnl"] - daily_pnl["total_fee"].abs()

    def calc_win_rate(group):
        wins = (group["Closed PnL"] > 0).sum()
        return wins / len(group) if len(group) > 0 else 0

    wr = trader.groupby(["Account", "date"]).apply(calc_win_rate, include_groups=False).reset_index(name="win_rate")
    daily_pnl = daily_pnl.merge(wr, on=["Account", "date"], how="left")

    side_counts = trader.groupby(["Account", "date", "Side"]).size().unstack(fill_value=0).reset_index()
    for c in ["BUY", "SELL"]:
        if c not in side_counts.columns:
            side_counts[c] = 0
    side_counts.rename(columns={"BUY": "long_count", "SELL": "short_count"}, inplace=True)
    side_counts["long_short_ratio"] = side_counts["long_count"] / side_counts["short_count"].replace(0, np.nan)
    side_counts["long_short_ratio"] = side_counts["long_short_ratio"].fillna(side_counts["long_count"])

    daily_pnl = daily_pnl.merge(
        side_counts[["Account", "date", "long_count", "short_count", "long_short_ratio"]],
        on=["Account", "date"], how="left"
    )

    daily = daily_pnl.merge(
        sentiment[["date", "value", "classification", "sentiment"]],
        on="date", how="inner",
    )

    return sentiment, trader, daily


sentiment, trader, daily = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🎛️ Filters")
sent_filter = st.sidebar.multiselect(
    "Sentiment", ["Fear", "Greed", "Neutral"],
    default=["Fear", "Greed"]
)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(daily["date"].min(), daily["date"].max()),
    min_value=daily["date"].min(),
    max_value=daily["date"].max(),
)

filtered = daily[
    (daily["sentiment"].isin(sent_filter)) &
    (daily["date"] >= pd.Timestamp(date_range[0])) &
    (daily["date"] <= pd.Timestamp(date_range[1]))
]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📊 Trader Performance vs Market Sentiment")
st.markdown("*Analyzing Hyperliquid trader behavior against Bitcoin Fear & Greed Index*")

# ── KPIs ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Trades (filtered)", f"{filtered['num_trades'].sum():,.0f}")
c2.metric("Avg Daily PnL", f"${filtered['total_pnl'].mean():,.0f}")
c3.metric("Avg Win Rate", f"{filtered['win_rate'].mean():.1%}")
c4.metric("Unique Traders", f"{filtered['Account'].nunique()}")

st.divider()

# ── Tab Layout ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Performance", "🔄 Behavior", "👥 Segments", "🤖 Model & Clusters"
])

with tab1:
    st.subheader("Performance: Fear vs Greed")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            filtered, x="sentiment", y="total_pnl",
            color="sentiment",
            color_discrete_map={"Fear": "#e74c3c", "Greed": "#2ecc71", "Neutral": "#95a5a6"},
            title="Daily PnL Distribution by Sentiment"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.violin(
            filtered, x="sentiment", y="win_rate",
            color="sentiment",
            color_discrete_map={"Fear": "#e74c3c", "Greed": "#2ecc71", "Neutral": "#95a5a6"},
            title="Win Rate Distribution by Sentiment",
            box=True, points="all"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Time series
    market_daily = filtered.groupby("date").agg(
        avg_pnl=("total_pnl", "mean"),
        total_trades=("num_trades", "sum"),
    ).reset_index().merge(
        sentiment[["date", "value", "sentiment"]], on="date", how="inner"
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=market_daily["date"], y=market_daily["avg_pnl"],
                   name="Avg PnL", fill="tozeroy", opacity=0.5),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=market_daily["date"], y=market_daily["value"],
                   name="Sentiment Score", line=dict(color="orange")),
        secondary_y=True
    )
    fig.update_layout(title="Sentiment Score vs Average Daily PnL Over Time")
    fig.update_yaxes(title_text="Avg PnL (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Fear & Greed Index", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Summary Statistics")
    summary = filtered.groupby("sentiment").agg(
        Mean_PnL=("total_pnl", "mean"),
        Median_PnL=("total_pnl", "median"),
        Std_PnL=("total_pnl", "std"),
        Mean_Win_Rate=("win_rate", "mean"),
        Count=("total_pnl", "count"),
    ).round(2)
    st.dataframe(summary, use_container_width=True)

with tab2:
    st.subheader("Behavioral Changes by Sentiment")
    col1, col2 = st.columns(2)

    with col1:
        behavior = filtered.groupby("sentiment").agg(
            avg_trades=("num_trades", "mean"),
            avg_size=("avg_trade_size_usd", "mean"),
        ).reset_index()
        fig = px.bar(behavior, x="sentiment", y="avg_trades",
                     color="sentiment",
                     color_discrete_map={"Fear": "#e74c3c", "Greed": "#2ecc71", "Neutral": "#95a5a6"},
                     title="Avg Trades per Day by Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(behavior, x="sentiment", y="avg_size",
                     color="sentiment",
                     color_discrete_map={"Fear": "#e74c3c", "Greed": "#2ecc71", "Neutral": "#95a5a6"},
                     title="Avg Trade Size (USD) by Sentiment")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        ls = filtered.groupby("sentiment")["long_short_ratio"].mean().reset_index()
        fig = px.bar(ls, x="sentiment", y="long_short_ratio",
                     color="sentiment",
                     color_discrete_map={"Fear": "#e74c3c", "Greed": "#2ecc71", "Neutral": "#95a5a6"},
                     title="Avg Long/Short Ratio by Sentiment")
        fig.add_hline(y=1, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        vol = filtered.groupby("sentiment")["total_volume_usd"].mean().reset_index()
        fig = px.bar(vol, x="sentiment", y="total_volume_usd",
                     color="sentiment",
                     color_discrete_map={"Fear": "#e74c3c", "Greed": "#2ecc71", "Neutral": "#95a5a6"},
                     title="Avg Daily Volume (USD) by Sentiment")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Trader Segments")

    trader_metrics = daily.groupby("Account").agg(
        total_pnl=("total_pnl", "sum"),
        mean_pnl=("total_pnl", "mean"),
        total_trades=("num_trades", "sum"),
        avg_trade_size=("avg_trade_size_usd", "mean"),
        avg_win_rate=("win_rate", "mean"),
        active_days=("date", "nunique"),
    ).reset_index()

    median_size = trader_metrics["avg_trade_size"].median()
    trader_metrics["Size Segment"] = np.where(
        trader_metrics["avg_trade_size"] > median_size, "High Size", "Low Size"
    )
    median_trades = trader_metrics["total_trades"].median()
    trader_metrics["Frequency"] = np.where(
        trader_metrics["total_trades"] > median_trades, "Frequent", "Infrequent"
    )
    trader_metrics["Winner?"] = np.where(
        trader_metrics["avg_win_rate"] > 0.5, "Winner", "Inconsistent"
    )

    seg_choice = st.selectbox("Select Segment", ["Size Segment", "Frequency", "Winner?"])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            trader_metrics.groupby(seg_choice)["total_pnl"].mean().reset_index(),
            x=seg_choice, y="total_pnl",
            color=seg_choice, title=f"Avg Total PnL by {seg_choice}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            trader_metrics.groupby(seg_choice)["avg_win_rate"].mean().reset_index(),
            x=seg_choice, y="avg_win_rate",
            color=seg_choice, title=f"Avg Win Rate by {seg_choice}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment × Sentiment Interaction")
    daily_seg = daily.merge(trader_metrics[["Account", seg_choice]], on="Account", how="left")
    seg_sent = daily_seg[daily_seg["sentiment"].isin(["Fear", "Greed"])].groupby(
        [seg_choice, "sentiment"]
    )["total_pnl"].mean().reset_index()
    fig = px.bar(seg_sent, x=seg_choice, y="total_pnl", color="sentiment",
                 barmode="group",
                 color_discrete_map={"Fear": "#e74c3c", "Greed": "#2ecc71"},
                 title=f"Avg PnL: {seg_choice} × Sentiment")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("🤖 Predictive Model Results")
    if os.path.exists("output/Bonus1_feature_importance.png"):
        col1, col2 = st.columns(2)
        with col1:
            st.image("output/Bonus1_feature_importance.png", caption="Feature Importance (Random Forest)")
        with col2:
            st.image("output/Bonus1_confusion_matrix.png", caption="Confusion Matrix")
    else:
        st.info("Run analysis.py first to generate model outputs.")

    st.subheader("👥 Trader Archetypes (K-Means Clustering)")
    if os.path.exists("output/Bonus2_clustering.png"):
        st.image("output/Bonus2_clustering.png", caption="Behavioral Archetype Clustering")
        if os.path.exists("output/cluster_profiles.csv"):
            profiles = pd.read_csv("output/cluster_profiles.csv")
            st.dataframe(profiles, use_container_width=True)
    if os.path.exists("output/Bonus2b_archetype_x_sentiment.png"):
        st.image("output/Bonus2b_archetype_x_sentiment.png", caption="Archetype Performance by Sentiment")

    st.subheader("📌 Strategy Recommendations")
    if os.path.exists("output/strategies.json"):
        with open("output/strategies.json") as f:
            strategies = json.load(f)
        for name, data in strategies.items():
            st.markdown(f"**{name}**")
            st.info(data["rule"])
    else:
        st.info("Run analysis.py first to generate strategies.")
