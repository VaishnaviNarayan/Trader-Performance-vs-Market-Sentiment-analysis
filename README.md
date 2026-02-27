# 📊 Trader Performance vs Market Sentiment

Analyze how Bitcoin market sentiment (Fear/Greed Index) relates to trader behavior and performance on Hyperliquid.

## Project Structure
```
Project6/
├── data/
│   ├── sentiment.csv          # Bitcoin Fear & Greed Index (2,644 rows)
│   └── trader_data.csv        # Hyperliquid trades (211,224 rows)
├── output/                    # Generated charts, tables, JSON
├── analysis.py                # Main analysis script (Parts A, B, C, Bonus)
├── dashboard.py               # Interactive Streamlit dashboard
├── requirements.txt
├── README.md
└── WRITEUP.md                 # Methodology, insights, strategies
```

## Setup

### Prerequisites
- Python 3.10+

### Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** If your C: drive is low on space, install to an alternate directory:
> ```bash
> pip install --target D:\pylibs -r requirements.txt
> ```
> Then set `PYTHONPATH=D:\pylibs` before running scripts.

### Download Datasets
The datasets are already included in `data/`. To re-download:
```bash
pip install gdown
python -c "import gdown; gdown.download(id='1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf', output='data/sentiment.csv')"
python -c "import gdown; gdown.download(id='1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs', output='data/trader_data.csv')"
```

## How to Run

### 1. Run Full Analysis
```bash
python analysis.py
```
Generates all charts, tables, and model outputs in `output/`.

### 2. Launch Dashboard
```bash
streamlit run dashboard.py
```
Opens an interactive dashboard at `http://localhost:8501`.

## Output Files
| File | Description |
|------|-------------|
| `B1_performance_fear_vs_greed.png` | PnL & win rate distributions by sentiment |
| `B2_behavior_fear_vs_greed.png` | Trade frequency, size, L/S ratio by sentiment |
| `B3_trader_segments.png` | Segment analysis (size, frequency, winners) |
| `B3b_segment_x_sentiment.png` | Segment × Sentiment cross-analysis |
| `B4_sentiment_vs_pnl_timeseries.png` | Time series overlay |
| `B4b_correlation_heatmap.png` | Feature correlation matrix |
| `B4c_pnl_by_detailed_classification.png` | PnL by 5-level classification |
| `Bonus1_feature_importance.png` | Random Forest feature importance |
| `Bonus1_confusion_matrix.png` | Confusion matrix |
| `Bonus2_clustering.png` | K-Means behavioral archetypes |
| `Bonus2b_archetype_x_sentiment.png` | Archetype × Sentiment PnL |
| `summary_stats_by_sentiment.csv` | Summary statistics table |
| `cluster_profiles.csv` | Cluster profile metrics |
| `strategies.json` | Data-driven strategy recommendations |

## Key Results
- **Random Forest Accuracy:** 81.25% (5-fold CV: 84.33%)
- **3 Trader Archetypes** identified via K-Means clustering
- **2 Actionable Strategies** derived from data
- See `WRITEUP.md` for full analysis and insights
