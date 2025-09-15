# Financial Time Series Forecasting with Deep Learning and Social Media Sentiment

Clean, reproducible pipelines for forecasting stock prices using ARIMA, Google TimesFM, Amazon Chronos, and Temporal Fusion Transformer (TFT), with optional Reddit sentiment and activity features.

## How to run the code

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Prepare datasets (one-time or as needed)
- Download historical OHLCV and engineer base features (TSLA default):
```bash
python -c "from models.stock_data_extraction import run_stock_data_extraction; run_stock_data_extraction('TSLA')"
```
- Extract Reddit posts and run FinBERT sentiment, mapped to NYSE trading days (edit dates if needed):
```bash
python -c "from models.reddit_sentiment import run_reddit_sentiment; \
__import__('datetime'); from datetime import datetime; \
run_reddit_sentiment(subreddit_name='wallstreetbets', start_date=datetime(2024,6,1), end_date=datetime(2025,7,22), max_posts=2000, \
output_csv='data/interim/activitiy recognition/tesla_sentiment.csv', verbose=True)"
```
- Detect Reddit activity spikes and export features (also plots):
```bash
python -c "from models.activity_timing import run_activity_timing; \
run_activity_timing(input_csv='data/interim/activitiy recognition/tesla_sentiment.csv', \
output_csv='data/interim/activitiy recognition/spike_data.csv', ticker='TSLA', show_plots=True)"
```

3) Run models via notebooks (recommended order)
- `notebooks/0_arima_baseline.ipynb`
- `notebooks/1_timesfm_baseline.ipynb`
- `notebooks/2_chronos_baseline.ipynb`
- `notebooks/3_tft_baseline_runner.ipynb`
- `notebooks/4_tft_with_reddit_sentiment_runner.ipynb`

4) Optional analysis notebooks
- `notebooks/activity_timing_spike_detection.ipynb` (delegates to `models/activity_timing.py`)
- `notebooks/wordcloud_analysis.ipynb` (delegates to `models/wordcloud.py`)
- `notebooks/Sentiment Analysis.ipynb` (VADER vs FinBERT vs DistilBERT benchmark)

## Top Results

Best overall RMSE on TSLA (lower is better), parsed from `results/result_matrix.csv`:

| Model                | MAE  | MSE    | RMSE  |
|----------------------|------:|-------:|------:|
| ARIMA                | 30.08 | 1600.08| 40.00 |
| timesfm              |  7.56 |  104.85| 10.24 |
| chronos              | 10.89 |  303.91| 17.43 |
| TFT_reddit_N         |  9.84 |  130.77| 11.44 |
| **TFT_Reddit_Y**     | **5.99** | **40.72** | **6.38** |

- Winner: **TFT_Reddit_Y** (TFT + Reddit sentiment & spike features) with RMSE ≈ 6.38.
- Metrics are computed on the forecast horizon produced by each notebook runner.

## Project structure

```
├── data/
│   ├── external/                       # External datasets (e.g., FinancialPhraseBank)
│   ├── interim/                        # Intermediate artifacts (sentiment, spikes)
│   ├── processed/                      # Feature-engineered datasets
│   └── TSLA_close.csv                  # Raw/utility series
├── models/
│   ├── activity_timing.py              # Map Reddit posts to trading days, detect spikes
│   ├── reddit_sentiment.py             # Reddit extraction + FinBERT sentiment
│   ├── sentiment_analysis.py           # FinancialPhraseBank benchmark
│   ├── tft_baseline.py                 # TFT (price-only) runner (notebook recommended)
│   ├── tft_with_reddit_sentiment.py    # TFT with sentiment+spike runner (notebook recommended)
│   ├── arima.py                        # ARIMA baseline
│   ├── timesfm.py                      # Google TimesFM baseline
│   ├── chronos.py                      # Amazon Chronos baseline
│   └── wordcloud.py                    # Wordcloud + keyword analytics
├── notebooks/
│   ├── 0_arima_baseline.ipynb
│   ├── 1_timesfm_baseline.ipynb
│   ├── 2_chronos_baseline.ipynb
│   ├── 3_tft_baseline_runner.ipynb
│   ├── 4_tft_with_reddit_sentiment_runner.ipynb
│   ├── activity_timing_spike_detection.ipynb
│   ├── stock_data_extraction.ipynb
│   ├── reddit_tsla_sentiment_extraction.ipynb
│   └── wordcloud_analysis.ipynb
├── results/
│   ├── result_matrix.csv               # Aggregate results summary (top RMSE here)
│   ├── TSLA_results_matrix.pkl         # Persisted performance matrix
│   └── wordcloud.png                   # Example output
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Notes
- Set your Reddit API keys in `.env` as `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` before running Reddit extraction.
- Paths in notebooks call into the corresponding modules so you can rerun end-to-end quickly.

## License
MIT License — see `LICENSE`.