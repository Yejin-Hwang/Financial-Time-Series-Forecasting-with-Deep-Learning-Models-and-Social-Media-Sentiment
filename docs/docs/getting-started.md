Getting started
===============

Quickstart (processed data ready)
---------------------------------

If the processed features are already present under `data/processed/` (this repository includes prepared datasets for TSLA/AAPL/NVDA), you can run the notebooks immediately without extracting raw data.

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Open a notebook (recommended order)

- `notebooks/0_arima_baseline.ipynb`
- `notebooks/1_timesfm_baseline.ipynb`
- `notebooks/2_chronos_baseline.ipynb`
- `notebooks/3_tft_baseline_runner.ipynb`
- `notebooks/4_tft_with_reddit_sentiment_runner.ipynb`

3) Configure run parameters in the first cell

- Choose a `training_start_date` within the available data range
- Default context window: 96 trading days
- Default forecast horizon: 5 trading days

4) Run all cells

- Models will train using the specified start date and defaults
- Forecasts and plots will be saved to `results/`

Optional: refresh or extend datasets
------------------------------------

If you need to refresh raw data or extend coverage:

- Historical OHLCV + base features (e.g., TSLA)

```bash
python -c "from models.stock_data_extraction import run_stock_data_extraction; run_stock_data_extraction('TSLA')"
```

- Reddit extraction + FinBERT sentiment (edit dates/ticker/subreddit as needed)

```bash
python -c "from models.reddit_sentiment import run_reddit_sentiment; \
__import__('datetime'); from datetime import datetime; \
run_reddit_sentiment(subreddit_name='wallstreetbets', start_date=datetime(2024,6,1), end_date=datetime(2025,7,22), max_posts=2000, \
output_csv='data/interim/activitiy recognition/tesla_sentiment.csv', verbose=True)"
```

- Reddit activity spikes and features

```bash
python -c "from models.activity_timing import run_activity_timing; \
run_activity_timing(input_csv='data/interim/activitiy recognition/tesla_sentiment.csv', \
output_csv='data/interim/activitiy recognition/spike_data.csv', ticker='TSLA', show_plots=True)"
```

Notes
-----

- Set Reddit API credentials in `.env` as `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` before Reddit extraction.
- Notebooks delegate heavy lifting to `models/` modules, so you can iterate quickly.
