# Financial Time Series Forecasting

## Overview

Clean, reproducible pipelines for forecasting stock prices using ARIMA, Google TimesFM, Amazon Chronos, and Temporal Fusion Transformer (TFT), with optional Reddit sentiment and activity features.

## Quickstart

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Open a notebook (e.g., `notebooks/3_tft_baseline_runner.ipynb` or `notebooks/4_tft_with_reddit_sentiment_runner.ipynb`)

3) Set `training_start_date` within available range, then run all cells

- Default context: 96 trading days
- Forecast horizon: 5 trading days
- Outputs written to `results/`

### Pipeline overview

![Pipeline Overview](../assets/pipeline-overview.png)

## Optional data refresh

If you want to refresh data from sources:

- OHLCV + features (e.g., TSLA)

```bash
python -c "from models.stock_data_extraction import run_stock_data_extraction; run_stock_data_extraction('TSLA')"
```

- Reddit sentiment

```bash
python -c "from models.reddit_sentiment import run_reddit_sentiment; \
__import__('datetime'); from datetime import datetime; \
run_reddit_sentiment(subreddit_name='wallstreetbets', start_date=datetime(2024,6,1), end_date=datetime(2025,7,22), max_posts=2000, \
output_csv='data/interim/activitiy recognition/tesla_sentiment.csv', verbose=True)"
```

- Activity spikes

```bash
python -c "from models.activity_timing import run_activity_timing; \
run_activity_timing(input_csv='data/interim/activitiy recognition/tesla_sentiment.csv', \
output_csv='data/interim/activitiy recognition/spike_data.csv', ticker='TSLA', show_plots=True)"
```

See Getting Started for details.


