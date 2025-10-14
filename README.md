# Financial Time Series Forecasting with Deep Learning and Social Media Sentiment

Clean, reproducible pipelines for forecasting stock prices using ARIMA, Google TimesFM, Amazon Chronos, and Temporal Fusion Transformer (TFT), with optional Reddit sentiment and activity features.

## Quickstart (processed data ready)

### Pipeline overview

Add the following image at `docs/pipeline_overview.png` to render the diagram:

![Pipeline Overview](docs/pipeline_overview.png?v=2)

If you already have processed features in `data/processed/` (default repo includes TSLA/AAPL/NVDA), you can skip raw data extraction and run the notebooks directly:

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Open a notebook under `notebooks/` (e.g., `3_tft_baseline_runner.ipynb` or `4_tft_with_reddit_sentiment_runner.ipynb`)

3) When prompted, enter `training_start_date`

   - Enter a date within the displayed available range (the notebook validates this)
   - Default context window: 96 trading days
   - Default forecast horizon: 5 trading days

4) Run all cells to train and generate the forecast/plots. Results are saved to `results/`.

Data extraction (Reddit, Yahoo Finance API) is optional—only needed if you want to refresh or expand the datasets.

## How to run the code

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Prepare datasets (optional — notebooks can handle this)
- Prefer notebooks first: `notebooks/stock_data_extraction.ipynb` and `notebooks/reddit_tsla_sentiment_extraction.ipynb` walk you through extraction and feature building end-to-end.
- If you prefer CLI, use the commands below.
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
- `notebooks/sentiment_analysis.ipynb` (VADER vs FinBERT vs DistilBERT benchmark)

## What is OHLCV?

- Open: period start price
- High: highest price in the period
- Low: lowest price in the period
- Close: period end price
- Volume: traded quantity in the period

The period can be daily, hourly, etc. Some sources also provide Adjusted Close (dividends/splits).

## Top Results

Aggregate results on **TSLA** (lower is better), from `results/result_matrix.csv`:

| Model         |    MAE |     MSE |   RMSE |  MAPE |
|---------------|-------:|--------:|-------:|------:|
| ARIMA         |  18.95 |  371.01 |  19.26 |  6.04 |
| TimesFM       |  23.23 |  583.64 |  24.16 |  7.39 |
| Chronos       |  17.68 |  332.02 |  18.22 |  5.58 |
| TFT_baseline  |  10.49 |  117.57 |  10.84 |  3.33 |
| **TFT_Reddit**|  **4.40** |  **20.19** |  **3.33** |  **0.93** |

- Winner: **TFT_Reddit** (TFT + Reddit sentiment & spike features) with RMSE ≈ 4.49 and MAPE ≈ 1.39%.
- Metrics are computed on each model runner’s forecast horizon.

### Execution Time (seconds)

Parsed from `results/TSLA_execution_time_matrix.csv` (lower is faster; values vary by run/hardware):

| Model                   |  Time (s) |
|-------------------------|----------:|
| ARIMA                   |      6.86 |
| TimesFM                 |     20.48 |
| Chronos                 |      9.72 |
| TFT_baseline            |     33.81 |
| TFT_with_Reddit_Sentiment |     66.98 |

## Results Visualizations

### Model Forecasts (TSLA)

<table>
<tr>
<td align="center" width="50%">
<img src="results/TSLA_ARIMA_forecast.png" width="375">
<br><b>ARIMA</b><br>RMSE: 19.26
</td>
<td align="center" width="50%">
<img src="results/TSLA_TimesFM_forecast.png" width="375">
<br><b>TimesFM</b><br>RMSE: 24.16
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="results/TSLA_Chronos_forecast.png" width="375">
<br><b>Chronos</b><br>RMSE: 18.22
</td>
<td align="center" width="50%">
<img src="results/TSLA_TFT_baseline_forecast.png" width="375">
<br><b>TFT Baseline</b><br>RMSE: 10.84
</td>
</tr>
<tr>
<td align="center" width="50%">
<img src="results/TSLA_TFT_with_reddit_sentiment_forecast_2.png" width="375">
<br><b>TFT with Reddit Sentiment</b><br>RMSE: 4.49 ⭐
</td>
<td align="center" width="50%">
</td>
</tr>
</table>

### TFT Model Interpretability

The Temporal Fusion Transformer (TFT) model provides rich interpretability through attention mechanisms and variable importance analysis. These visualizations help understand how the model makes predictions and which features contribute most to the forecasting performance.

**Key Insights from the Results:**

- **Attention Patterns**: Baseline TFT shows concentrated attention on distant past periods (-80 to -60), while Reddit Sentiment TFT displays more dynamic attention across all time periods, indicating that sentiment data helps the model utilize information from various time horizons.

- **Encoder Importance**: Baseline TFT prioritizes temporal features (`is_month_start`: 50%, `day_of_week`: 15%), while Reddit Sentiment TFT heavily focuses on sentiment features (`daily_sentiment_lag5`: 78%, `spike_presence_sum_3`: ~8%), showing sentiment data dominates the encoder's processing.

- **Variable Importance**: Baseline TFT relies on temporal patterns (`is_month_start`: 24%, `year`: 21%), while Reddit Sentiment TFT shows overwhelming dependence on sentiment volatility (`daily_sentiment_std_14`: 98%), indicating that sentiment data's variability is the strongest predictor for stock price movements.

#### Baseline TFT (Price Features Only)
<table>
<tr>
<td align="center" width="33%">
<img src="results/TSLA_TFT_baseline_Attention.png" width="250">
<br><b>Attention Patterns</b>
</td>
<td align="center" width="33%">
<img src="results/TSLA_TFT_baseline_Encoder.png" width="250">
<br><b>Encoder Visualization</b>
</td>
<td align="center" width="33%">
<img src="results/TSLA_TFT_baseline_variable_importance_20250928_130820.png" width="250">
<br><b>Variable Importance</b>
</td>
</tr>
</table>

#### TFT with Reddit Sentiment
<table>
<tr>
<td align="center" width="33%">
<img src="results/TSLA_TFT_with_reddit_sentiment_Attention.png" width="250">
<br><b>Attention Patterns</b>
</td>
<td align="center" width="33%">
<img src="results/TSLA_TFT_with_reddit_sentiment_Encoder.png" width="250">
<br><b>Encoder Visualization</b>
</td>
<td align="center" width="33%">
<img src="results/TSLA_TFT_sentiment_variable_importance_20250928_142923.png" width="250">
<br><b>Variable Importance</b>
</td>
</tr>
</table>

### Reddit Activity Analysis
![TSLA Activity Timing Spike Detection](results/TSLA_activity_timing_spike_price.png)
*Reddit activity spike detection mapped to TSLA price movements*

## Volume Normalization Experiment (TFT inputs)

We added a log1p(volume) + RobustScaler (fit on the 96-day train window starting 2025-02-01) and used `volume_norm` in TFT runners. Only TFT consumes volume in this project; other models remain univariate on close.


| Model        | RMSE (before) | RMSE (after) | MAPE (before) | MAPE (after) |
|--------------|---------------:|-------------:|--------------:|-------------:|
| TFT_baseline |         10.84  |       10.07  |         3.33  |        3.12  |
| TFT_Reddit   |          4.56  |        4.49  |         1.39  |        0.93  |

- before: `data/processed/tsla_price_sentiment_spike.csv`
- after: `data/processed/tsla_price_sentiment_spike_norm.csv` (preferred if present)
- seed=42, other hyperparameters unchanged.

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
│   ├── sentiment_benchmark_metrics.csv # Sentiment model performance metrics
│   ├── TSLA_results_matrix.pkl         # Persisted performance matrix
│   ├── TSLA_*_forecast.png             # Model forecast visualizations
│   ├── TSLA_TFT_*_Attention.png        # TFT attention pattern visualizations
│   ├── TSLA_TFT_*_Encoder.png          # TFT encoder importance plots
│   ├── TSLA_TFT_*_variable_importance_*.png # TFT variable importance analysis
│   └── TSLA_activity_timing_spike_price.png # Reddit activity spike detection
├── requirements.txt
├── pyproject.toml
└── README.md
```

**Note:** `*` represents wildcard patterns. For example:
- `TSLA_*_forecast.png` includes: `TSLA_ARIMA_forecast.png`, `TSLA_TimesFM_forecast.png`, `TSLA_Chronos_forecast.png`, `TSLA_TFT_baseline_forecast.png`, `TSLA_TFT_with_reddit_sentiment_forecast.png`
- `TSLA_TFT_*_Attention.png` includes: `TSLA_TFT_baseline_Attention.png`, `TSLA_TFT_with_reddit_sentiment_Attention.png`
- `TSLA_TFT_*_variable_importance_*.png` includes all TFT variable importance plots with timestamps

## Notes
- Set your Reddit API keys in `.env` as `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` before running Reddit extraction.
- Paths in notebooks call into the corresponding modules so you can rerun end-to-end quickly.

## Sentiment Benchmark (FinancialPhraseBank)

Model accuracy on FinancialPhraseBank splits (higher is better). Generated by `notebooks/sentiment_analysis.ipynb` and saved to `results/sentiment_benchmark_metrics.csv`.

| Dataset                   |  VADER | FinBERT | DistilBERT |
|---------------------------|-------:|--------:|-----------:|
| Sentences_AllAgree.txt    | 0.5707 |  0.9717 |     0.2584 |
| Sentences_75Agree.txt     | 0.5627 |  0.9473 |     0.2667 |
| Sentences_66Agree.txt     | 0.5563 |  0.9182 |     0.2912 |
| Sentences_50Agree.txt     | 0.5429 |  0.8896 |     0.2992 |

- Selected for inference: **FinBERT** (best across all splits).
- CSV: `results/sentiment_benchmark_metrics.csv` (re-generate by running the notebook end-to-end).


## License
MIT License — see `LICENSE`.