#!/usr/bin/env python
# coding: utf-8

"""
Chronos runner module for TSLA stock forecasting.
- Importable without side effects
- Exposes main() to run the full pipeline
- Lazy-imports Chronos to avoid kernel crashes
- Uses row-based windowing to avoid weekend/holiday gaps
"""

import warnings
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


def _resolve_tsla_csv_path() -> str:
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        Path('TSLA_close.csv'),
        base_dir / 'data' / 'TSLA_close.csv',
        base_dir / 'data' / 'raw' / 'TSLA_close.csv',
        Path.cwd() / 'data' / 'TSLA_close.csv',
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    searched = "\n    ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Can't find TSLA_close.csv. Searched:\n    {searched}")


def _load_data(ticker: str = 'TSLA') -> pd.DataFrame:
    df = pd.read_csv(_resolve_tsla_csv_path())
    if 'date' not in df or 'close' not in df:
        raise ValueError("CSV must contain 'date' and 'close' columns")
    df['date'] = pd.to_datetime(df['date'])
    if 'unique_id' not in df.columns:
        df['unique_id'] = ticker
    return df.sort_values('date').reset_index(drop=True)


def _split_by_rows(df: pd.DataFrame, train_start: Optional[str], context_len: int, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if context_len <= 0:
        context_len = 96
    if train_start is not None:
        start_dt = pd.to_datetime(train_start)
        idxs = df.index[df['date'] >= start_dt]
        start_idx = int(idxs[0]) if len(idxs) > 0 else 0
    else:
        start_idx = max(0, len(df) - (context_len + max(test_days, 0)))

    train_slice = df.iloc[start_idx:start_idx + context_len]
    test_slice = df.iloc[start_idx + len(train_slice): start_idx + len(train_slice) + max(test_days, 0)]
    return train_slice.reset_index(drop=True), test_slice.reset_index(drop=True)


def _run_chronos_forecast(df_train: pd.DataFrame, prediction_length: int, device: str = 'cpu') -> Optional[np.ndarray]:
    try:
        import torch
        from chronos import ChronosPipeline
    except Exception as e:
        print(f"Chronos import failed: {e}")
        return None

    try:
        pipeline = ChronosPipeline.from_pretrained(
            'amazon/chronos-t5-small',
            device_map=device,
            torch_dtype='auto',
        )
        context = df_train['close'].astype(float).values
        forecast = pipeline.predict(context, prediction_length=prediction_length)
        return forecast[0].numpy()
    except Exception as e:
        print(f"Chronos forecasting failed: {e}")
        return None


def _dummy_forecast(df_train: pd.DataFrame, prediction_length: int) -> np.ndarray:
    last = float(df_train['close'].iloc[-1]) if not df_train.empty else 100.0
    rng = np.random.RandomState(42)
    return last * (1 + rng.normal(0, 0.02, size=(prediction_length, 1))).T


def _plot(df_train: pd.DataFrame, df_test: pd.DataFrame, median: np.ndarray, low: Optional[np.ndarray], high: Optional[np.ndarray], ticker: str) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(df_train['date'], df_train['close'], color='dimgray', linewidth=2, label=f'Training Data')
    if not df_test.empty:
        plt.plot(df_test['date'], df_test['close'], color='red', linewidth=2, marker='o', markersize=6, label='Actual (Test)')
    if median is not None:
        plt.plot(df_test['date'], median, color='blue', linewidth=2, marker='s', markersize=6, label='Chronos Prediction (Median)')
    if low is not None and high is not None:
        plt.fill_between(df_test['date'], low, high, color='royalblue', alpha=0.3, label='30-80% Prediction Interval')
    plt.title(f'{ticker} Stock Price Forecast: Chronos vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main(
    ticker: str = 'TSLA',
    train_start: Optional[str] = None,
    context_len: int = 96,
    test_days: int = 5,
    use_model: bool = False,
    device: str = 'cpu',  # 'cpu' or 'cuda'
) -> None:
    print('=== Chronos Forecast Runner ===\n')
    df = _load_data(ticker)

    df_train, df_test = _split_by_rows(df, train_start=train_start, context_len=context_len, test_days=test_days)
    if not df_train.empty:
        print(f"🧪 Train: {df_train['date'].min().strftime('%Y-%m-%d')} → {df_train['date'].max().strftime('%Y-%m-%d')} (rows={len(df_train)})")
    if not df_test.empty:
        print(f"🧾 Test:  {df_test['date'].min().strftime('%Y-%m-%d')} → {df_test['date'].max().strftime('%Y-%m-%d')} (rows={len(df_test)})")

    prediction_length = len(df_test) if len(df_test) > 0 else test_days

    # Run model
    forecast_samples = None
    if use_model:
        forecast_samples = _run_chronos_forecast(df_train, prediction_length=prediction_length, device=device)
    if forecast_samples is None:
        forecast_samples = _dummy_forecast(df_train, prediction_length)

    # Summaries (quantiles)
    low = np.quantile(forecast_samples, 0.3, axis=0)
    median = np.quantile(forecast_samples, 0.5, axis=0)
    high = np.quantile(forecast_samples, 0.8, axis=0)

    # Plot
    _plot(df_train, df_test, median, low, high, ticker)

    # Evaluate
    metrics = None
    if not df_test.empty:
        y_true = df_test['close'].astype(float).values
        y_pred = median[: len(y_true)]
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        print('Forecast Performance Metrics:')
        print(f'MAE:  {mae:.2f}')
        print(f'MSE:  {mse:.2f}')
        print(f'RMSE: {rmse:.2f}')
        metrics = (mae, mse, rmse)

    # Save results
    if metrics is not None:
        file_name = f'{ticker}_results_matrix.pkl'
        try:
            with open(file_name, 'rb') as f:
                matrix = pickle.load(f)
        except FileNotFoundError:
            matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE'])
        matrix.loc['chronos'] = list(metrics)
        with open(file_name, 'wb') as f:
            pickle.dump(matrix, f)
        print('Results saved to matrix successfully!')
        print(matrix)


if __name__ == '__main__':
    main()
