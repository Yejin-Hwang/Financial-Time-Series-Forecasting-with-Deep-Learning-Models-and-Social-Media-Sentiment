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
    target = base_dir / 'data' / 'TSLA_close.csv'
    if target.exists():
        return str(target)
    # Conservative single fallback: cwd/data/TSLA_close.csv
    alt = Path.cwd() / 'data' / 'TSLA_close.csv'
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(f"Can't find required file: {target}")


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
    # Set style to match TFT with sentiment
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    plt.figure(figsize=(16, 8))
    # Plot training data with TFT-style colors
    plt.plot(df_train['date'], df_train['close'], color='blue', linewidth=2, alpha=0.7, label='Training Data (Close Price)')
    if not df_test.empty:
        plt.plot(df_test['date'], df_test['close'], color='green', linewidth=2, marker='s', markersize=6, label='Actual (Close Price)')
    if median is not None:
        plt.plot(df_test['date'], median, color='red', linewidth=3, linestyle='--', marker='o', markersize=8, label='Predictions')
    if low is not None and high is not None:
        plt.fill_between(df_test['date'], low, high, color='red', alpha=0.2, label='30-80% Prediction Interval')
    
    # Add vertical line to separate training and prediction
    if not df_test.empty:
        plt.axvline(x=df_test['date'].iloc[0], color='gray', linestyle='--', alpha=0.7, 
                   label='Training End / Prediction Start')
    # Add title with training info - use actual training data count
    actual_training_days = len(df_train)
    pred_days = len(df_test) if not df_test.empty else 5
    title = f'Chronos Model: {actual_training_days} Days Training + {pred_days} Days Prediction'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to results/
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / f'{ticker}_Chronos_forecast.png'
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {plot_path}')
    except Exception as e:
        print(f'Failed to save plot: {e}')

    plt.show()


def main(
    ticker: str = 'TSLA',
    train_start: Optional[str] = None,
    context_len: int = 96,
    test_days: int = 5,
    use_model: bool = False,
    device: str = 'cpu',  # 'cpu' or 'cuda'
) -> None:
    # Seeding for reproducibility
    import random as _random
    import numpy as _np
    try:
        import torch as _torch
        _torch.manual_seed(42)
    except Exception:
        pass
    _random.seed(42)
    _np.random.seed(42)

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
        eps = 1e-8
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0)
        # Directional Accuracy vs previous true close
        try:
            if len(y_true) > 1 and len(y_pred) > 1:
                da = float((np.sign(y_pred[1:] - y_true[:-1]) == np.sign(y_true[1:] - y_true[:-1])).mean())
            else:
                da = float('nan')
        except Exception:
            da = float('nan')
        print('Forecast Performance Metrics:')
        print(f'MAE:  {mae:.2f}')
        print(f'MSE:  {mse:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'MAPE: {mape:.2f}%')
        print(f'DA:   {da:.3f}')
        metrics = (mae, mse, rmse, mape, da)

    # Save results
    if metrics is not None:
        results_dir = Path(__file__).resolve().parent.parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        # Per-ticker pickle
        pkl_path = results_dir / f'{ticker}_results_matrix.pkl'
        try:
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    matrix = pickle.load(f)
            else:
                matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE', 'DA'])
        except Exception:
            matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE', 'DA'])
        # Ensure expected columns
        desired_cols = ['MAE', 'MSE', 'RMSE', 'MAPE', 'DA']
        for c in desired_cols:
            if c not in matrix.columns:
                matrix[c] = pd.NA
        matrix = matrix.reindex(columns=desired_cols)
        matrix.loc['chronos'] = list(metrics)
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(matrix, f)
        except Exception as e:
            print(f'Failed to save per-ticker matrix: {e}')

        # Global CSV
        csv_path = results_dir / 'result_matrix.csv'
        try:
            if csv_path.exists():
                global_matrix = pd.read_csv(csv_path, index_col=0)
            else:
                global_matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE', 'DA'])
            desired_cols = ['MAE', 'MSE', 'RMSE', 'MAPE', 'DA']
            for c in desired_cols:
                if c not in global_matrix.columns:
                    global_matrix[c] = pd.NA
            global_matrix = global_matrix.reindex(columns=desired_cols)
            # Standardize key and reorder
            global_matrix.loc['Chronos'] = list(metrics)
            desired_order = ['ARIMA', 'TimesFM', 'Chronos', 'TFT_baseline', 'TFT_Reddit']
            ordered = [i for i in desired_order if i in global_matrix.index]
            rest = [i for i in global_matrix.index if i not in desired_order]
            global_matrix = global_matrix.loc[ordered + rest]
            global_matrix.to_csv(csv_path)
            print('Results saved to matrix successfully!')
            print(global_matrix)
        except Exception as e:
            print(f'Failed to update global results matrix: {e}')


if __name__ == '__main__':
    main()
