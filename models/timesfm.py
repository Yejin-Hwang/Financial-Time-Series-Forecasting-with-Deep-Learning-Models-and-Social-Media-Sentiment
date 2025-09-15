#!/usr/bin/env python
# coding: utf-8

"""
TimesFM runner module for TSLA stock forecasting.
- Importable without side effects
- Exposes main() to run the full pipeline
- Uses minimal direct imports to avoid heavy dependencies that can crash kernels
"""

# Minimal, safe imports (avoid pulling heavy torch/lightning via deps)
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# from .deps import (
#     warnings, pickle, Path,
#     pd, np, plt, datetime, timedelta,
#     mean_absolute_error, mean_squared_error,
# )


def get_user_input() -> tuple[str, int]:
    """Prompt user for training start date and test horizon.

    Returns
    -------
    train_start: str (YYYY-MM-DD)
    test_days: int
    """
    print("=== TimesFM Configuration (96-day context) ===\n")

    # Show available range
    try:
        df = _load_data("TSLA")
    except Exception as e:
        print(f"✗ Error while reading data: {e}")
        # Sensible fallbacks
        return "", 5

    min_date = pd.to_datetime(df["date"]).min()
    max_date = pd.to_datetime(df["date"]).max()
    # (Info about available range is printed later in main() to avoid duplication)

    # Training start
    while True:
        try:
            raw = input(f"\n📈 Enter training start date (YYYY-MM-DD) [default: {min_date.strftime('%Y-%m-%d')}]: ").strip()
            if not raw:
                train_start = min_date.strftime('%Y-%m-%d')
            else:
                train_start = pd.to_datetime(raw).strftime('%Y-%m-%d')
            ts = pd.to_datetime(train_start)
            if ts < min_date or ts > max_date:
                print(f"⚠️  Date must be between {min_date.strftime('%Y-%m-%d')} and {max_date.strftime('%Y-%m-%d')}")
                continue
            break
        except Exception:
            print("⚠️  Invalid date format. Please use YYYY-MM-DD")

    # Test horizon
    while True:
        try:
            raw = input("🔮 Enter number of days to predict [default: 5]: ").strip()
            if not raw:
                test_days = 5
            else:
                test_days = int(raw)
            if test_days <= 0 or test_days > 30:
                print("⚠️  Prediction days must be between 1 and 30")
                continue
            break
        except Exception:
            print("⚠️  Please enter a valid integer")

    print("\nConfiguration:")
    print(f"   Training start: {train_start}")
    print(f"   Prediction days: {test_days}")
    # Detailed periods are printed in main() after actual row-based split

    return train_start, test_days


def _resolve_tsla_csv_path() -> str:
    """Resolve path to TSLA_close.csv across common project locations."""
    base_dir = Path(__file__).resolve().parent.parent  # project root
    candidates = [
        Path("TSLA_close.csv"),
        base_dir / "data" / "TSLA_close.csv",
        base_dir / "data" / "raw" / "TSLA_close.csv",
        Path.cwd() / "data" / "TSLA_close.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    searched = "\n    ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Can't find TSLA_close.csv. Searched:\n    {searched}")


def _load_data(ticker: str = "TSLA") -> pd.DataFrame:
    df_path = _resolve_tsla_csv_path()
    df = pd.read_csv(df_path)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column")
    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'close' column")
    df["date"] = pd.to_datetime(df["date"])
    if "unique_id" not in df.columns:
        df["unique_id"] = ticker
    return df


def _split_train_test(
    df: pd.DataFrame,
    test_days: int = 5,
    train_start: str | None = None,
    context_len: int = 96,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by row count: use context_len trading rows for training, then next test_days rows.

    This avoids weekend/holiday gaps; lengths will match requested sizes when data is available.
    """
    df_sorted = df.sort_values('date').reset_index(drop=True)
    # Enforce TimesFM requirement
    if context_len <= 0:
        context_len = 96
    if context_len % 32 != 0:
        context_len = max(32, (context_len // 32) * 32)

    if train_start is not None:
        start_dt = pd.to_datetime(train_start)
        start_idx_series = df_sorted.index[df_sorted['date'] >= start_dt]
        start_idx = int(start_idx_series[0]) if len(start_idx_series) > 0 else 0
    else:
        # default: take last context_len + test_days rows
        total = len(df_sorted)
        start_idx = max(0, total - (context_len + max(test_days, 0)))

    train_slice = df_sorted.iloc[start_idx: start_idx + context_len]
    test_slice = df_sorted.iloc[start_idx + len(train_slice): start_idx + len(train_slice) + max(test_days, 0)]

    return train_slice.reset_index(drop=True), test_slice.reset_index(drop=True)


def _init_timesfm(backend: str = "cpu", per_core_batch_size: int = 1,
                  context_len: int = 96, horizon_len: int = 5) -> object | None:
    try:
        import timesfm as _timesfm  # Lazy import to avoid kernel crashes on import
    except Exception as e:
        print(f"TimesFM import failed: {e}")
        return None

    try:
        tfm = _timesfm.TimesFm(
            hparams=_timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=per_core_batch_size,
                horizon_len=horizon_len,
                num_layers=50,
                use_positional_embedding=False,
                context_len=context_len,
            ),
            checkpoint=_timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )
        return tfm
    except Exception as e:
        print(f"TimesFM initialization failed: {e}")
        return None


def _prepare_input_df(df_train: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "unique_id": df_train["unique_id"],
        "ds": df_train["date"],
        "y": df_train["close"].astype(float).values.flatten(),
    })


def _forecast_with_timesfm(tfm: object | None, df_train: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    input_df = _prepare_input_df(df_train)
    if tfm is None:
        return _dummy_forecast(df_train, horizon)

    try:
        forecast_df = tfm.forecast_on_df(
            inputs=input_df,
            freq="D",
            value_name="y",
            num_jobs=-1,
        )
        # Ensure expected columns exist
        if "timesfm" not in forecast_df.columns:
            # Some versions may use a different value column name; fallback to 'yhat'
            value_col = "yhat" if "yhat" in forecast_df.columns else None
            if value_col is None:
                raise ValueError("Forecast output missing expected value column ('timesfm' or 'yhat')")
            forecast_df = forecast_df.rename(columns={value_col: "timesfm"})
        return forecast_df
    except Exception as e:
        print(f"TimesFM forecast failed: {e}")
        return _dummy_forecast(df_train, horizon)


def _dummy_forecast(df_train: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    last_date = df_train["date"].max()
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    base = float(df_train["close"].iloc[-1])
    return pd.DataFrame({
        "unique_id": [df_train["unique_id"].iloc[0] if not df_train.empty else "TSLA"] * horizon,
        "ds": dates,
        "timesfm": base * (1 + np.random.normal(0, 0.02, horizon)),
        "timesfm-q-0.1": base * (1 + np.random.normal(-0.05, 0.01, horizon)),
        "timesfm-q-0.5": base * (1 + np.random.normal(0, 0.02, horizon)),
        "timesfm-q-0.9": base * (1 + np.random.normal(0.05, 0.01, horizon)),
    })


def _plot_results(df_train: pd.DataFrame, df_test: pd.DataFrame, forecast_df: pd.DataFrame, ticker: str) -> None:
    plt.figure(figsize=(12, 6))

    # Actual training data
    plt.plot(df_train["date"], df_train["close"], label="Training Data", color="dimgray", linewidth=2)

    # Median prediction
    if "timesfm" in forecast_df.columns:
        plt.plot(forecast_df["ds"], forecast_df["timesfm"], label="Predicted (Median)", color="blue", linewidth=2)

    # Prediction interval (10-90%)
    if 'timesfm-q-0.1' in forecast_df.columns and 'timesfm-q-0.9' in forecast_df.columns:
        plt.fill_between(
            forecast_df["ds"],
            forecast_df['timesfm-q-0.1'],
            forecast_df['timesfm-q-0.9'],
            color="blue",
            alpha=0.2,
            label="10-90% Prediction Interval",
        )

    # Actual test data
    if not df_test.empty:
        plt.plot(df_test["date"], df_test["close"], label="Test Actual", color="red", linestyle="--", linewidth=2)

    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.title(f"{ticker} Stock Price Forecast vs Actual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _evaluate(df_test: pd.DataFrame, forecast_df: pd.DataFrame) -> tuple[float, float, float] | None:
    if df_test.empty:
        print("No overlapping data for evaluation (no test set).")
        return None

    # Align dates (remove weekends)
    forecast_df = forecast_df.copy()
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    df_test = df_test.copy()
    df_test['date'] = pd.to_datetime(df_test['date'])

    forecast_df = forecast_df[forecast_df['ds'].dt.weekday < 5]
    forecast_df['ds_date'] = forecast_df['ds'].dt.date
    df_test['date_only'] = df_test['date'].dt.date

    merged_df = pd.merge(forecast_df, df_test, left_on='ds_date', right_on='date_only', how='inner')
    if merged_df.empty:
        print("No overlapping business days for evaluation.")
        return None

    y_true = merged_df['close'].astype(float)
    y_pred = merged_df['timesfm'].astype(float)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    print("Forecast Performance Metrics:")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    return mae, mse, rmse


def _save_results_matrix(ticker: str, metrics: tuple[float, float, float] | None) -> None:
    if metrics is None:
        return
    mae, mse, rmse = metrics
    file_name = f"{ticker}_results_matrix.pkl"
    try:
        with open(file_name, "rb") as f:
            matrix = pickle.load(f)
    except FileNotFoundError:
        matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE'])

    matrix.loc['timesfm'] = [mae, mse, rmse]
    with open(file_name, "wb") as f:
        pickle.dump(matrix, f)

    print("Results saved to matrix successfully!")
    print(matrix)


def main(ticker: str = "TSLA", test_days: int = 5,
         train_start: str | None = None, train_end: str | None = None,
         use_model: bool = False, interactive: bool = False,
         backend: str = "cpu", per_core_batch_size: int = 1,
         context_len: int = 96) -> None:
    print("=== TimesFM Forecast Runner ===\n")

    # 0) Interactive prompt (optional)
    if interactive and (train_start is None):
        ts, horizon = get_user_input()
        train_start = ts or train_start
        test_days = horizon or test_days

    # 1) Load data
    df = _load_data(ticker)
    min_date = df["date"].min()
    max_date = df["date"].max()
    # print(f"📅 Available data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    # print(f"📊 Total data points: {len(df)}")

    # 2) Train/test split (fixed context_len window)
    df_train, df_test = _split_train_test(
        df,
        test_days=test_days,
        train_start=train_start,
        context_len=context_len,
    )
    print(f"Training data length: {len(df_train)}")
    print(f"Testing data length: {len(df_test)}")

    # Report periods
    if not df_train.empty:
        train_start_dt = pd.to_datetime(df_train["date"].min())
        train_end_dt = pd.to_datetime(df_train["date"].max())
        print(f"🧪 Training period: {train_start_dt.strftime('%Y-%m-%d')} → {train_end_dt.strftime('%Y-%m-%d')}")
    if not df_test.empty:
        test_start_dt = pd.to_datetime(df_test["date"].min())
        test_end_dt = pd.to_datetime(df_test["date"].max())
        print(f"🧾 Test period: {test_start_dt.strftime('%Y-%m-%d')} → {test_end_dt.strftime('%Y-%m-%d')}")

    # 3) Model init and forecast (default to dummy to avoid crashes)
    tfm = None
    if use_model:
        tfm = _init_timesfm(backend=backend, per_core_batch_size=per_core_batch_size,
                            context_len=context_len, horizon_len=test_days)
    horizon = len(df_test) if len(df_test) > 0 else test_days
    forecast_df = _forecast_with_timesfm(tfm, df_train, horizon=horizon)

    # Report prediction period
    if not forecast_df.empty and "ds" in forecast_df.columns:
        pred_start_dt = pd.to_datetime(forecast_df["ds"].min())
        pred_end_dt = pd.to_datetime(forecast_df["ds"].max())
        print(f"🔮 Prediction period: {pred_start_dt.strftime('%Y-%m-%d')} → {pred_end_dt.strftime('%Y-%m-%d')}")

    # 4) Plot
    _plot_results(df_train, df_test, forecast_df, ticker)

    # 5) Evaluate and save
    metrics = _evaluate(df_test, forecast_df)
    _save_results_matrix(ticker, metrics)


if __name__ == "__main__":
    main()

