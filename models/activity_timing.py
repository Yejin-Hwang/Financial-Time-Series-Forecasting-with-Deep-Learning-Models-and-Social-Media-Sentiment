"""
Activity timing and spike detection utilities.

This module encapsulates the logic originally developed in the
`notebooks/activity_timing_spike_detection.ipynb` notebook. It provides a functional API to:

- Load Reddit post data with a `datetime` column
- Map posts to their impact trading day (US/Eastern, after 4pm rolls to next day)
- Aggregate daily post counts
- Smooth with LOWESS and compute a dynamic upper band
- Detect spike presence and intensity
- Optionally download close prices for comparison and plot results

Dependencies: pandas, numpy, matplotlib, statsmodels, scikit-learn, yfinance
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess


def _compute_business_day_set(start_date: pd.Timestamp, end_date: pd.Timestamp) -> set:
    """Return a set of dates that are business days (Mon-Fri, excluding US federal holidays).

    Note: This approximates US market trading days (NYSE/NASDAQ) by excluding
    US federal holidays. It is sufficient for the current use case.
    """
    # Build daily range and holiday list
    all_days = pd.date_range(start=start_date.normalize(), end=end_date.normalize(), freq='D')
    cal = USFederalHolidayCalendar()
    holidays = set(cal.holidays(start=start_date, end=end_date).normalize())

    # Keep weekdays that are not holidays; store as python date objects for fast membership
    business_days = {d.date() for d in all_days if d.weekday() < 5 and d.normalize() not in holidays}
    return business_days


def _map_to_impact_trading_day(
    timestamps_est: pd.Series,
    business_day_set: set,
) -> pd.Series:
    """Map each timezone-aware timestamp (US/Eastern) to its impact trading day.

    Rules:
    - If the timestamp is at/after 16:00 (4pm), roll to the next day
    - If that day is not a business day, roll forward until the next business day
    Returns a pandas Series of python `date` objects.
    """
    def to_trading_day(ts: pd.Timestamp) -> pd.Timestamp.date:
        ts_floor_min = ts.floor('min')
        candidate = (ts_floor_min + pd.Timedelta(days=1)).date() if ts_floor_min.hour >= 16 else ts_floor_min.date()
        while candidate not in business_day_set:
            candidate = (pd.Timestamp(candidate) + pd.Timedelta(days=1)).date()
        return candidate

    return timestamps_est.apply(to_trading_day)


def aggregate_daily_post_counts(df: pd.DataFrame, tz: str = 'US/Eastern') -> pd.DataFrame:
    """Aggregate post counts by impact trading day.

    Expects a column named `datetime` in UTC or any tz; converts to US/Eastern.
    Returns a DataFrame with columns: ['impact_trading_day', 'post_count'].
    """
    if 'datetime' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'datetime' column")

    dt_utc = pd.to_datetime(df['datetime'], utc=True)
    dt_est = dt_utc.dt.tz_convert(tz)

    start = dt_est.min()
    end = dt_est.max() + pd.Timedelta(days=10)
    business_day_set = _compute_business_day_set(start, end)

    impact_trading_day = _map_to_impact_trading_day(dt_est, business_day_set)
    out = (
        impact_trading_day.to_frame(name='impact_trading_day')
        .groupby('impact_trading_day')
        .size()
        .reset_index(name='post_count')
    )
    return out


def smooth_and_threshold(
    daily_counts: pd.DataFrame,
    frac_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply LOWESS smoothing and compute a 95% upper band threshold.

    - Select the LOWESS `frac` by minimizing MSE over `frac_grid`
    - Return (smoothed, loess_upper, best_frac)
    """
    if frac_grid is None:
        frac_grid = np.linspace(0.05, 0.5, 20)

    y = daily_counts['post_count'].to_numpy()
    x = np.arange(len(y))

    mse_values = []
    smoothed_candidates = []
    for frac in frac_grid:
        s = lowess(y, x, frac=frac, return_sorted=False)
        smoothed_candidates.append(s)
        mse_values.append(mean_squared_error(y, s))

    best_index = int(np.argmin(mse_values))
    best_frac = float(frac_grid[best_index])
    smoothed = smoothed_candidates[best_index]

    residuals = y - smoothed
    resid_std = float(np.std(residuals))
    loess_upper = smoothed + 1.96 * resid_std

    return smoothed, loess_upper, best_frac


def detect_spikes(
    daily_counts: pd.DataFrame,
    loess_upper: np.ndarray,
) -> pd.DataFrame:
    """Add spike features to the daily_counts DataFrame.

    Adds columns:
    - spike_presence (0/1)
    - spike_intensity (float, 0 if no spike)
    - loess_upper (float)
    - smoothed (float) if present on the input
    """
    result = daily_counts.copy()
    threshold = loess_upper
    result['spike_presence'] = (result['post_count'].to_numpy() > threshold).astype(int)
    result['spike_intensity'] = np.where(result['spike_presence'] == 1, result['post_count'].to_numpy() - threshold, 0.0)
    result['loess_upper'] = threshold
    return result


def download_close_prices(
    trading_days: pd.Series,
    ticker: str = 'TSLA',
) -> pd.Series:
    """Download close prices for the given ticker and align to provided trading days."""
    start = pd.to_datetime(trading_days.min())
    end = pd.to_datetime(trading_days.max()) + pd.Timedelta(days=1)

    stock = yf.download(ticker, start=start, end=end)
    close = stock['Close'].reindex(pd.to_datetime(trading_days))
    return close


def plot_results(
    daily_counts: pd.DataFrame,
    smoothed: np.ndarray,
    loess_upper: np.ndarray,
    mean_plus_3std: float,
    close_price: Optional[pd.Series] = None,
    save_second_plot_path: Optional[str] = None,
) -> None:
    """Plot daily post counts with smoothing/thresholds, and optional close price overlay."""
    import matplotlib.pyplot as plt

    x_dates = pd.to_datetime(daily_counts['impact_trading_day'])

    # Main plot
    plt.figure(figsize=(12, 5))
    plt.plot(x_dates, daily_counts['post_count'], label='post_count', marker='o')
    plt.plot(x_dates, smoothed, label='LOWESS (best)', color='green')
    plt.plot(x_dates, loess_upper, label='LOWESS 95% upper', linestyle='--', color='orange')
    plt.axhline(mean_plus_3std, color='red', linestyle=':', label='Mean + 3*Std')
    plt.legend()
    plt.title('Spike Detection - Smoothing and Thresholds')
    plt.tight_layout()
    plt.show()

    if close_price is not None:
        fig, ax1 = plt.subplots(figsize=(12, 5))

        color_left = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Post Count', color=color_left)
        ax1.plot(x_dates, daily_counts['post_count'], label='post_count', marker='o', color=color_left)
        ax1.plot(x_dates, smoothed, label='LOWESS (best)', color='green')
        ax1.plot(x_dates, loess_upper, label='LOWESS 95% upper', linestyle='--', color='orange')
        ax1.axhline(mean_plus_3std, color='red', linestyle=':', label='Mean + 3*Std')
        ax1.tick_params(axis='y', labelcolor=color_left)
        # Collect legends from both axes, including closing price on right axis
        handles1, labels1 = ax1.get_legend_handles_labels()

        ax2 = ax1.twinx()
        color_right = 'tab:gray'
        ax2.set_ylabel('Stock Close Price', color=color_right)
        ax2.plot(x_dates, close_price, label='Closing Price', color=color_right, linewidth=1.5, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color_right)

        # Merge legends so the gray line appears as 'Closing Price'
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

        plt.title('Spike Detection & Stock Price')
        fig.tight_layout()
        if save_second_plot_path:
            save_path = Path(save_second_plot_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.show()


def run_activity_timing(
    input_csv: str,
    output_csv: Optional[str] = None,
    ticker: str = 'TSLA',
    timezone: str = 'US/Eastern',
    show_plots: bool = True,
) -> pd.DataFrame:
    """End-to-end pipeline.

    - Reads `input_csv` with a `datetime` column
    - Aggregates post counts by impact trading day
    - Smooths counts and computes thresholds
    - Adds spike features
    - Optionally downloads close prices and plots
    - Saves the result to `output_csv` if provided

    Returns the resulting DataFrame.
    """
    df = pd.read_csv(input_csv)

    daily_counts = aggregate_daily_post_counts(df, tz=timezone)

    # Static threshold for reference
    mean_val = float(daily_counts['post_count'].mean())
    std_val = float(daily_counts['post_count'].std(ddof=0))
    mean_plus_3std = mean_val + 3.0 * std_val

    smoothed, loess_upper, best_frac = smooth_and_threshold(daily_counts)

    result = daily_counts.copy()
    result['smoothed'] = smoothed
    result['loess_upper'] = loess_upper
    result = detect_spikes(result[['impact_trading_day', 'post_count', 'smoothed']], loess_upper)

    close_price = None
    if show_plots or output_csv is None:
        # Safe to download if we will visualize; otherwise skip to save time
        try:
            close_price = download_close_prices(result['impact_trading_day'], ticker=ticker)
        except Exception:
            # Non-fatal: proceed without price overlay
            close_price = None

    if show_plots:
        # Save the second plot (with price overlay) to the results folder
        save_second_plot_path = str(Path('results') / f"{ticker}_activity_timing_spike_price.png")
        plot_results(
            result[['impact_trading_day', 'post_count']],
            smoothed,
            loess_upper,
            mean_plus_3std,
            close_price,
            save_second_plot_path=save_second_plot_path,
        )

    if output_csv:
        result.to_csv(output_csv, index=False)

    return result


__all__ = [
    'aggregate_daily_post_counts',
    'smooth_and_threshold',
    'detect_spikes',
    'download_close_prices',
    'plot_results',
    'run_activity_timing',
]


if __name__ == '__main__':
    # Example CLI usage (adjust paths as needed)
    import argparse

    parser = argparse.ArgumentParser(description='Run activity timing spike detection.')
    parser.add_argument('--input', required=True, help='Path to input CSV with a datetime column')
    parser.add_argument('--output', required=False, default=None, help='Optional output CSV path')
    parser.add_argument('--ticker', required=False, default='TSLA', help='Ticker symbol for price overlay')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    args = parser.parse_args()

    run_activity_timing(
        input_csv=args.input,
        output_csv=args.output,
        ticker=args.ticker,
        show_plots=not args.no_plots,
    )


