"""
Utilities to merge price, Reddit sentiment, and optional spike CSVs anchored on price dates.
"""

from __future__ import annotations

import os
import pandas as pd


def coerce_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None).dt.floor('D')
        return df
    for col in ['Date', 'datetime', 'timestamp', 'created', 'day']:
        if col in df.columns:
            s = df[col]
            dt = pd.to_datetime(s, utc=True, errors='coerce')
            if dt.isna().all() and pd.api.types.is_numeric_dtype(s):
                dt = pd.to_datetime(s, unit='s', utc=True, errors='coerce')
                if dt.isna().all():
                    dt = pd.to_datetime(s, unit='ms', utc=True, errors='coerce')
            if not dt.isna().all():
                df['date'] = dt.dt.tz_localize(None).dt.floor('D')
                return df
    raise ValueError('No date-like column found')


def group_by_date_mean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        return df
    if not df['date'].is_monotonic_increasing or df.duplicated('date').any():
        num_cols = df.select_dtypes(include=['number', 'boolean']).columns.tolist()
        num_cols = [c for c in num_cols if c != 'date']
        agg = {c: 'mean' for c in num_cols}
        for c in df.columns:
            if c not in agg and c != 'date':
                agg[c] = 'first'
        df = df.groupby('date', as_index=False).agg(agg)
    return df


def merge_price_sentiment_spike(price_csv: str, sentiment_csv: str, spike_csv: str | None, out_csv: str) -> pd.DataFrame:
    price = pd.read_csv(price_csv)
    price_date = pd.to_datetime(price.get('Date', price.get('date')), utc=True, errors='coerce').dt.tz_localize(None).dt.floor('D')
    price = price.assign(date=price_date).sort_values('date').reset_index(drop=True)

    senti = pd.read_csv(sentiment_csv)
    senti = coerce_date_col(senti)
    senti = group_by_date_mean_numeric(senti)

    spike = None
    if spike_csv and os.path.exists(spike_csv):
        spike = pd.read_csv(spike_csv)
        spike = coerce_date_col(spike)
        spike = group_by_date_mean_numeric(spike)
        spike = spike.rename(columns={c: (c if c == 'date' else f'spike_{c}') for c in spike.columns})

    merged = price.merge(senti, on='date', how='left', suffixes=('', '_dup'))
    if spike is not None:
        merged = merged.merge(spike, on='date', how='left', suffixes=('', '_dup2'))

    dup_cols = [c for c in merged.columns if c.endswith('_dup') or c.endswith('_dup2')]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    merged.to_csv(out_csv, index=False)
    return merged


__all__ = ['merge_price_sentiment_spike', 'coerce_date_col', 'group_by_date_mean_numeric']



