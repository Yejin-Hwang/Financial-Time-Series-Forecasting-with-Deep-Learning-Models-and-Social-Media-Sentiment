"""
BYND end-to-end pipeline:
 - Fetch BYND price history via Yahoo Finance
 - Collect Reddit sentiment for BYND keywords
 - Map posts to NYSE trading day and aggregate by day
 - Merge into a single dataset and save under data/processed

Usage:
  python -m models.bynd_pipeline
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from models.stock_data_extraction import run_stock_data_extraction
from models.reddit_bynd_sentiment import run_reddit_sentiment, BYND_KEYWORDS_DEFAULT
from models.utils_merge import merge_price_sentiment_spike


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data'
    interim_dir = data_dir / 'interim'
    processed_dir = data_dir / 'processed'
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    ticker = 'BYND'

    # 1) Price
    price_csv = interim_dir / f'{ticker}_price_full.csv'
    price_df = run_stock_data_extraction(ticker=ticker, output_csv=str(price_csv))

    # 2) Reddit sentiment (recent 60 days by default)
    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=90)
    senti_csv = interim_dir / f'{ticker.lower()}_reddit_sentiment_{start_dt.date()}_{end_dt.date()}.csv'
    # Try multiple subreddits and concatenate to increase hit rate
    senti_frames = []
    for sub in ['wallstreetbets', 'stocks', 'investing']:
        try:
            df_sub = run_reddit_sentiment(
                subreddit_name=sub,
                start_date=start_dt,
                end_date=end_dt,
                max_posts=1500,
                keywords=BYND_KEYWORDS_DEFAULT,
                output_csv=None,
                verbose=False,
            )
            if df_sub is not None and not df_sub.empty:
                df_sub['source_subreddit'] = sub
                senti_frames.append(df_sub)
        except Exception as e:
            print(f"⚠️ Reddit fetch failed for r/{sub}: {e}")
            continue

    if senti_frames:
        senti_df = pd.concat(senti_frames, ignore_index=True).drop_duplicates(subset=['post_id'])
        try:
            senti_df.to_csv(senti_csv, index=False)
        except Exception:
            pass
    else:
        print("⚠️ No Reddit posts collected across subreddits. Using empty placeholder.")
        senti_df = pd.DataFrame(columns=['date', 'sentiment', 'sentiment_score', 'post_id'])

    # 3) Aggregate sentiment to daily (mean of sentiment_score; majority for label)
    if not senti_df.empty:
        df = senti_df.copy()
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None).dt.floor('D')
        # numerical daily sentiment
        daily = df.groupby('date', as_index=False).agg(
            daily_sentiment=('sentiment_score', 'mean'),
            posts=('post_id', 'count')
        )
        daily_out = interim_dir / f'{ticker.lower()}_reddit_sentiment_daily.csv'
        daily.to_csv(daily_out, index=False)
        sentiment_daily_csv = str(daily_out)
    else:
        # create empty placeholder to allow merge to proceed
        sentiment_daily_csv = str(interim_dir / f'{ticker.lower()}_reddit_sentiment_daily.csv')
        pd.DataFrame(columns=['date', 'daily_sentiment', 'posts']).to_csv(sentiment_daily_csv, index=False)

    # 4) Merge with optional spike file if exists (BYND spike not provided yet)
    spike_csv = None
    out_csv = processed_dir / f'{ticker.lower()}_price_sentiment_spike_merged.csv'
    merged = merge_price_sentiment_spike(
        price_csv=str(price_csv),
        sentiment_csv=sentiment_daily_csv,
        spike_csv=spike_csv,
        out_csv=str(out_csv),
    )

    print(f'✅ BYND pipeline complete. Rows: {len(merged)}')
    print(f'📄 Saved: {out_csv}')


if __name__ == '__main__':
    main()


