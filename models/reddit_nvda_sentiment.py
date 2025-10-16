"""
Reddit Nvidia sentiment extraction and analysis utilities.

This module encapsulates the logic from `notebooks/reddit_tsla_sentiment_extraction.ipynb`:
- Connect to Reddit via PRAW (env: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)
- Collect posts from a subreddit within a date range using multiple strategies
- Filter by Nvidia-related keywords
- Clean text for CSV
- Analyze sentiment using FinBERT (ProsusAI/finbert)
- Map each post's timestamp to the next NYSE trading day after 4pm ET
- Save results to CSV

Dependencies: praw, pandas, numpy, transformers, torch, python-dotenv, pandas_market_calendars (optional)
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import praw
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


NVIDIA_KEYWORDS_DEFAULT: Tuple[str, ...] = (
    'nvda', 'nvidia', 'jensen huang', 'ada lovelace', 'hopper', 'cuda', 'rtx', 'a100', 'gpu',
    'h100', 'tensor core', 'geforce', 'g-sync', 'nvlink', 'nvswitch', 'omniverse', 'dgx',
    'inference engine', 'l40s', 'blackwell', 'b100', 'poseidon', 'nvsmi', 'jetson', 'dali',
    'v100', 'tesla v100', 'nvidia ai', 'nvidia drive', 'nvidia shield'
)


def load_reddit_from_env() -> praw.Reddit:
    """Create a PRAW Reddit client using .env or environment variables."""
    load_dotenv()
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT') or os.getenv('REDDIT9_USER_AGENT')
    if not client_id or not client_secret or not user_agent:
        raise RuntimeError('Missing Reddit API credentials in environment (.env)')
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)


def clean_text(text: str) -> str:
    """Clean text for CSV: remove HTML, URLs, normalize whitespace and quotes."""
    if text is None:
        return ''
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://[^\s]+', '', text)
    text = text.replace('"', "'")
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^\w\s,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def contains_keywords(text: str, keywords: Sequence[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for kw in keywords:
        if re.search(rf'\b{re.escape(kw.lower())}\b', lowered):
            return True
    return False


def load_finbert_pipeline():
    model_name = 'ProsusAI/finbert'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)
    return nlp, tokenizer


def analyze_sentiment(text: str, nlp, tokenizer) -> Tuple[str, float]:
    if not text or not text.strip():
        return 'neutral', 0.0
    # Truncate to 512 tokens to avoid indexing errors
    tokens = tokenizer.encode(text, truncation=True, max_length=512)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        result = nlp(truncated_text, truncation=True, max_length=512)[0]
        label = result['label'].lower()
        if label not in {'positive', 'negative', 'neutral'}:
            label = 'neutral'
        score = float(result.get('score', 0.0))
        return label, score
    except Exception:
        return 'neutral', 0.0


def collect_posts_by_date_range(
    reddit: praw.Reddit,
    subreddit_name: str,
    start_date: datetime,
    end_date: datetime,
    keywords: Sequence[str] = NVIDIA_KEYWORDS_DEFAULT,
    max_posts: int = 1000,
) -> List[Dict]:
    subreddit = reddit.subreddit(subreddit_name)
    start_ts = start_date.replace(tzinfo=timezone.utc).timestamp() if start_date.tzinfo is None else start_date.timestamp()
    end_ts = end_date.replace(tzinfo=timezone.utc).timestamp() if end_date.tzinfo is None else end_date.timestamp()

    collected: List[Dict] = []
    seen_ids: set[str] = set()

    sorting_methods = (
        ('new', 'newest posts'),
        ('top', 'top posts'),
        ('hot', 'hot posts'),
        ('rising', 'rising posts'),
    )

    per_method = max(1, max_posts // max(len(sorting_methods), 1))
    for sort_method, _desc in sorting_methods:
        try:
            if sort_method == 'top':
                gen = subreddit.top(time_filter='all', limit=per_method)
            elif sort_method == 'new':
                gen = subreddit.new(limit=per_method)
            elif sort_method == 'hot':
                gen = subreddit.hot(limit=per_method)
            else:
                gen = subreddit.rising(limit=per_method)

            for submission in gen:
                created = float(submission.created_utc)
                if created < start_ts or created > end_ts:
                    continue
                title_text = f"{submission.title} {submission.selftext}"
                if not contains_keywords(title_text, keywords):
                    continue
                if submission.id in seen_ids:
                    continue
                seen_ids.add(submission.id)
                collected.append({
                    'id': submission.id,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_utc': created,
                    'author': str(submission.author),
                    'url': submission.url,
                    'permalink': f"https://reddit.com{submission.permalink}",
                    'collection_method': sort_method,
                    'flair': submission.link_flair_text,
                })
        except Exception:
            pass
        time.sleep(1)

    # Keyword search
    try:
        for term in keywords:
            try:
                for submission in subreddit.search(term, sort='new', time_filter='all', limit=100):
                    created = float(submission.created_utc)
                    if created < start_ts or created > end_ts:
                        continue
                    if submission.id in seen_ids:
                        continue
                    seen_ids.add(submission.id)
                    collected.append({
                        'id': submission.id,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'created_utc': created,
                        'author': str(submission.author),
                        'url': submission.url,
                        'permalink': f"https://reddit.com{submission.permalink}",
                        'collection_method': f'search_{term}',
                        'flair': submission.link_flair_text,
                    })
                time.sleep(1)
            except Exception:
                pass
    except Exception:
        pass

    collected.sort(key=lambda x: x['created_utc'], reverse=True)
    return collected[:max_posts]


def analyze_posts(posts: List[Dict]) -> pd.DataFrame:
    if not posts:
        return pd.DataFrame()
    nlp, tokenizer = load_finbert_pipeline()
    records: List[Dict] = []
    for i, post in enumerate(posts):
        text = f"{post.get('title','')} {post.get('selftext','')}"
        cleaned = clean_text(text)
        sentiment, score = analyze_sentiment(cleaned, nlp, tokenizer)
        created_dt = datetime.fromtimestamp(float(post['created_utc']))
        record = {
            'post_id': post['id'],
            'date': created_dt.strftime('%Y-%m-%d'),
            'datetime': created_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'title': post.get('title') or '',
            'text': (cleaned[:200] + '...') if len(cleaned) > 200 else cleaned,
            'reddit_score': post.get('score', 0),
            'upvote_ratio': post.get('upvote_ratio', np.nan),
            'num_comments': post.get('num_comments', 0),
            'author': post.get('author') or '',
            'sentiment': sentiment,
            'sentiment_score': score,
            'permalink': post.get('permalink') or '',
            'collection_method': post.get('collection_method', 'unknown'),
            'flair': post.get('flair'),
        }
        records.append(record)
        if i % 10 == 0:
            time.sleep(0.1)
    return pd.DataFrame.from_records(records)


def map_to_nyse_trading_day(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """Map each row to its impact trading day using NYSE calendar (4pm ET rule)."""
    if datetime_col not in df.columns:
        return df
    # Parse to timezone-aware; assume naive is local; convert to US/Eastern
    dt = pd.to_datetime(df[datetime_col], utc=True, errors='coerce')

    # Build NYSE schedule around the date range
    try:
        import pandas_market_calendars as mcal
        start = (dt.min() - pd.Timedelta(days=5)).date().isoformat()
        end = (dt.max() + pd.Timedelta(days=10)).date().isoformat()
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start, end_date=end)
        business_days = pd.DatetimeIndex(schedule.index).tz_localize('UTC')
        bd_set = set(business_days.date)
    except Exception:
        # Fallback: weekdays only
        rng = pd.date_range(start=dt.min().date() - pd.Timedelta(days=5), end=dt.max().date() + pd.Timedelta(days=10), freq='D')
        bd_set = {d.date() for d in rng if d.weekday() < 5}

    def impact_day(ts: pd.Timestamp) -> datetime.date:
        if pd.isna(ts):
            return pd.NaT
        ts_local = ts.tz_convert('US/Eastern')
        candidate = (ts_local + pd.Timedelta(days=1)).date() if ts_local.hour >= 16 else ts_local.date()
        while candidate not in bd_set:
            candidate = (pd.Timestamp(candidate) + pd.Timedelta(days=1)).date()
        return candidate

    df = df.copy()
    df['date'] = dt.apply(impact_day)
    return df


def run_reddit_sentiment(
    subreddit_name: str = 'wallstreetbets',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_posts: int = 1000,
    keywords: Sequence[str] = NVIDIA_KEYWORDS_DEFAULT,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now(tz=timezone.utc)
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    reddit = load_reddit_from_env()
    posts = collect_posts_by_date_range(
        reddit=reddit,
        subreddit_name=subreddit_name,
        start_date=start_date,
        end_date=end_date,
        keywords=keywords,
        max_posts=max_posts,
    )
    df = analyze_posts(posts)
    if df.empty:
        if output_csv:
            try:
                df.to_csv(output_csv, index=False)
            except Exception:
                pass
        return df

    df = map_to_nyse_trading_day(df, datetime_col='datetime')

    if verbose:
        try:
            print("=" * 60)
            print("ANALYSIS RESULTS")
            print("=" * 60)
            print(f"Total posts analyzed: {len(df)}")
            if start_date and end_date:
                try:
                    sd = start_date.strftime('%Y-%m-%d')
                    ed = end_date.strftime('%Y-%m-%d')
                    print(f"Date range: {sd} to {ed}")
                except Exception:
                    pass
            avg_score = pd.to_numeric(df.get('sentiment_score'), errors='coerce').mean()
            if pd.notna(avg_score):
                print(f"Average sentiment score: {avg_score:.3f}")

            if 'sentiment' in df.columns:
                print("\nSentiment distribution:")
                print(df['sentiment'].value_counts())

            if 'collection_method' in df.columns:
                print("\nCollection method distribution:")
                print(df['collection_method'].value_counts())

            if 'flair' in df.columns:
                print("\nFlair distribution:")
                print(df['flair'].value_counts())

            if 'date' in df.columns:
                print("\nDaily post counts:")
                daily_counts = df['date'].value_counts().sort_index()
                print(daily_counts)

            printable_cols = [c for c in ['title', 'reddit_score', 'sentiment', 'sentiment_score', 'date', 'flair'] if c in df.columns]
            if printable_cols:
                print("\nTop 5 posts by Reddit score:")
                try:
                    top_posts = df.nlargest(5, 'reddit_score')[printable_cols]
                    print(top_posts)
                except Exception:
                    pass
        except Exception:
            pass

    if output_csv:
        try:
            df.to_csv(output_csv, index=False)
        except Exception:
            pass
    return df


__all__ = [
    'run_reddit_sentiment',
    'clean_text',
    'contains_keywords',
    'collect_posts_by_date_range',
    'analyze_posts',
    'map_to_nyse_trading_day',
]


