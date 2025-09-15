"""
Word cloud and keyword analysis utilities for Reddit sentiment data.

This module encapsulates logic from `notebooks/wordcloud_analysis.ipynb` and provides
functional APIs to:

- Filter rows by keyword(s) across text columns and optionally export
- Build a word cloud from text within an optional date range
- Compute top tokens by frequency
- Compute top keywords via TF-IDF

Dependencies: pandas, numpy, wordcloud, scikit-learn, matplotlib (optional for display)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter


DEFAULT_TEXT_COLUMNS: Tuple[str, str] = ("title", "text")


def _normalize_keywords(keywords: Optional[Iterable[str]]) -> List[str]:
    if not keywords:
        return []
    return [str(k).strip().lower() for k in keywords if str(k).strip()]


def filter_rows_by_keywords(
    df: pd.DataFrame,
    text_columns: Sequence[str] = DEFAULT_TEXT_COLUMNS,
    keywords: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return subset of rows where any of the `text_columns` contains a keyword (case-insensitive)."""
    if not keywords:
        return df.head(0).copy()
    norm_keywords = _normalize_keywords(keywords)
    if not norm_keywords:
        return df.head(0).copy()

    # Build boolean mask lazily
    mask = pd.Series(False, index=df.index)
    for col in text_columns:
        if col not in df.columns:
            continue
        series = df[col].astype(str).str.lower()
        col_mask = False
        for kw in norm_keywords:
            col_mask = col_mask | series.str.contains(re.escape(kw), na=False)
        mask = mask | col_mask
    return df.loc[mask].copy()


def build_corpus(
    df: pd.DataFrame,
    text_columns: Sequence[str] = DEFAULT_TEXT_COLUMNS,
) -> List[str]:
    """Combine text columns into a single string per row."""
    parts = []
    for col in text_columns:
        if col in df.columns:
            parts.append(df[col].fillna(''))
        else:
            parts.append(pd.Series([''] * len(df), index=df.index))
    combined = parts[0]
    for s in parts[1:]:
        combined = combined + ' ' + s
    return combined.astype(str).tolist()


def generate_wordcloud(
    text: str,
    extra_stopwords: Optional[Iterable[str]] = None,
    width: int = 800,
    height: int = 400,
    background_color: str = 'white',
) -> WordCloud:
    stopwords = set(STOPWORDS)
    if extra_stopwords:
        stopwords |= set(map(str.lower, extra_stopwords))
    wc = WordCloud(width=width, height=height, background_color=background_color, stopwords=stopwords)
    wc.generate(text)
    return wc


def compute_top_counts(
    text: str,
    extra_stopwords: Optional[Iterable[str]] = None,
    min_token_length: int = 3,
    top_n: int = 20,
) -> List[Tuple[str, int]]:
    stopwords = set(STOPWORDS)
    if extra_stopwords:
        stopwords |= set(map(str.lower, extra_stopwords))
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if len(t) >= min_token_length and t not in stopwords]
    counter = Counter(tokens)
    return counter.most_common(top_n)


def compute_tfidf_keywords(
    documents: Sequence[str],
    max_features: int = 20,
    stop_words: str | None = 'english',
) -> List[str]:
    if not documents:
        return []
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    _ = vectorizer.fit_transform(list(documents))
    return list(vectorizer.get_feature_names_out())


@dataclass
class WordcloudResult:
    filtered_df: pd.DataFrame
    wordcloud: Optional[WordCloud]
    top_counts: List[Tuple[str, int]]
    tfidf_keywords: List[str]


def run_worldcloud(
    input_csv: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_column: str = 'date',
    text_columns: Sequence[str] = DEFAULT_TEXT_COLUMNS,
    keyword_filter: Optional[Iterable[str]] = ("deepseek",),
    output_filtered_csv: Optional[str] = 'deepseek_only.csv',
    show_plot: bool = True,
    save_wordcloud_png: Optional[str] = None,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> WordcloudResult:
    """End-to-end pipeline used by the notebook.

    - Loads CSV
    - Optionally filters rows by [start_date, end_date] on `date_column`
    - Filters rows containing any of `keyword_filter` in `text_columns` and optionally saves
    - Builds a word cloud from the period text
    - Computes top token counts and TF-IDF keywords
    - Optionally shows and/or saves the wordcloud figure
    """
    df = pd.read_csv(input_csv)

    # Normalize and filter by date range if present
    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors='coerce')
        df = df.assign(**{date_column: dates})
        if start_date is not None:
            df = df.loc[df[date_column] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df.loc[df[date_column] <= pd.to_datetime(end_date)]
        df = df.dropna(subset=[date_column])

    # Filter rows by keywords (for export)
    filtered_df = filter_rows_by_keywords(df, text_columns=text_columns, keywords=keyword_filter)
    if output_filtered_csv:
        try:
            filtered_df.to_csv(output_filtered_csv, index=False)
        except Exception:
            pass

    # Build corpus and aggregate to a single string for WC
    corpus = build_corpus(df, text_columns=text_columns)
    text_all = ' '.join(corpus)

    # Compute wordcloud and analytics
    wc = generate_wordcloud(text_all, extra_stopwords=extra_stopwords)
    counts = compute_top_counts(text_all, extra_stopwords=extra_stopwords)
    tfidf = compute_tfidf_keywords(corpus, max_features=20, stop_words='english')

    if save_wordcloud_png or show_plot:
        # Only import matplotlib if needed
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        if save_wordcloud_png:
            try:
                plt.savefig(save_wordcloud_png, bbox_inches='tight')
            except Exception:
                pass
        if show_plot:
            plt.show()
        else:
            plt.close()

    return WordcloudResult(
        filtered_df=filtered_df,
        wordcloud=wc,
        top_counts=counts,
        tfidf_keywords=tfidf,
    )


__all__ = [
    'filter_rows_by_keywords',
    'build_corpus',
    'generate_wordcloud',
    'compute_top_counts',
    'compute_tfidf_keywords',
    'run_worldcloud',
    'WordcloudResult',
]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate word cloud and keywords from sentiment CSV')
    parser.add_argument('--input', required=True, help='Input CSV path')
    parser.add_argument('--start', default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-filtered', default='deepseek_only.csv', help='Filtered CSV output path')
    parser.add_argument('--save-png', default=None, help='Optional PNG path to save the wordcloud')
    parser.add_argument('--no-show', action='store_true', help='Disable displaying the plot')
    args = parser.parse_args()

    run_worldcloud(
        input_csv=args.input,
        start_date=args.start,
        end_date=args.end,
        output_filtered_csv=args.output_filtered,
        save_wordcloud_png=args.save_png,
        show_plot=not args.no_show,
    )




