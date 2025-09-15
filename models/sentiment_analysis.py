"""
Sentiment benchmarking utilities for FinancialPhraseBank.

This module encapsulates logic from the Sentiment Analysis notebook:
- Load FinancialPhraseBank text datasets
- Evaluate VADER, FinBERT, DistilBERT on labeled sentences
- Print and return accuracy metrics

Expected dataset location (default): data/external/FinancialPhraseBank-v1.0
Files used: Sentences_AllAgree.txt, Sentences_75Agree.txt, Sentences_66Agree.txt, Sentences_50Agree.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
from transformers import pipeline


def load_and_preprocess_data(file_path: str | Path) -> Tuple[List[str], List[str]]:
    sentences: List[str] = []
    labels: List[str] = []
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit('@', 1)
            if len(parts) == 2:
                sentence = parts[0].strip()
                sentiment = parts[1].strip().lower()
                sentences.append(sentence)
                labels.append(sentiment)
    return sentences, labels


def convert_vader_to_sentiment(compound_score: float) -> str:
    if compound_score >= 0.05:
        return 'positive'
    if compound_score <= -0.05:
        return 'negative'
    return 'neutral'


def evaluate_vader(sentences: Sequence[str], true_labels: Sequence[str]) -> float:
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except Exception:
        try:
            nltk.download('vader_lexicon')
        except Exception:
            pass
    sid = SentimentIntensityAnalyzer()
    predicted = []
    for sentence in sentences:
        scores = sid.polarity_scores(sentence)
        predicted.append(convert_vader_to_sentiment(scores['compound']))
    return float(accuracy_score(true_labels, predicted))


def evaluate_bert_model(model_name: str, sentences: Sequence[str], true_labels: Sequence[str]) -> float:
    clf = pipeline(
        'sentiment-analysis',
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        padding=True,
        max_length=512,
    )
    results = clf(list(sentences))
    predicted = [r['label'].lower() for r in results]
    return float(accuracy_score(true_labels, predicted))


def run_sentiment_benchmark(
    dataset_dir: str | Path,
    focus_filename: str = 'Sentences_AllAgree.txt',
    extra_filenames: Optional[Sequence[str]] = ('Sentences_75Agree.txt',),
    run_vader: bool = True,
    run_finbert: bool = True,
    run_distilbert: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    dataset_dir = Path(dataset_dir)
    files_to_eval = [focus_filename]
    if extra_filenames:
        files_to_eval.extend(list(extra_filenames))

    accuracies: Dict[str, Dict[str, float]] = {}

    for filename in files_to_eval:
        file_path = dataset_dir / filename
        if not file_path.exists():
            if verbose:
                print(f"Warning: dataset file not found: {file_path}")
            continue
        sentences, labels = load_and_preprocess_data(file_path)

        if verbose:
            print(f"\n--- Evaluating models on {filename} ---")

        per_file: Dict[str, float] = {}

        if run_vader:
            try:
                acc = evaluate_vader(sentences, labels)
                per_file['vader'] = acc
                if verbose:
                    print(f"VADER Accuracy: {acc:.4f}")
            except Exception as e:
                if verbose:
                    print(f"VADER evaluation failed: {e}")

        if run_finbert:
            try:
                acc = evaluate_bert_model('ProsusAI/finbert', sentences, labels)
                per_file['finbert'] = acc
                if verbose:
                    print(f"FinBERT Accuracy: {acc:.4f}")
            except Exception as e:
                if verbose:
                    print(f"FinBERT evaluation failed: {e}")

        if run_distilbert:
            try:
                acc = evaluate_bert_model('distilbert-base-uncased-finetuned-sst-2-english', sentences, labels)
                per_file['distilbert'] = acc
                if verbose:
                    print(f"DistilBERT Accuracy: {acc:.4f} (binary model, neutral handling may differ)")
            except Exception as e:
                if verbose:
                    print(f"DistilBERT evaluation failed: {e}")

        accuracies[filename] = per_file

    return accuracies


__all__ = [
    'load_and_preprocess_data',
    'evaluate_vader',
    'evaluate_bert_model',
    'run_sentiment_benchmark',
]


