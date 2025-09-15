# deps.py
"""
Centralized dependencies for all models (ARIMA, Chronos, TFT, TimesFM).
Import this file in each model instead of repeating imports.
"""

import warnings
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Core scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Time & date
from time import time
from datetime import date, datetime, timedelta

# Stats & ML
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Statsmodels (ARIMA, diagnostics, stationarity test)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# External forecasting libraries
try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# Deep learning (Chronos, TFT, TimesFM)
import torch

try:
    from chronos import ChronosPipeline
except ImportError:
    ChronosPipeline = None

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torchmetrics import MeanSquaredError, MeanAbsoluteError

try:
    import timesfm
except ImportError:
    timesfm = None


# Export symbols
__all__ = [
    # Utils
    "warnings", "pickle", "Path", "Dict", "Tuple", "Optional", "Any",
    # Core
    "np", "pd", "plt", "mdates", "time", "date", "datetime", "timedelta",
    # Metrics
    "mean_absolute_error", "mean_squared_error",
    # Statsmodels
    "ARIMA", "adfuller", "acorr_ljungbox", "auto_arima",
    # Finance
    "yf",
    # Torch/Deep learning
    "torch", "ChronosPipeline",
    "pl", "EarlyStopping", "LearningRateMonitor", "ModelCheckpoint",
    "TemporalFusionTransformer", "TimeSeriesDataSet", "GroupNormalizer",
    "MeanSquaredError", "MeanAbsoluteError",
    # TimesFM
    "timesfm",
]
