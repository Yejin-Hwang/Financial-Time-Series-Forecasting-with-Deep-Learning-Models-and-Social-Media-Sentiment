#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated TFT (Temporal Fusion Transformer) Stock Price Forecasting

This script provides a clean, automated implementation of TFT for stock price prediction.

Features:
- User-configurable training period (default: 90 days)
- 5-day prediction horizon
- Automated data loading and preprocessing
- Model training with early stopping
- Performance evaluation and visualization
- Variable importance analysis
"""

import sys

# =============================================================================
# 1. Environment Setup and Dependencies
# =============================================================================

def install_dependencies():
    """Install required packages if not already installed"""
    try:
        import pytorch_forecasting
        print("✓ pytorch_forecasting already installed")
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "pytorch-forecasting==1.0.0", "lightning==2.0.9", "torchmetrics", "--quiet"])
        print("✓ Installation complete!")

    # Install additional dependencies if needed
    try:
        import yfinance
        print("✓ yfinance already installed")
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.run(["pip", "install", "yfinance", "--quiet"])
        print("✓ yfinance installation complete!")

# =============================================================================
# 2. Import Libraries
# =============================================================================

def import_libraries():
    """Import all required libraries"""
    import pandas as pd
    import numpy as np
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from torch.utils.data import DataLoader
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from torchmetrics import MeanSquaredError, MeanAbsoluteError
    import matplotlib.pyplot as plt
    import warnings
    import pickle
    from datetime import datetime, timedelta
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("✓ All libraries imported successfully!")
    
    return (pd, np, torch, pl, Trainer, EarlyStopping, LearningRateMonitor, 
            ModelCheckpoint, DataLoader, TimeSeriesDataSet, TemporalFusionTransformer,
            GroupNormalizer, MeanSquaredError, MeanAbsoluteError, plt, 
            warnings, pickle, datetime, timedelta)

# =============================================================================
# 3. Configuration and User Input
# =============================================================================

def get_user_config():
    """Get user configuration for training and prediction"""
    print("\n=== TFT Configuration ===")
    
    # Training period: start date only, fixed 96-day window (based on data rows)
    from datetime import datetime, timedelta
    try:
        start_date_str = input("Enter start date (YYYY-MM-DD, e.g., 2023-01-01): ").strip()
    except EOFError:
        # Auto-use default date when running in non-interactive mode
        start_date_str = "2025-02-01"
        print(f"Using default start date: {start_date_str}")
    
    training_days = 96
    config = {
        'training_type': 'date_anchor',
        'train_start': start_date_str,
        'training_days': training_days,
        'prediction_days': 5,
        'max_epochs': 20,
        'batch_size': 128,
        'learning_rate': 0.03,
    }
    
    # All hyperparameters are already set in config above
    
    print(f"\n✓ Configuration set:")
    print(f"  - Training start: {config['train_start']}")
    print(f"  - Training days: {config['training_days']}")
    print(f"  - Prediction days: {config['prediction_days']}")
    print(f"  - Max epochs: {config['max_epochs']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    
    return config

# =============================================================================
# 4. Data Loading and Preprocessing
# =============================================================================

def load_and_prepare_data(file_path=None, config=None):
    """Load and prepare the dataset for TFT"""
    import pandas as pd  # Import pandas here
    import os
    
    print("\n=== Loading and Preparing Data ===")
    
    try:
        # Resolve default file path if not provided
        if file_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            processed_dir = os.path.join(project_root, 'data', 'processed')
            # Prefer normalized dataset if available
            candidates = [
                os.path.join(processed_dir, 'tsla_price_sentiment_spike_norm.csv'),
                os.path.join(processed_dir, 'tsla_price_sentiment_spike.csv'),
                os.path.join(processed_dir, 'tsla_sentiment_spike.csv'),
                os.path.join(processed_dir, 'TSLA_full_features.csv'),
                os.path.join(project_root, 'data', 'TSLA_close.csv')
            ]
            file_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

        # Load data
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully from {file_path}")
        
        # Normalize column names (lowercase, strip spaces)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Ensure 'date' column exists (handle common alternatives)
        if 'date' not in df.columns:
            alternative_date_cols = ['ds', 'timestamp', 'time', 'datetime']
            matched_alt = next((c for c in alternative_date_cols if c in df.columns), None)
            if matched_alt is None:
                # Try fuzzy match for any column containing 'date'
                matched_alt = next((c for c in df.columns if 'date' in c), None)
            if matched_alt is not None:
                df = df.rename(columns={matched_alt: 'date'})
        
        # If still missing, try to use DatetimeIndex
        if 'date' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={'index': 'date'})
            else:
                print("❌ Error: No 'date' column found and index is not datetime.")
                print(f"  - Available columns (normalized): {df.columns.tolist()}")
                return None
        
        # Parse dates (normalize tz-aware to UTC, then drop tz info) and drop invalid
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"⚠️ Dropping {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=['date'])
        
        # Basic info
        print(f"  - Shape: {df.shape}")
        min_date = df['date'].min()
        max_date = df['date'].max()
        def _fmt(d):
            try:
                return d.strftime('%Y-%m-%d')
            except Exception:
                return str(d)
        print(f"  - Date range: {_fmt(min_date)} to {_fmt(max_date)}")
        print("\nData columns (normalized):")
        print(df.columns.tolist())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values:")
            print(missing_values[missing_values > 0])
            # Fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            print("✓ Missing values filled")
        else:
            print("✓ No missing values found")
        
        # Ensure time_idx is properly set
        if 'time_idx' not in df.columns:
            df['time_idx'] = range(len(df))
            print("✓ time_idx column created")
        
        # Ensure unique_id exists
        if 'unique_id' not in df.columns:
            df['unique_id'] = 'TSLA'
            print("✓ unique_id column created")
        
        # Filter data by date range if specified
        if config and config.get('training_type') == 'date_range':
            start_date = config['start_date']
            end_date = config['end_date']
            
            # Ensure datetime (normalize tz-aware to UTC, then drop tz info)
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
            
            # Show available date range (safe)
            min_date = df['date'].min(); max_date = df['date'].max()
            min_str = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else str(min_date)
            max_str = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else str(max_date)
            print(f"  - Available data range: {min_str} to {max_str}")
            print(f"  - Requested range: {start_date} to {end_date}")
            
            # Filter by date range
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df_filtered = df[mask].copy()
            
            if len(df_filtered) == 0:
                print(f"❌ Error: No data found in date range {start_date} to {end_date}")
                print("\n💡 Available date ranges:")
                min_date = df['date'].min(); max_date = df['date'].max()
                min_str = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else str(min_date)
                max_str = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else str(max_date)
                recent90 = (max_date - pd.Timedelta(days=90)) if pd.notna(max_date) else max_date
                recent180 = (max_date - pd.Timedelta(days=180)) if pd.notna(max_date) else max_date
                recent90_str = recent90.strftime('%Y-%m-%d') if pd.notna(recent90) else str(recent90)
                recent180_str = recent180.strftime('%Y-%m-%d') if pd.notna(recent180) else str(recent180)
                print(f"  - Full dataset: {min_str} to {max_str}")
                print(f"  - Recent 90 days: {recent90_str} to {max_str}")
                print(f"  - Recent 180 days: {recent180_str} to {max_str}")
                print("\nUsing full dataset instead.")
            else:
                # Check if filtered data is sufficient
                min_required = 100  # Minimum data points needed for TFT
                if len(df_filtered) < min_required:
                    print(f"⚠️ Warning: Filtered data too small ({len(df_filtered)} points)")
                    print(f"  - Minimum recommended: {min_required} points")
                    print(f"  - Available in full dataset: {len(df)} points")
                    print("\n💡 Suggestions:")
                    print(f"  - Use full dataset for best performance")
                    print(f"  - Or choose a longer date range (e.g., {min_required + 50} days)")
                    print("\nUsing full dataset instead for better model performance.")
                else:
                    df = df_filtered
                    # Recreate time_idx for filtered data
                    df['time_idx'] = range(len(df))
                    print(f"✓ Data filtered to date range: {start_date} to {end_date}")
                    print(f"  - Filtered shape: {df.shape}")
        # Use date_anchor approach - use specified start date + 96 trading days
        if config and config.get('training_type') == 'date_anchor':
            print(f"✓ Using date_anchor approach with user-specified start date")
            print(f"  - Training start: {config.get('train_start', 'auto from data')}")
            print(f"  - Training days: {config.get('training_days', 96)}")
            print(f"  - Prediction days: {config.get('prediction_days', 5)}")
            
            # Use user-specified start date + 96 trading days
            training_days = config.get('training_days', 96)
            start_date = config.get('train_start')
            
            if start_date:
                # Find start index from user-specified date
                start_date_dt = pd.to_datetime(start_date)
                start_idx = df[df['date'] >= start_date_dt].index[0] if len(df[df['date'] >= start_date_dt]) > 0 else 0
                # Use training_days + prediction_days to ensure we have enough data for both training and prediction
                prediction_days = config.get('prediction_days', 5)
                total_days = training_days + prediction_days
                end_idx = min(start_idx + total_days, len(df))
                df = df.iloc[start_idx:end_idx].copy()
                df['time_idx'] = range(len(df))
                print(f"✓ Using {training_days} training days + {prediction_days} prediction days from {start_date}")
                print(f"  - Total data range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
                print(f"  - Total data points: {len(df)}")
            else:
                # Fallback to last 96 days if no start date specified
                df = df.tail(training_days).copy()
                df['time_idx'] = range(len(df))
                print(f"✓ No start date specified, using last {training_days} days")
                print(f"  - Training data range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
                print(f"  - Training data points: {len(df)}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found!")
        print("Please ensure the data file is in the current directory.")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

# =============================================================================
# 5. Create TFT Dataset
# =============================================================================

def create_tft_dataset(df, config):
    """Create TFT dataset with user-specified configuration"""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    
    print("\n=== Creating TFT Dataset ===")
    
    # Calculate training cutoff
    max_encoder_length = config['training_days']
    max_prediction_length = config['prediction_days']
    
    # Validate data length requirements
    total_data_points = len(df)
    min_required_length = max_encoder_length + max_prediction_length
    
    print(f"  - Total data points: {total_data_points}")
    print(f"  - Encoder length: {max_encoder_length} days")
    print(f"  - Prediction length: {max_prediction_length} days")
    print(f"  - Minimum required length: {min_required_length} days")
    
    if total_data_points < min_required_length:
        print(f"❌ Error: Insufficient data!")
        print(f"  - Available: {total_data_points} days")
        print(f"  - Required: {min_required_length} days")
        print(f"  - Shortage: {min_required_length - total_data_points} days")
        
        # Auto-adjust to use available data
        print("\n🔄 Auto-adjusting to use available data...")
        max_encoder_length = total_data_points - max_prediction_length
        if max_encoder_length < 10:  # Minimum reasonable encoder length
            print("❌ Error: Even with adjustment, insufficient data for TFT model")
            print("  - Need at least 15 data points for meaningful training")
            raise ValueError("Insufficient data for TFT model")
        
        config['training_days'] = max_encoder_length
        print(f"  - Adjusted encoder length: {max_encoder_length} days")
    
    # Calculate training cutoff - ensure we have enough data for training
    max_time_idx = df["time_idx"].max()
    training_cutoff = max_time_idx - max_prediction_length
    
    # Ensure we have enough data for training
    if training_cutoff < max_encoder_length:
        # Adjust encoder length to use all available data
        max_encoder_length = training_cutoff
        print(f"  - Adjusted encoder length to {max_encoder_length} to use all available data")
        training_cutoff = max_time_idx
        print(f"  - Using all {len(df)} data points for training")
    else:
        print(f"  - Training cutoff: time_idx {training_cutoff}")
        print(f"  - Available training data points: {len(df[df['time_idx'] <= training_cutoff])}")
    
    # Ensure training dataset retains at least one full (encoder+decoder) window
    # If dataset is small (e.g., start_date_window with encoder==training_days),
    # using the cutoff filter can remove decoder context entirely, yielding zero samples.
    min_rows_for_mask = max_encoder_length + 2 * max_prediction_length
    use_masked = total_data_points >= min_rows_for_mask
    # For explicit start_date_window, always use the full window without mask
    if config.get('training_type') == 'start_date_window':
        use_masked = False
    if not use_masked:
        print(f"  - Small window detected ({total_data_points} rows). Using full window without mask.")
    
    # Define time-varying features
    time_varying_known_reals = [
        "time_idx", "month", "day_of_week", "quarter", "year", 
        "is_month_end", "is_month_start", "days_since_earning", "rolling_volatility"
    ]
    
    # Prefer normalized volume if available
    time_varying_unknown_reals = ["close", "volume"]
    if "volume_norm" in df.columns:
        time_varying_unknown_reals = ["close", "volume_norm"]
    
    # Filter out features that don't exist in the dataset
    available_features = df.columns.tolist()
    time_varying_known_reals = [f for f in time_varying_known_reals if f in available_features]
    time_varying_unknown_reals = [f for f in time_varying_unknown_reals if f in available_features]
    
    print(f"  - Known features: {time_varying_known_reals}")
    print(f"  - Unknown features: {time_varying_unknown_reals}")
    
    # Create training dataset
    # Always use the prepared window without applying training cutoff mask to avoid empty datasets
    training_dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="close",
        group_ids=["unique_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["unique_id"],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["unique_id"], transformation="softplus"
        ),
    )
    
    print(f"✓ Training dataset created with {len(training_dataset)} samples")
    
    return training_dataset, training_cutoff

# =============================================================================
# 6. Create DataLoader and Model
# =============================================================================

def create_model_and_dataloader(training_dataset, config):
    """Create TFT model and data loader"""
    from pytorch_forecasting import TemporalFusionTransformer
    from torchmetrics import MeanSquaredError
    
    print("\n=== Creating Model and DataLoader ===")
    
    # Create data loader
    train_dataloader = training_dataset.to_dataloader(
        train=True, 
        batch_size=config['batch_size'], 
        num_workers=0
    )
    print(f"✓ DataLoader created with batch size {config['batch_size']}")
    
    # Create TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config['learning_rate'],
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        loss=MeanSquaredError(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    print(f"✓ TFT model created with {sum(p.numel() for p in tft.parameters()):,} parameters")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Hidden size: 16")
    print(f"  - Attention heads: 1")
    
    return tft, train_dataloader

# =============================================================================
# 7. Model Training
# =============================================================================

def train_model(tft, train_dataloader, config):
    """Train the TFT model with callbacks and monitoring"""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    
    print("\n=== Training TFT Model ===")
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor="train_loss", 
        min_delta=1e-4, 
        patience=5, 
        verbose=True, 
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='./checkpoints',
        filename='tft-{epoch:02d}-{train_loss:.4f}',
        save_top_k=3,
        mode='min',
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator="auto",
        devices="auto",
        callbacks=[early_stopping, lr_monitor, checkpoint_callback],
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    print(f"✓ Trainer configured with {config['max_epochs']} max epochs")
    print("  - Early stopping enabled")
    print("  - Learning rate monitoring enabled")
    print("  - Model checkpointing enabled")
    
    # Train the model
    print("\nStarting model training...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
    )
    
    print("✅ Training completed!")
    
    return trainer

# =============================================================================
# 8. Load Best Model and Create Validation Dataset
# =============================================================================

def load_best_model_and_validate(trainer, training_dataset, df, tft, config):
    """Load the best model and create validation dataset"""
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    
    print("\n=== Loading Best Model and Creating Validation Dataset ===")
    
    # Load best model
    if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback.best_model_path:
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        print(f"✓ Best model loaded from: {best_model_path}")
    else:
        best_tft = tft  # Use the last trained model if no checkpoint
        print("✓ Using last trained model (no checkpoint available)")
    
    # Create validation dataset
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df,
        predict=True,
        stop_randomization=True,
    )
    
    # Create validation dataloader
    val_dataloader = validation_dataset.to_dataloader(
        train=False, 
        batch_size=config['batch_size'], 
        num_workers=0
    )
    
    print(f"✓ Validation dataset created with {len(validation_dataset)} samples")
    
    return best_tft, val_dataloader

# =============================================================================
# 9. Make Predictions
# =============================================================================

def make_predictions(best_tft, val_dataloader):
    """Make predictions using the trained model"""
    import torch
    
    print("\n=== Making Predictions ===")
    
    # Make predictions
    predictions = best_tft.predict(val_dataloader)
    
    # Get actual values
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    
    print(f"✓ Predictions made successfully")
    print(f"  - Prediction shape: {predictions.shape}")
    print(f"  - Actuals shape: {actuals.shape}")
    
    return predictions, actuals

# =============================================================================
# 10. Performance Evaluation
# =============================================================================

def evaluate_performance(predictions, actuals):
    """Evaluate model performance using various metrics"""
    import pandas as pd  # Import pandas here
    import torch
    from torchmetrics import MeanAbsoluteError, MeanSquaredError
    
    print("\n=== Performance Evaluation ===")
    
    # Move tensors to CPU for metric calculation
    predictions_cpu = predictions.cpu()
    actuals_cpu = actuals.cpu()
    
    # Calculate metrics
    mae_metric = MeanAbsoluteError()
    mae = mae_metric(predictions_cpu, actuals_cpu).item()
    
    mse_metric = MeanSquaredError()
    mse = mse_metric(predictions_cpu, actuals_cpu).item()
    
    rmse_metric = MeanSquaredError(squared=False)
    rmse = rmse_metric(predictions_cpu, actuals_cpu).item()
    
    # Calculate MAPE as Python float to avoid tensor prints
    mape = (torch.abs((actuals_cpu - predictions_cpu) / actuals_cpu)).mean().mul(100).item()
    
    # Create performance summary
    performance_metrics = {
        'Metric': [
            'MAE (Mean Absolute Error)',
            'MSE (Mean Squared Error)',
            'RMSE (Root Mean Squared Error)',
            'MAPE (Mean Absolute Percentage Error)'
        ],
        'Value': [
            f'{mae:.4f}',
            f'{mse:.4f}',
            f'{rmse:.4f}',
            f'{mape:.4f}'
        ]
    }
    
    metrics_df = pd.DataFrame(performance_metrics)
    print("\n📊 Model Performance Metrics:")
    print(metrics_df)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }

# =============================================================================
# 11. Visualization
# =============================================================================

def create_visualizations(predictions, actuals, config, df=None):
    """Create comprehensive visualizations of predictions vs actuals"""
    import matplotlib.pyplot as plt  # Import matplotlib here
    import numpy as np  # Import numpy here
    
    print("\n=== Creating Visualizations ===")
    
    # Debug: Check the data being passed to visualization
    if df is not None:
        print(f"DEBUG: Visualization df shape: {df.shape}")
        print(f"DEBUG: Visualization df date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"DEBUG: Visualization df time_idx range: {df['time_idx'].min()} to {df['time_idx'].max()}")
    else:
        print("DEBUG: No df passed to visualization")
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')  # Fallback to default style
    
    # Move to CPU for plotting
    predictions_cpu = predictions.cpu().numpy()
    actuals_cpu = actuals.cpu().numpy()
    
    # Create standalone Actual vs Prediction plot with training period
    create_standalone_plot(predictions, actuals, config, df)

def create_standalone_plot(predictions, actuals, config, df=None):
    """Create a standalone plot showing training period and predictions with dates"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("\n=== Creating Standalone Training + Prediction Plot ===")
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    # Move to CPU for plotting
    predictions_cpu = predictions.cpu().numpy()
    actuals_cpu = actuals.cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(16, 8))
    
    if df is not None and 'date' in df.columns:
        # Convert dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
        
        # Get training period dates - use the actual training data from the model
        total_points = len(df)
        training_points = config['training_days']
        
        # Get the actual training data (last N days before prediction)
        training_dates = df['date'].iloc[-training_points-config['prediction_days']:-config['prediction_days']].tolist()
        training_values = df['close'].iloc[-training_points-config['prediction_days']:-config['prediction_days']].values
        
        # Get prediction dates (the last N days that were used for prediction)
        prediction_dates = df['date'].iloc[-config['prediction_days']:].tolist()
        
        # Plot training data
        plt.plot(training_dates, training_values, 
                'b-', label='Training Data (Close Price)', linewidth=2, alpha=0.7)
        
        # Plot predictions
        plt.plot(prediction_dates, predictions_cpu[0], 'r--', label='Predictions', 
                linewidth=3, marker='o', markersize=8)
        
        # Plot actuals for prediction period
        actual_values = df['close'].iloc[-config['prediction_days']:].values
        plt.plot(prediction_dates, actual_values, 'g-', label='Actual (Close Price)', 
                linewidth=2, marker='s', markersize=6)
        
        # Add vertical line to separate training and prediction
        last_training_date = training_dates[-1]
        plt.axvline(x=last_training_date, color='gray', linestyle='--', alpha=0.7, 
                   label='Training End / Prediction Start')
        
        # Format x-axis - show only year and month for cleaner display
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add grid and labels
        plt.grid(True, alpha=0.3)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price (USD)', fontsize=12)
        
        # Add title with training info - use actual training data count
        training_cutoff = df["time_idx"].max() - config["prediction_days"]
        actual_training_days = len(df[df['time_idx'] <= training_cutoff])
        if config.get('training_type') == 'date_range':
            title = f'TFT Model: Training ({config["start_date"]} to {config["end_date"]}) + {config["prediction_days"]} Days Prediction'
        else:
            title = f'TFT Model: {actual_training_days} Days Training + {config["prediction_days"]} Days Prediction'
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        
    else:
        # Fallback without dates
        training_points = config['training_days']
        x_training = list(range(training_points))
        x_prediction = list(range(training_points, training_points + config['prediction_days']))
        
        # Plot training data (placeholder - would need actual training data)
        plt.plot(x_training, [0] * training_points, 'b-', label='Training Data', linewidth=2, alpha=0.7)
        
        # Plot predictions
        plt.plot(x_prediction, predictions_cpu[0], 'r--', label='Predictions', 
                linewidth=3, marker='o', markersize=8)
        
        # Plot actuals
        plt.plot(x_prediction, actuals_cpu[0], 'g-', label='Actual', 
                linewidth=2, marker='s', markersize=6)
        
        # Add vertical line
        plt.axvline(x=training_points, color='gray', linestyle='--', alpha=0.7, 
                   label='Training End / Prediction Start')
        
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Stock Price (Normalized)', fontsize=12)
        plt.title(f'TFT Model: {config["training_days"]} Days Training + {config["prediction_days"]} Days Prediction', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to results directory with consistent naming
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'TSLA_TFT_baseline_forecast.png')
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"⚠️ Failed to save plot: {e}")
    finally:
        plt.show()
    
    print("✓ Standalone plot created successfully!")

# =============================================================================
# 12. Model Interpretation (Optional)
# =============================================================================

def interpret_model(best_tft, val_dataloader):
    """Interpret model predictions and show variable importance"""
    import matplotlib.pyplot as plt  # Import matplotlib here
    
    print("\n=== Model Interpretation ===")
    
    try:
        # Get raw predictions for interpretation
        raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
        
        # Variable importance
        interpretation = best_tft.interpret_output(raw_predictions[0], reduction="sum")
        
        # Plot variable importance
        plt.figure(figsize=(10, 6))
        best_tft.plot_interpretation(interpretation)
        plt.title("Variable Importance Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Attention visualization
        x = raw_predictions[1]
        plt.figure(figsize=(12, 8))
        best_tft.plot_attention(x, raw_predictions[0], idx=0)
        plt.title("Attention Weights Visualization", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("✓ Model interpretation completed!")
        
    except Exception as e:
        print(f"⚠️ Model interpretation failed: {e}")
        print("This is optional and doesn't affect the main predictions.")

# =============================================================================
# 13. Save Results and Update Performance Matrix
# =============================================================================

def save_results_and_update_matrix(performance_metrics, config):
    """Save results and update the performance matrix"""
    import pandas as pd  # Import pandas here
    from datetime import datetime  # Import datetime here
    import pickle  # Import pickle here
    
    print("\n=== Saving Results ===")
    
    # Create results summary
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_days': config['training_days'],
        'prediction_days': config['prediction_days'],
        'max_epochs': config['max_epochs'],
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'mae': performance_metrics['mae'],
        'mse': performance_metrics['mse'],
        'rmse': performance_metrics['rmse'],
        'mape': performance_metrics['mape']
        
    }
    
   # Prepare results directory under project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save results to CSV
    results_df = pd.DataFrame([results_summary])
    # results_filename = os.path.join(results_dir, f"TFT_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    # results_df.to_csv(results_filename, index=False)
    # print(f"✓ Results saved to: {results_filename}")
    
    # Update Results Matrix
    print("\n## Update Results Matrix")
    
    # Load existing results matrix
    try:
        with open(os.path.join(results_dir, "TSLA_results_matrix.pkl"), "rb") as f:
            matrix = pickle.load(f)
        print("✓ Loaded existing results matrix")
        print("\nCurrent matrix:")
        print(matrix)
    except FileNotFoundError:
        print("⚠️  No existing results matrix found. Creating new one...")
        # Create new matrix if none exists
        matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE'])

    # Add TFT model results (standardized key)
    print("\nAdding TFT model results...")
    matrix.loc["TFT_baseline"] = [
        performance_metrics['mae'], 
        performance_metrics['mse'], 
        performance_metrics['rmse'],
        performance_metrics['mape']
    ]
    
    print("\nUpdated matrix:")
    print(matrix)
    
    # Save updated matrix
    with open(os.path.join(results_dir, "TSLA_results_matrix.pkl"), "wb") as f:
        pickle.dump(matrix, f)
    # Also save as CSV (reordered rows)
    csv_path = os.path.join(results_dir, "result_matrix.csv")
    try:
        if os.path.exists(csv_path):
            global_matrix = pd.read_csv(csv_path, index_col=0)
        else:
            global_matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE'])
        if 'MAPE' not in global_matrix.columns:
            global_matrix = global_matrix.reindex(columns=['MAE', 'MSE', 'RMSE', 'MAPE'])
        global_matrix.loc['TFT_baseline'] = [
            performance_metrics['mae'],
            performance_metrics['mse'],
            performance_metrics['rmse'],
            performance_metrics['mape']
        ]
        desired_order = ['ARIMA', 'TimesFM', 'Chronos', 'TFT_baseline', 'TFT_Reddit']
        ordered = [i for i in desired_order if i in global_matrix.index]
        rest = [i for i in global_matrix.index if i not in desired_order]
        global_matrix = global_matrix.loc[ordered + rest]
        global_matrix.to_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Failed to update global results matrix CSV: {e}")
    
    print("\nTFT analysis complete.")
    print(f"📊 Model trained on {config['training_days']} days of data")
    print(f"🔮 Predictions made for {config['prediction_days']} days ahead")
    print(f"📈 Best RMSE: {performance_metrics['rmse']:.4f}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function"""
    print("🚀 Starting Automated TFT Analysis...")
    
    # Set random seed for reproducibility
    import torch
    import os
    import random
    import numpy as np
    
    # Check if seed is provided as command line argument
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            print(f"Seed set to {seed}")
        except ValueError:
            torch.manual_seed(42)
            random.seed(42)
            np.random.seed(42)
            print("Seed set to 42 (default)")
    else:
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        print("Seed set to 42")
    
    # Install dependencies
    install_dependencies()
    
    # Import libraries
    (pd, np, torch, pl, Trainer, EarlyStopping, LearningRateMonitor, 
     ModelCheckpoint, DataLoader, TimeSeriesDataSet, TemporalFusionTransformer,
     GroupNormalizer, MeanSquaredError, MeanAbsoluteError, plt, 
     warnings, pickle, datetime, timedelta) = import_libraries()
    
    # Get user configuration
    config = get_user_config()
    
    # Build dataset path from project root and load data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    candidates = [
        os.path.join(processed_dir, 'tsla_price_sentiment_spike.csv'),
        os.path.join(processed_dir, 'TSLA_full_features.csv'),
        os.path.join(project_root, 'data', 'TSLA_close.csv')
    ]
    data_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    print(f"Using data file: {data_path}")
    df = load_and_prepare_data(file_path=data_path, config=config)
    print("\nFirst few rows:")
    print(df.head())
    
    # Create TFT dataset
    training_dataset, training_cutoff = create_tft_dataset(df, config)
    
    # Create model and dataloader
    tft, train_dataloader = create_model_and_dataloader(training_dataset, config)
    
    # Train the model
    trainer = train_model(tft, train_dataloader, config)
    
    # Load best model and create validation dataset
    best_tft, val_dataloader = load_best_model_and_validate(trainer, training_dataset, df, tft, config)
    
    # Make predictions
    predictions, actuals = make_predictions(best_tft, val_dataloader)
    
    # Evaluate performance
    performance_metrics = evaluate_performance(predictions, actuals)
    
    # Create visualizations
    create_visualizations(predictions, actuals, config, df)
    
    # Interpret model (optional)
    interpret_model(best_tft, val_dataloader)
    
    # Save results and update matrix
    save_results_and_update_matrix(performance_metrics, config)

if __name__ == "__main__":
    main()
