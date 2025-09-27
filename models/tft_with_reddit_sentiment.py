#!/usr/bin/env python3
"""
TFT with Reddit Sentiment and Spike Data - Automated Version
============================================================

This script automates the entire TFT forecasting process with Reddit sentiment analysis
and spike detection data. It includes:

- Automated data loading and preprocessing
- Interactive training period selection
- TFT model training with sentiment features
- Performance evaluation and visualization
- Results saving and matrix updates

Date: 2025-08-29
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Install Dependencies
# =============================================================================

def install_dependencies():
    """Install required packages if not already installed"""
    print("🔧 Checking and installing dependencies...")
    
    try:
        import pytorch_forecasting
        print("✓ pytorch_forecasting already installed")
    except ImportError:
        print("Installing pytorch_forecasting...")
        os.system("pip install pytorch_forecasting==1.0.0 lightning==2.0.9 torchmetrics --quiet")
        print("✓ pytorch_forecasting installed")
    
    try:
        import yfinance
        print("✓ yfinance already installed")
    except ImportError:
        print("Installing yfinance...")
        os.system("pip install yfinance --quiet")
        print("✓ yfinance installed")

# =============================================================================
# 2. Import Libraries
# =============================================================================

def import_libraries():
    """Import all required libraries"""
    print("📚 Importing libraries...")
    
    try:
        import pandas as pd
        import torch
        import lightning.pytorch as pl
        from lightning.pytorch import Trainer
        from torch.utils.data import DataLoader
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from torchmetrics import MeanSquaredError, MeanAbsoluteError, SymmetricMeanAbsolutePercentageError
        import matplotlib.pyplot as plt
        import numpy as np
        from datetime import datetime, timedelta
        import pickle
        
        print("✓ All libraries imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Error importing libraries: {e}")
        return False

# =============================================================================
# 3. User Configuration
# =============================================================================

def get_user_config():
    """Ask for training start date only; auto 96-day training + 5-day prediction."""
    print("\n=== TFT Configuration (96-day training, 5-day prediction) ===")
    try:
        start_date_str = input("📅 Enter training start date (YYYY-MM-DD): ").strip()
    except EOFError:
        # Auto-use default date when running in non-interactive mode
        start_date_str = "2025-02-01"
        print(f"Using default start date: {start_date_str}")

    config = {
        'training_type': 'date_anchor',
        'train_start': start_date_str,
        'training_days': 96,
        'prediction_days': 5,
        'max_epochs': 20,
        'batch_size': 128,
        'learning_rate': 0.03,
    }

    print("\n✓ Configuration set:")
    print(f"  - Training start: {config['train_start'] or '(auto from data)'}")
    print(f"  - Training days: {config['training_days']}")
    print(f"  - Prediction days: {config['prediction_days']}")
    print(f"  - Max epochs: {config['max_epochs']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['learning_rate']}")

    return config

# =============================================================================
# 4. Data Loading and Preparation
# =============================================================================

def load_and_prepare_data(file_path="tsla_price_sentiment_spike.csv", config=None):
    """Load and prepare data for TFT model"""
    import pandas as pd
    
    print("\n=== Loading and Preparing Data ===")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully from {file_path}")
        print(f"  - Shape: {df.shape}")
        
        # Convert date column to datetime and sort
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  - Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Ensure essential columns
        if 'time_idx' not in df.columns:
            df['time_idx'] = range(len(df))
        if 'unique_id' not in df.columns:
            df['unique_id'] = 'TSLA'

        # Basic feature hygiene
        df['volume'] = df.get('volume', 0).fillna(0)
        df['close'] = df['close'].astype(float)

        # Feature engineering to leverage sentiment/spike information without leakage
        # Price returns and volatility (encoder-only)
        df['return_1d'] = df['close'].pct_change()
        df['rolling_volatility'] = df['return_1d'].rolling(window=14, min_periods=1).std()

        # Sentiment lags and rolling stats (known at prediction time)
        if 'daily_sentiment' in df.columns:
            for i in range(1, 6):
                df[f'daily_sentiment_lag{i}'] = df['daily_sentiment'].shift(i)
            for w in (3, 7, 14):
                df[f'daily_sentiment_mean_{w}'] = df['daily_sentiment'].shift(1).rolling(window=w, min_periods=1).mean()
                df[f'daily_sentiment_std_{w}'] = df['daily_sentiment'].shift(1).rolling(window=w, min_periods=1).std()

        # Spike aggregations (known at prediction time)
        if 'spike_presence' in df.columns:
            for w in (3, 7, 14):
                df[f'spike_presence_sum_{w}'] = df['spike_presence'].shift(1).rolling(window=w, min_periods=1).sum()
        if 'spike_intensity' in df.columns:
            for w in (3, 7, 14):
                df[f'spike_intensity_max_{w}'] = df['spike_intensity'].shift(1).rolling(window=w, min_periods=1).max()

        # Calendar features if not present
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        # Placeholder for earnings proximity if not provided
        if 'days_since_earning' not in df.columns:
            df['days_since_earning'] = 0

        # Fill any NaNs introduced by lag/rolling
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Display data info
        print("\nData columns:")
        print(df.columns.tolist())
        
        # Filter data by date range if specified
        if config and config.get('training_type') == 'date_range':
            start_date = config['start_date']
            end_date = config['end_date']
            
            # Convert date column to datetime if it's not already
            df['date'] = pd.to_datetime(df['date'])
            
            # Show available date range
            print(f"  - Available data range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            print(f"  - Requested range: {start_date} to {end_date}")
            
            # Filter by date range
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df_filtered = df[mask].copy()
            
            if len(df_filtered) == 0:
                print(f"❌ Error: No data found in date range {start_date} to {end_date}")
                print("\n💡 Available date ranges:")
                print(f"  - Full dataset: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
                print(f"  - Recent 90 days: {(df['date'].max() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
                print(f"  - Recent 180 days: {(df['date'].max() - pd.Timedelta(days=180)).strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
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

        # Display first few rows
        print("\nFirst few rows:")
        print(df.head())
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found!")
        print("Please make sure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

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
    
    # Define time-varying features (engineered sentiment/spike features as known; target/price as unknown)
    time_varying_known_reals = [
        "time_idx", "month", "day_of_week", "quarter", "year", 
        "is_month_end", "is_month_start", "days_since_earning",
        # Sentiment lags and aggregates
        "daily_sentiment_lag1", "daily_sentiment_lag2", "daily_sentiment_lag3", "daily_sentiment_lag4", "daily_sentiment_lag5",
        "daily_sentiment_mean_3", "daily_sentiment_mean_7", "daily_sentiment_mean_14",
        "daily_sentiment_std_7", "daily_sentiment_std_14",
        # Spike aggregates
        "spike_presence_sum_3", "spike_presence_sum_7", "spike_presence_sum_14",
        "spike_intensity_max_3", "spike_intensity_max_7", "spike_intensity_max_14"
    ]
    
    time_varying_unknown_reals = [
        "close", "volume", "rolling_volatility"
    ]
    
    # Filter out features that don't exist in the dataset
    available_features = df.columns.tolist()
    time_varying_known_reals = [f for f in time_varying_known_reals if f in available_features]
    time_varying_unknown_reals = [f for f in time_varying_unknown_reals if f in available_features]
    
    print(f"  - Known features: {time_varying_known_reals}")
    print(f"  - Unknown features: {time_varying_unknown_reals}")
    
    # Create training dataset
    training_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
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
# 6. Create Model and DataLoader
# =============================================================================

def create_model_and_dataloader(training_dataset, config):
    """Create TFT model and data loader"""
    from pytorch_forecasting import TemporalFusionTransformer
    from torchmetrics import MeanSquaredError
    
    print("\n=== Creating Model and DataLoader ===")
    
    # Create DataLoader
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
    
    print(f"✓ TFT model created with {sum(p.numel() for p in tft.parameters())} parameters")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Hidden size: 16")
    print(f"  - Attention heads: 1")
    
    return tft, train_dataloader

# =============================================================================
# 7. Train Model
# =============================================================================

def train_model(tft, train_dataloader, training_dataset, config):
    """Train the TFT model with callbacks and monitoring"""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    
    print("\n=== Training TFT Model ===")

    # Create validation loader from training_dataset (non-shuffled)
    val_dataloader = training_dataset.to_dataloader(train=False, batch_size=config['batch_size'], num_workers=0)

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

    # Create trainer with validation monitoring
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator="auto",
        devices="auto",
        callbacks=[early_stopping, lr_monitor, checkpoint_callback],
        gradient_clip_val=0.1,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    print(f"✓ Trainer configured with {config['max_epochs']} max epochs (monitoring train_loss)")
    print("  - Early stopping enabled")
    print("  - Learning rate monitoring enabled")
    print("  - Model checkpointing enabled")
    
    # Train the model
    print("\nStarting model training...")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("✅ Training completed!")
    
    return trainer

# =============================================================================
# 8. Load Best Model and Validate
# =============================================================================

def load_best_model_and_validate(trainer, training_dataset, df, tft, config):
    """Load the best model and create validation dataset"""
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    
    print("\n=== Loading Best Model and Creating Validation Dataset ===")
    
    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    print(f"✓ Best model loaded from: {best_model_path}")
    
    # Create validation dataset
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df, 
        predict=True,
        stop_randomization=True,
    )
    
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
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    
    print("✓ Predictions made successfully")
    print(f"  - Prediction shape: {predictions.shape}")
    print(f"  - Actuals shape: {actuals.shape}")
    
    return predictions, actuals

# =============================================================================
# 10. Evaluate Performance
# =============================================================================

def evaluate_performance(predictions, actuals):
    """Evaluate model performance using various metrics"""
    import pandas as pd
    import torch
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, SymmetricMeanAbsolutePercentageError
    
    print("\n=== Performance Evaluation ===")
    
    # Move to CPU for evaluation
    predictions_cpu = predictions.cpu()
    actuals_cpu = actuals.cpu()
    
    # Calculate metrics
    mae_metric = MeanAbsoluteError()
    mae = mae_metric(predictions_cpu, actuals_cpu).item()
    
    mse_metric = MeanSquaredError()
    mse = mse_metric(predictions_cpu, actuals_cpu).item()
    
    rmse_metric = MeanSquaredError(squared=False)
    rmse = rmse_metric(predictions_cpu, actuals_cpu).item()
    
    # Create performance metrics DataFrame
    metrics_df = pd.DataFrame({
        'Metric': [
            'MAE',
            'MSE',
            'RMSE'
        ],
        'Value': [
            f'{mae:.4f}',
            f'{mse:.4f}',
            f'{rmse:.4f}'
        ]
    })
    
    print("\n📊 Model Performance Metrics:")
    print(metrics_df)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
    }

# =============================================================================
# 11. Create Visualizations
# =============================================================================

def create_visualizations(predictions, actuals, config, df=None):
    """Create comprehensive visualizations of predictions vs actuals"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n=== Creating Visualizations ===")
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
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
        df['date'] = pd.to_datetime(df['date'])
        
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
            title = f'TFT Model with Sentiment & Spike: Training ({config["start_date"]} to {config["end_date"]}) + {config["prediction_days"]} Days Prediction'
        else:
            title = f'TFT Model with Sentiment & Spike: {actual_training_days} Days Training + {config["prediction_days"]} Days Prediction'
        
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
        plt.title(f'TFT Model with Sentiment & Spike: {config["training_days"]} Days Training + {config["prediction_days"]} Days Prediction', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to results directory with consistent naming
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'TSLA_TFT_with_reddit_sentiment_forecast.png')
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
    """Interpret model outputs and attention weights"""
    import matplotlib.pyplot as plt
    
    print("\n=== Model Interpretation ===")
    
    try:
        # Get raw predictions for interpretation
        raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
        
        # Variable importance
        interpretation = best_tft.interpret_output(raw_predictions[0], reduction="sum")
        best_tft.plot_interpretation(interpretation)
        plt.title("Variable Importance")
        plt.show()
        
        # Prediction errors over time
        true = raw_predictions[1]["decoder_target"]
        pred = raw_predictions[0]
        error = (pred - true).abs().detach().cpu().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.plot(error[0], label="Prediction Error (abs)")
        plt.axvline(x=0, color='gray', linestyle='--', label="Prediction Start")
        plt.title("Prediction Error Over Forecast Horizon (Regime Change Detection)")
        plt.xlabel("Time Step")
        plt.ylabel("Error")
        plt.legend()
        plt.grid()
        plt.show()
        
        print("✓ Model interpretation completed!")
        
    except Exception as e:
        print(f"⚠️ Model interpretation failed: {e}")
        print("This is optional and doesn't affect the main predictions.")

# =============================================================================
# 13. Save Results and Update Matrix
# =============================================================================

def save_results_and_update_matrix(performance_metrics, config):
    """Save results and update performance matrix"""
    import pandas as pd
    from datetime import datetime
    
    print("\n=== Saving Results ===")
    
    # Update Results Matrix
    try:
        import pickle
        
        # Resolve results directory under project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)

        matrix_path = os.path.join(results_dir, "TSLA_results_matrix.pkl")

        # Load existing matrix
        if os.path.exists(matrix_path):
            with open(matrix_path, "rb") as f:
                matrix = pickle.load(f)
        else:
            # Create new matrix if it doesn't exist
            matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE'])
        
        # Add new results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"TFT_Reddit_Y"
        matrix.loc[model_name] = [
            performance_metrics['MAE'],
            performance_metrics['MSE'],
            performance_metrics['RMSE']
        ]
        
        # Save updated matrix to pickle
        with open(matrix_path, "wb") as f:
            pickle.dump(matrix, f)
        
        # Save updated matrix to CSV in results directory
        csv_path = os.path.join(results_dir, "result_matrix.csv")
        matrix.to_csv(csv_path, index=True)
        
        # Display updated matrix
        print(f"\n📋 Updated Results Matrix:")
        print(matrix)
        
    except Exception as e:
        print(f"⚠️ Error updating performance matrix: {e}")

# =============================================================================
# 14. Main Function
# =============================================================================

def main():
    """Main function to run the entire TFT analysis pipeline"""
    print("Starting Automated TFT Analysis with Reddit Sentiment & Spike Data...")
    
    # Set random seed for reproducibility
    import torch
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
    if not import_libraries():
        print("❌ Failed to import libraries. Exiting.")
        return
    
    # Get user configuration
    config = get_user_config()
    
    # Load and prepare data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "processed", "tsla_price_sentiment_spike.csv")
    df = load_and_prepare_data(data_path, config)
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create TFT dataset
    training_dataset, training_cutoff = create_tft_dataset(df, config)
    
    # Create model and dataloader
    tft, train_dataloader = create_model_and_dataloader(training_dataset, config)
    
    # Train model
    trainer = train_model(tft, train_dataloader, training_dataset, config)
    
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
    
    # Final summary
    print(f"\n🎉 TFT Analysis with Reddit Sentiment & Spike Complete!")
    print(f"📊 Model trained on {config['training_days']} days of data")
    print(f"🔮 Predictions made for {config['prediction_days']} days ahead")
    print(f"📈 Best RMSE: {performance_metrics['RMSE']:.4f}")

if __name__ == "__main__":
    main()
