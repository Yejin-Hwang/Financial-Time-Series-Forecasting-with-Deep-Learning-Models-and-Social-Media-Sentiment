"""
Visualization Module

This module provides plotting functions for financial time series forecasting results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import pickle


def setup_plot_style():
    """Setup consistent plot styling"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_predictions(
    predictions: torch.Tensor,
    actuals: torch.Tensor, 
    df: pd.DataFrame,
    config: Any,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Create comprehensive prediction plots
    
    Args:
        predictions: Model predictions tensor
        actuals: Actual values tensor
        df: Original dataframe with dates
        config: Model configuration
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()
    
    # Convert tensors to numpy
    pred_np = predictions.cpu().numpy()
    actual_np = actuals.cpu().numpy()
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'TFT Model Performance - {config.training_days} Days Training, {config.prediction_days} Days Prediction',
        fontsize=16, fontweight='bold'
    )
    
    # Plot 1: Time series comparison
    axes[0, 0].plot(actual_np[0], 'b-', label='Actual', linewidth=2, marker='o')
    axes[0, 0].plot(pred_np[0], 'r--', label='Prediction', linewidth=2, marker='s')
    axes[0, 0].set_title('Predictions vs Actuals')
    axes[0, 0].set_xlabel('Time Horizon (Days)')
    axes[0, 0].set_ylabel('Stock Price (Normalized)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction errors
    errors = np.abs(pred_np[0] - actual_np[0])
    axes[0, 1].bar(range(len(errors)), errors, color='orange', alpha=0.7)
    axes[0, 1].set_title('Absolute Prediction Errors')
    axes[0, 1].set_xlabel('Time Horizon (Days)')
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot
    axes[1, 0].scatter(actual_np.flatten(), pred_np.flatten(), alpha=0.6, color='green')
    axes[1, 0].plot([actual_np.min(), actual_np.max()], [actual_np.min(), actual_np.max()], 'r--', lw=2)
    axes[1, 0].set_title('Predictions vs Actuals (Scatter)')
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    all_errors = (pred_np - actual_np).flatten()
    axes[1, 1].hist(all_errors, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 1].set_title('Prediction Error Distribution')
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_and_prediction(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    df: pd.DataFrame,
    config: Any,
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Create a timeline plot showing training period and predictions
    
    Args:
        predictions: Model predictions tensor
        actuals: Actual values tensor  
        df: Original dataframe with dates
        config: Model configuration
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    setup_plot_style()
    
    # Convert tensors to numpy
    pred_np = predictions.cpu().numpy()
    actual_np = actuals.cpu().numpy()
    
    plt.figure(figsize=(16, 8))
    
    if 'date' in df.columns:
        # Convert dates
        df['date'] = pd.to_datetime(df['date'])
        
        # Get training and prediction periods
        total_points = len(df)
        training_points = config.training_days
        
        # Get training data
        training_dates = df['date'].iloc[-training_points-config.prediction_days:-config.prediction_days].tolist()
        training_values = df['close'].iloc[-training_points-config.prediction_days:-config.prediction_days].values
        
        # Get prediction dates
        prediction_dates = df['date'].iloc[-config.prediction_days:].tolist()
        
        # Plot training data
        plt.plot(training_dates, training_values, 
                'b-', label='Training Data', linewidth=2, alpha=0.7)
        
        # Plot predictions and actuals
        plt.plot(prediction_dates, pred_np[0], 'r--', label='Predictions', 
                linewidth=3, marker='o', markersize=8)
        plt.plot(prediction_dates, df['close'].iloc[-config.prediction_days:].values, 
                'g-', label='Actual', linewidth=2, marker='s', markersize=6)
        
        # Add separator line
        plt.axvline(x=training_dates[-1], color='gray', linestyle='--', alpha=0.7,
                   label='Training End / Prediction Start')
        
        # Format dates
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price (USD)', fontsize=12)
        
    else:
        # Fallback without dates
        training_points = config.training_days
        x_training = list(range(training_points))
        x_prediction = list(range(training_points, training_points + config.prediction_days))
        
        plt.plot(x_prediction, pred_np[0], 'r--', label='Predictions', 
                linewidth=3, marker='o', markersize=8)
        plt.plot(x_prediction, actual_np[0], 'g-', label='Actual', 
                linewidth=2, marker='s', markersize=6)
        
        plt.axvline(x=training_points, color='gray', linestyle='--', alpha=0.7,
                   label='Training End / Prediction Start')
        
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Stock Price (Normalized)', fontsize=12)
    
    plt.title(f'TFT Model: {config.training_days} Days Training + {config.prediction_days} Days Prediction',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Timeline plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_results(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    metrics: Dict[str, float],
    config: Any,
    output_dir: Path,
    symbol: str
) -> str:
    """
    Save model results and metrics
    
    Args:
        predictions: Model predictions
        actuals: Actual values
        metrics: Performance metrics
        config: Model configuration
        output_dir: Output directory
        symbol: Stock symbol
        
    Returns:
        Path to saved results file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results summary
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'training_days': config.training_days,
        'prediction_days': config.prediction_days,
        'max_epochs': config.max_epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'mae': metrics['mae'],
        'mse': metrics['mse'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape']
    }
    
    # Save as CSV
    results_df = pd.DataFrame([results_summary])
    csv_file = output_dir / f"{symbol}_TFT_results_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    
    # Save detailed results as pickle
    detailed_results = {
        'predictions': predictions.cpu().numpy(),
        'actuals': actuals.cpu().numpy(),
        'metrics': metrics,
        'config': config,
        'summary': results_summary
    }
    
    pickle_file = output_dir / f"{symbol}_TFT_detailed_{timestamp}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(detailed_results, f)
    
    print(f"✓ Results saved to: {csv_file}")
    print(f"✓ Detailed results saved to: {pickle_file}")
    
    return str(csv_file)
