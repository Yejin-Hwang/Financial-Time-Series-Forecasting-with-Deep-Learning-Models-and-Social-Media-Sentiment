#!/usr/bin/env python3
"""
Fixed ARIMA Model for Predicting Tesla Stock Price
Author: yejin
Fixed version addressing the main issues in the original notebook
"""

from .deps import (
    warnings, pickle, Path,
    pd, np, plt,
    ARIMA, auto_arima, adfuller, acorr_ljungbox,
    mean_absolute_error, mean_squared_error,
)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import warnings
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import pickle
# from pathlib import Path

warnings.filterwarnings("ignore")

def _resolve_tsla_csv_path() -> str:
    """Resolve path to TSLA_close.csv across common project locations."""
    base_dir = Path(__file__).resolve().parent.parent  # project root
    candidates = [
        Path("TSLA_close.csv"),
        base_dir / "data" / "TSLA_close.csv",
        base_dir / "data" / "raw" / "TSLA_close.csv",
        Path.cwd() / "data" / "TSLA_close.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    searched = "\n    ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Can't find TSLA_close.csv. Searched:\n    {searched}")

def get_user_input():
    """Get user input for training period and prediction days"""
    print("=== ARIMA Model Configuration ===\n")
    
    # Get available date range
    try:
        df_path = _resolve_tsla_csv_path()
        df = pd.read_csv(df_path)
        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min()
        max_date = df["date"].max()
        print(f"📅 Available data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"📊 Total data points: {len(df)}")
    except FileNotFoundError:
        print("✗ Error: can't find TSLA_close.csv file.")
        return None, None, None
    
    # Get training start date
    while True:
        try:
            train_start = input(f"\n📈 Enter training start date (YYYY-MM-DD) [default: {min_date.strftime('%Y-%m-%d')}]: ").strip()
            if not train_start:
                train_start = min_date.strftime('%Y-%m-%d')
            
            train_start_dt = pd.to_datetime(train_start)
            if train_start_dt < min_date or train_start_dt > max_date:
                print(f"⚠️  Date must be between {min_date.strftime('%Y-%m-%d')} and {max_date.strftime('%Y-%m-%d')}")
                continue
            break
        except ValueError:
            print("⚠️  Invalid date format. Please use YYYY-MM-DD")
    
    # Get training end date
    while True:
        try:
            train_end = input(f"📉 Enter training end date (YYYY-MM-DD) [default: {max_date.strftime('%Y-%m-%d')}]: ").strip()
            if not train_end:
                train_end = max_date.strftime('%Y-%m-%d')
            
            train_end_dt = pd.to_datetime(train_end)
            if train_end_dt <= train_start_dt:
                print("⚠️  Training end date must be after start date")
                continue
            if train_end_dt > max_date:
                print(f"⚠️  Date must be before or equal to {max_date.strftime('%Y-%m-%d')}")
                continue
            break
        except ValueError:
            print("⚠️  Invalid date format. Please use YYYY-MM-DD")
    
    # Get prediction days
    while True:
        try:
            pred_days = input("🔮 Enter number of days to predict [default: 5]: ").strip()
            if not pred_days:
                pred_days = 5
            else:
                pred_days = int(pred_days)
            
            if pred_days <= 0 or pred_days > 30:
                print("⚠️  Prediction days must be between 1 and 30")
                continue
            break
        except ValueError:
            print("⚠️  Please enter a valid number")
    
    print(f"\nConfiguration:")
    print(f"   Training period: {train_start_dt.strftime('%Y-%m-%d')} to {train_end_dt.strftime('%Y-%m-%d')}")
    print(f"   Prediction days: {pred_days}")
    
    return df, train_start_dt, train_end_dt, pred_days, max_date

def main():
    print("=== ARIMA Model for Tesla Stock Price Prediction ===\n")
    
    # Ensure ticker is defined even when there is no test data
    ticker = "TSLA"
    # Ensure results directory exists under project root
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get user configuration
    result = get_user_input()
    if result is None:
        return
    
    df, train_start_dt, train_end_dt, pred_days, max_date = result
    
    # 2. Data preprocessing
    df["date"] = pd.to_datetime(df["date"])
    
    # Training data
    df_train = df.loc[(df["date"] >= train_start_dt) & (df["date"] <= train_end_dt)].reset_index(drop=True)
    
    # Calculate test period (next N days after training)
    test_start_dt = train_end_dt + pd.Timedelta(days=1)
    test_end_dt = test_start_dt + pd.Timedelta(days=pred_days-1)
    
    # Check if we have enough future data for testing
    if test_end_dt <= max_date:
        df_test = df.loc[(df["date"] >= test_start_dt) & (df["date"] <= test_end_dt)].reset_index(drop=True)
        has_test_data = True
        print(f"✓ Testing data: {len(df_test)} samples (for validation)")
    else:
        has_test_data = False
        print(f"⚠️  Not enough future data for {pred_days} days testing")
        print(f"   Will only generate predictions without validation")
    
    print(f"✓ Training data: {len(df_train)} samples")
    
    # 3. Extract the 'close' prices
    series = df_train['close'].values
    print(f"✓ Total training samples: {len(df_train)}")
    
    # Basic statistics
    print(f"\nPrice Statistics:")
    print(f"  Mean: ${df_train['close'].mean():.2f}")
    print(f"  Std: ${df_train['close'].std():.2f}")
    print(f"  Min: ${df_train['close'].min():.2f}")
    print(f"  Max: ${df_train['close'].max():.2f}")
    
    # 4. Check stationarity
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(df_train['close'].dropna())
    print(f"\nStationarity Test (ADF):")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    
    if result[1] > 0.05:
        print("  → The time series is NON-STATIONARY (p-value > 0.05)")
        print("  → We need to find the order of differencing (d)")
    else:
        print("  → The time series is STATIONARY (p-value <= 0.05)")
    
    # 5. Find optimal ARIMA parameters
    print("\n🔍 Finding optimal ARIMA parameters...")
    
    # Method 1: Try pmdarima if available
    try:
        from pmdarima import auto_arima
        print("  Trying pmdarima auto_arima...")
        model_auto = auto_arima(df_train['close'],
                               seasonal=True,
                               m=5,
                               trace=True,
                               error_action="ignore",
                               suppress_warnings=True)
        
        print(f"✓ Best model: {model_auto}")
        order = model_auto.order
        print(f"✓ Using auto_arima suggested order: {order}")
        
    except Exception as e:
        print(f"  ⚠️  pmdarima failed: {str(e)[:100]}...")
        
        # Method 2: Manual grid search for optimal parameters
        print("  🔍 Performing manual grid search...")
        best_aic = float('inf')
        best_order = (1, 0, 1)
        
        # Define parameter ranges to search
        p_range = range(0, 3)  # AR order
        d_range = range(0, 3)  # Differencing order  
        q_range = range(0, 3)  # MA order
        
        print("  Searching through parameter combinations...")
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(df_train['close'], order=(p, d, q))
                        fitted = model.fit()
                        aic = fitted.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            print(f"    New best: ARIMA{best_order} with AIC: {best_aic:.2f}")
                            
                    except:
                        continue
        
        order = best_order
        print(f"✓ Best manual order: ARIMA{order} with AIC: {best_aic:.2f}")
        
        # Method 3: Check if differencing is needed based on stationarity
        if result[1] > 0.05:  # Non-stationary
            print("  📊 Time series is non-stationary, considering differencing...")
            if order[1] == 0:  # No differencing in best order
                print("  ⚠️  Warning: Best order has no differencing but series is non-stationary")
                print("  💡 Consider using ARIMA(p, 1, q) for better results")
    
    # 6. Fit ARIMA model
    print(f"\nFitting ARIMA{order} model...")
    model = ARIMA(df_train['close'], order=order)
    fitted = model.fit()
    
    print("✓ Model fitted successfully!")
    print(f"  AIC: {fitted.aic:.2f}")
    print(f"  BIC: {fitted.bic:.2f}")
    print(f"  Log Likelihood: {fitted.llf:.2f}")
    
    # 7. Generate forecasts
    print(f"\n📊 Generating forecasts...")
    
    # Generate prediction for specified number of days
    forecast_pred = fitted.get_forecast(steps=pred_days)
    forecast_pred_mean = forecast_pred.predicted_mean
    
    # Create future dates for predictions
    from datetime import timedelta
    last_date = train_end_dt
    future_dates = [last_date + timedelta(days=i+1) for i in range(pred_days)]
    
    print("✓ Forecast generated successfully!")
    print(f"  Prediction period: {pred_days} days")
    
    # If we have test data, also generate forecast for comparison
    if has_test_data:
        forecast_test = fitted.get_forecast(steps=len(df_test))
        forecast_test_mean = forecast_test.predicted_mean
        forecast_test_mean.index = df_test.index
        print(f"  Test period forecast length: {len(forecast_test_mean)}")
    
    # Extended forecast for visualization (20 days)
    forecast_long = fitted.get_forecast(steps=20)
    forecast_long_mean = forecast_long.predicted_mean
    
    # 8. Performance evaluation
    print(f"\n📈 Model Performance Evaluation:")
    
    if has_test_data:
        y_true = df_test['close']
        y_pred = forecast_test_mean
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"  MAE:  {mae:.2f}")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        
        # 9. Create results matrix
        matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE'])
        matrix.loc['ARIMA'] = [mae, mse, rmse]
        
        print(f"\n📋 Results Matrix:")
        print(matrix)
        
        # 10. Save results
        ticker = "TSLA"
        results_pkl = results_dir / f"{ticker}_results_matrix.pkl"
        with open(results_pkl, "wb") as f:
            pickle.dump(matrix, f)
        print(f"\n💾 Results saved to {results_pkl}")
    else:
        print("  ⚠️  No test data available for performance evaluation")
        print("  📊 Only predictions generated")
    
    # 11. Display predictions
    print(f"\n🔮 Future Price Predictions:")
    print(f"Model: ARIMA{order}")
    print(f"Training period: {train_start_dt.strftime('%Y-%m-%d')} to {train_end_dt.strftime('%Y-%m-%d')}")
    print(f"Prediction period: {pred_days} days starting from {(train_end_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}")
    print("\n📊 Predicted Prices:")
    
    for i, (date, price) in enumerate(zip(future_dates, forecast_pred_mean), 1):
        print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    # Summary statistics
    print(f"\n📈 Prediction Summary:")
    print(f"  Average predicted price: ${forecast_pred_mean.mean():.2f}")
    print(f"  Min predicted price: ${forecast_pred_mean.min():.2f}")
    print(f"  Max predicted price: ${forecast_pred_mean.max():.2f}")
    print(f"  Price change from last training day: ${forecast_pred_mean.iloc[-1] - df_train['close'].iloc[-1]:.2f}")
    
    # 12. Visualization
    print(f"\n🎨 Creating visualization...")
    
    plt.figure(figsize=(16, 8))
    
    # Create date range for training data
    train_dates = df_train['date'].values
    if has_test_data:
        test_dates = df_test['date'].values
    
    # Plot training data with actual dates
    plt.plot(df_train['date'], df_train['close'], label='Training Data', 
             color='blue', linewidth=2, alpha=0.8)
    
    # Plot test data if available
    if has_test_data:
        plt.plot(df_test['date'], df_test['close'], label='Actual Test Data', 
                 color='red', linewidth=2, marker='o', markersize=6)
        
        # Plot forecast for test period
        plt.plot(df_test['date'], forecast_test_mean, label='ARIMA Forecast (Test)', 
                 color='green', linewidth=2, linestyle='--', marker='s', markersize=6)
        
        # Add confidence intervals for test period if available
        try:
            forecast_ci = fitted.get_forecast(steps=len(df_test)).conf_int()
            plt.fill_between(df_test['date'], 
                             forecast_ci.iloc[:, 0], 
                             forecast_ci.iloc[:, 1], 
                             alpha=0.2, color='green', label='95% Confidence Interval (Test)')
        except:
            pass
    
    # Plot main predictions
    plt.plot(future_dates, forecast_pred_mean, label=f'ARIMA Predictions ({pred_days} days)', 
             color='purple', linewidth=3, linestyle='-', marker='D', markersize=8)
    
    # Plot extended forecast for visualization
    extended_future_dates = [train_end_dt + pd.Timedelta(days=i+1) for i in range(len(forecast_long_mean))]
    plt.plot(extended_future_dates, forecast_long_mean, label='Extended Forecast (20 days)', 
             color='orange', linewidth=2, linestyle=':', alpha=0.7)
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title(f'{ticker} ARIMA Forecast Results', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_path = results_dir / f'{ticker}_ARIMA_forecast.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_path}")
    
    plt.show()
    
    # 13. Model diagnostics
    print(f"\n🔍 Model Diagnostics:")
    
    # Residuals analysis
    residuals = fitted.resid
    print(f"  Residuals Statistics:")
    print(f"    Mean: {residuals.mean():.4f}")
    print(f"    Std: {residuals.std():.4f}")
    print(f"    Min: {residuals.min():.4f}")
    print(f"    Max: {residuals.max():.4f}")
    
    # Check for autocorrelation in residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox
    try:
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        print(f"  Ljung-Box test p-value: {lb_test['lb_pvalue'].iloc[-1]:.4f}")
        if lb_test['lb_pvalue'].iloc[-1] > 0.05:
            print("    → Residuals are not autocorrelated (good!)")
        else:
            print("    → Residuals show autocorrelation (may need model improvement)")
    except:
        print("  Ljung-Box test not available")
    
    print(f"\n🎉 ARIMA analysis completed successfully!")
    print(f"   All major issues have been fixed:")
    print(f"   ✓ Column name consistency ('close' used throughout)")
    print(f"   ✓ Missing forecast_mean variable defined")
    print(f"   ✓ Proper error handling added")
    print(f"   ✓ Visualization code uncommented and fixed")
    print(f"   ✓ Model diagnostics included")

if __name__ == "__main__":
    main()
