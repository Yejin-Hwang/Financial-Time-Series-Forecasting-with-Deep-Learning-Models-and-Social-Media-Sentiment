#!/usr/bin/env python3
"""
Fixed ARIMA Model for Predicting Nvidia (NVDA) Stock Price
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

def _resolve_nvda_csv_path() -> str:
    """Resolve path to NVDA_close.csv under data/ exactly as required."""
    base_dir = Path(__file__).resolve().parent.parent  # project root
    target = base_dir / "data" / "NVDA_close.csv"
    if target.exists():
        return str(target)
    raise FileNotFoundError(f"Can't find required file: {target}")

def get_user_input():
    """Get user input for training period and prediction days"""
    print("=== ARIMA Model Configuration (NVDA) ===\n")
    
    # Get available date range
    try:
        df_path = _resolve_nvda_csv_path()
        df = pd.read_csv(df_path)
        # Normalize column names to lowercase (handle 'Date' -> 'date')
        df.columns = [str(c).lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min()
        max_date = df["date"].max()
        print(f"📅 Available data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"📊 Total data points: {len(df)}")
    except FileNotFoundError:
        print("✗ Error: can't find NVDA_close.csv file.")
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
    
    # Compute default end date so that training covers 96 rows from the chosen start (or as many as available)
    desired_rows = 96
    df_after_start = df.loc[df["date"] >= train_start_dt].sort_values("date")
    if len(df_after_start) >= 1:
        end_idx = min(desired_rows - 1, len(df_after_start) - 1)
        train_end_default_dt = pd.to_datetime(df_after_start.iloc[end_idx]["date"])  # 96th row (or last)
    else:
        train_end_default_dt = max_date

    # Get training end date (default = computed 96-row end)
    while True:
        try:
            prompt_default = train_end_default_dt.strftime('%Y-%m-%d')
            train_end = input(f"📉 Enter training end date (YYYY-MM-DD) [default: {prompt_default}]: ").strip()
            if not train_end:
                train_end_dt = train_end_default_dt
            else:
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
    # Seeding for reproducibility
    import random as _random
    import numpy as _np
    _random.seed(42)
    _np.random.seed(42)

    print("=== ARIMA Model for Nvidia (NVDA) Stock Price Prediction ===\n")
    
    # Ensure ticker is defined even when there is no test data
    ticker = "NVDA"
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
    
    # Calculate test period using business days (next pred_days rows strictly after train_end)
    future_slice = df.loc[df["date"] > train_end_dt].sort_values("date").head(int(pred_days))
    if len(future_slice) == int(pred_days):
        df_test = future_slice.reset_index(drop=True)
        has_test_data = True
        print(f"✓ Testing data: {len(df_test)} samples (for validation)")
    else:
        # Backshift training end to guarantee pred_days test rows when possible
        try:
            end_idx_series = df.index[df['date'] == train_end_dt]
            if len(end_idx_series) > 0:
                end_idx = int(end_idx_series[0])
                shortage = int(pred_days) - len(future_slice)
                new_end_idx = max(0, end_idx - shortage)
                # Keep train_end after start
                start_idx_series = df.index[df['date'] >= train_start_dt]
                start_idx = int(start_idx_series[0]) if len(start_idx_series) > 0 else 0
                if new_end_idx <= start_idx:
                    new_end_idx = min(end_idx, start_idx + max(1, len(df) - start_idx - int(pred_days)))
                new_train_end_dt = pd.to_datetime(df.loc[new_end_idx, 'date'])
                if new_train_end_dt != train_end_dt:
                    print(f"⚠️  Shifting training end earlier to {new_train_end_dt.date()} to ensure {pred_days} test rows")
                    train_end_dt = new_train_end_dt
                    df_train = df.loc[(df['date'] >= train_start_dt) & (df['date'] <= train_end_dt)].reset_index(drop=True)
                    future_slice = df.loc[df['date'] > train_end_dt].sort_values('date').head(int(pred_days))
            if len(future_slice) == int(pred_days):
                df_test = future_slice.reset_index(drop=True)
                has_test_data = True
                print(f"✓ Testing data: {len(df_test)} samples (for validation)")
            else:
                has_test_data = False
                df_test = pd.DataFrame(columns=df.columns)
                print(f"⚠️  Not enough future data for {pred_days} business days testing")
                print(f"   Will only generate predictions without validation")
        except Exception:
            has_test_data = False
            df_test = pd.DataFrame(columns=df.columns)
            print(f"⚠️  Not enough future data for {pred_days} business days testing")
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
    
    # Create future dates for predictions (align to business days if available)
    from datetime import timedelta
    if has_test_data and len(df_test) > 0:
        future_dates = pd.to_datetime(df_test["date"]).tolist()
    else:
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
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        # Directional Accuracy vs previous true close
        try:
            y_true_np = np.asarray(y_true, dtype=float)
            y_pred_np = np.asarray(y_pred, dtype=float)
            if y_true_np.size > 1 and y_pred_np.size > 1:
                da = float((np.sign(y_pred_np[1:] - y_true_np[:-1]) == np.sign(y_true_np[1:] - y_true_np[:-1])).mean())
            else:
                da = float('nan')
        except Exception:
            da = float('nan')

        print(f"  MAE:  {mae:.2f}")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  DA:   {da:.3f}")

        # 9. Create results matrix
        matrix = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'MAPE', 'DA'])
        matrix.loc['ARIMA'] = [mae, mse, rmse, mape, da]
        print(f"\n📋 Results Matrix:")
        print(matrix)
        
        # 10. Save results
        ticker = "NVDA"
        results_pkl = results_dir / f"{ticker}_results_matrix.pkl"
        with open(results_pkl, "wb") as f:
            pickle.dump(matrix, f)
        print(f"\n💾 Results saved to {results_pkl}")

        # Also update the NVDA CSV matrix so it's easy to view
        results_csv = results_dir / "result_matrix_nvda.csv"
        try:
            if results_csv.exists():
                global_matrix = pd.read_csv(results_csv, index_col=0)
            else:
                global_matrix = pd.DataFrame(columns=["MAE", "MSE", "RMSE", "MAPE", "DA"])
            if 'DA' not in global_matrix.columns:
                global_matrix = global_matrix.reindex(columns=["MAE","MSE","RMSE","MAPE","DA"])
            global_matrix.loc["ARIMA"] = [mae, mse, rmse, mape, da]
            desired_order = ['ARIMA', 'TimesFM', 'Chronos', 'TFT_baseline', 'TFT_Reddit']
            ordered = [i for i in desired_order if i in global_matrix.index]
            rest = [i for i in global_matrix.index if i not in desired_order]
            global_matrix = global_matrix.loc[ordered + rest]
            global_matrix.to_csv(results_csv)
            print(f"✓ Global results matrix updated: {results_csv}")
            print("\n📋 Global Results Matrix:")
            print(global_matrix)
        except Exception as e:
            print(f"⚠️  Failed to update global results matrix CSV: {e}")
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
    
    # Set style to match TFT with sentiment
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    plt.figure(figsize=(16, 8))

    # Index-based axis to remove calendar gaps
    train_values = df_train['close'].to_numpy()
    train_dates_list = pd.to_datetime(df_train['date']).tolist()
    x_train = list(range(len(train_values)))
    plt.plot(x_train, train_values, label='Training Data (Close Price)', color='blue', linewidth=2, alpha=0.7)

    tick_dates = train_dates_list.copy()
    x_pred_line = []

    if has_test_data and len(df_test) > 0:
        test_values = df_test['close'].to_numpy()
        x_test = list(range(len(x_train), len(x_train) + len(test_values)))
        plt.plot(x_test, test_values, label='Actual (Close Price)', color='green', linewidth=2, marker='s', markersize=6)

        plt.plot(x_test, forecast_test_mean.to_numpy(), label='Predictions', color='red', linewidth=3, linestyle='--', marker='o', markersize=8)
        try:
            forecast_ci = fitted.get_forecast(steps=len(df_test)).conf_int()
            plt.fill_between(x_test, forecast_ci.iloc[:, 0].to_numpy(), forecast_ci.iloc[:, 1].to_numpy(), alpha=0.2, color='red', label='95% Confidence Interval')
        except Exception:
            pass
        tick_dates += pd.to_datetime(df_test['date']).tolist()
        x_pred_line = x_test
    else:
        x_future = list(range(len(x_train), len(x_train) + len(forecast_pred_mean)))
        plt.plot(x_future, forecast_pred_mean.to_numpy(), label=f'ARIMA Predictions ({pred_days} days)', color='red', linewidth=3, linestyle='--', marker='o', markersize=8)
        x_pred_line = x_future
        tick_dates += future_dates

    # Separator at training end
    if len(x_train) > 0:
        plt.axvline(x=len(x_train) - 1, color='gray', linestyle='--', alpha=0.7, label='Training End / Prediction Start')

    # Date tick labels
    max_x = len(x_train) + (len(x_pred_line) if x_pred_line is not None else 0)
    if max_x > 0:
        tick_idx = list(np.linspace(0, max_x - 1, num=8, dtype=int))
        tick_labels = []
        for i in tick_idx:
            if i < len(tick_dates):
                try:
                    tick_labels.append(pd.to_datetime(tick_dates[i]).strftime('%Y-%m-%d'))
                except Exception:
                    tick_labels.append(str(tick_dates[i]))
            else:
                tick_labels.append('')
        plt.xticks(tick_idx, tick_labels, rotation=45)
    
    # Add title with training info - use actual training data count
    actual_training_days = len(df_train)
    title = f'ARIMA Model: {actual_training_days} Days Training + {pred_days} Days Prediction'
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot under results/
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
