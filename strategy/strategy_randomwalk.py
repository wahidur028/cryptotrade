"""
Cryptocurrency Random Walk Strategy Backtesting
Academic Research Implementation

This script implements and backtests a Random Walk trading strategy
for cryptocurrency trading.
"""
import os
import gc
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from openbb import obb
from backtesting import Backtest, Strategy
import random

# Suppress warnings
warnings.filterwarnings('ignore')

# ===========================
# Load Historical Crypto Data
# ===========================

def load_data(symbol='BTC-USD', start_date='2021-01-01', end_date='2024-01-01', interval='1d'):
    """
    Load historical cryptocurrency data using OpenBB's yfinance provider.
    """
    try:
        df = obb.crypto.price.historical(
            symbol=symbol,
            provider='yfinance',
            start_date=start_date,
            end_date=end_date,
            interval=interval
        ).to_df()

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Rename columns to match Backtesting.py convention
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        df.dropna(inplace=True)
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ===========================
# Random Walk Strategy
# ===========================

class RandomWalkStrategy(Strategy):
    """
    Random Walk Strategy implementation for Backtesting.py
    
    Parameters:
    -----------
    entry_prob : float
        Probability of entering a trade on any given day (default: 0.1)
    exit_prob : float
        Probability of exiting a trade on any given day (default: 0.1)
    """
    entry_prob = 0.1  # 10% chance to enter a trade each day
    exit_prob = 0.1   # 10% chance to exit a trade each day
    
    def init(self):
        """Initialize the strategy (required by backtesting.py)"""
        pass
    
    def next(self):
        """
        Generate random buy/sell signals.
        """
        # Random entry
        if random.random() < self.entry_prob:
            if not self.position:
                self.buy()
        
        # Random exit
        if random.random() < self.exit_prob:
            if self.position:
                self.sell()

# ===========================
# Run Backtest
# ===========================

def run_backtest(df, optimize=False):
    """
    Run the backtest and return performance statistics.
    """
    bt = Backtest(
        df, 
        RandomWalkStrategy, 
        cash=20000, 
        commission=.001,
        exclusive_orders=True
    )
    
    if optimize:
        stats = bt.optimize(
            entry_prob=[0.05, 0.1, 0.15, 0.2],
            exit_prob=[0.05, 0.1, 0.15, 0.2],
            maximize='Sharpe Ratio',
            constraint=lambda p: p.entry_prob <= p.exit_prob * 2,
            max_tries=100,
            return_heatmap=False
        )
    else:
        stats = bt.run()
    
    # Get the equity curve
    equity_curve = stats._equity_curve
    returns = pd.Series(equity_curve['Equity'].pct_change().dropna())
    daily_return_mean = returns.mean() * 100
    daily_return_std = returns.std() * 100
    
    # Display key metrics
    print("\n" + "="*50)
    print("RANDOM WALK BACKTEST RESULTS")
    print("="*50)
    print(f"Start/End: {df.index[0].date()} / {df.index[-1].date()}")
    print(f"Duration: {len(df)} days")
    print(f"Total Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print("-"*50)
    print("Daily Returns Statistics:")
    print(f"Mean: {daily_return_mean:.4f}%")
    print(f"Std Dev: {daily_return_std:.4f}%")
    print(f"Risk-Adjusted Return (Mean/Std): {daily_return_mean/daily_return_std:.4f}")
    print("="*50 + "\n")
    
    bt.plot()
    return stats

# ===========================
# Main Functionality
# ===========================

# if __name__ == "__main__":
#     # Load data (BTC-USD daily by default)
#     df = load_data()
    
#     if df is not None:
#         # Print dataset information
#         print(f"\n{' Backtest Summary ':=^50}")
#         print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
#         print(f"Calendar Days: {(df.index[-1] - df.index[0]).days}")
#         print(f"Trading Days: {len(df)}")
        
#         # Run backtest
#         stats = run_backtest(df, optimize=True)  # Set to False for default parameters

#         # Save results
#         results_dir = "results"
#         os.makedirs(results_dir, exist_ok=True)
        
#         # Generate filename components
#         strategy_name = RandomWalkStrategy.__name__.replace('Strategy', '')
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M")
#         sharpe = f"sharpe_{stats['Sharpe Ratio']:.2f}".replace('.', '')
#         returns = f"ret_{stats['Return [%]']:.0f}pc"
        
#         # Construct full path
#         results_filename = f"{strategy_name}_{returns}_{sharpe}_{timestamp}.csv"
#         results_path = os.path.join(results_dir, results_filename)
        
#         # Save and report
#         stats.to_csv(results_path)
#         print(f"\n{' Results Saved ':=^50}")
#         print(f"Location: {os.path.abspath(results_path)}")
#         print(f"Size: {os.path.getsize(results_path)/1024:.1f} KB")
#         print("=" * 50) 

#         gc.collect()

# ===========================
# Main Functionality for Multiple Iterations
# ===========================

if __name__ == "__main__":
    # Change these values as needed
    NUM_ITERATIONS = 10  # Set this to the number of runs you want
    OPTIMIZE = False      # Set to True if you want optimization (not recommended for multiple runs)
    
    # Load data (BTC-USD daily by default)
    df = load_data()
    
    if df is not None:
        # Print dataset information
        print(f"\n{' Backtest Summary ':=^50}")
        print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Calendar Days: {(df.index[-1] - df.index[0]).days}")
        print(f"Trading Days: {len(df)}")
        print(f"Number of Iterations: {NUM_ITERATIONS}")

        # Initialize metrics storage
        metrics = {
            'Returns': [],
            'Sharpe': [],
            'Drawdowns': [],
            'Win_Rate': [],
            'Trades': [],
            'Return_Std': [],
            'Sharpe_Std': []
        }

        # Run multiple backtests
        print(f"\n{' Running Random Walk Backtests ':=^50}")
        for i in range(NUM_ITERATIONS):
            print(f"Progress: {i+1}/{NUM_ITERATIONS}", end='\r')
            stats = run_backtest(df, optimize=OPTIMIZE)
            
            # Store metrics
            metrics['Returns'].append(stats['Return [%]'])
            metrics['Sharpe'].append(stats['Sharpe Ratio'])
            metrics['Drawdowns'].append(stats['Max. Drawdown [%]'])
            metrics['Win_Rate'].append(stats['Win Rate [%]'])
            metrics['Trades'].append(stats['# Trades'])
            metrics['Return_Std'].append(stats['Return [%]'] / stats['Exposure Time [%]'] if stats['Exposure Time [%]'] > 0 else 0)
            metrics['Sharpe_Std'].append(stats['Sharpe Ratio'] / stats['Exposure Time [%]'] if stats['Exposure Time [%]'] > 0 else 0)

        # Calculate statistics
        results = {
            'Iterations': NUM_ITERATIONS,
            'Avg Return': f"{np.mean(metrics['Returns']):.2f}%",
            'Return Std': f"{np.std(metrics['Returns']):.2f}",
            'Min Return': f"{np.min(metrics['Returns']):.2f}%",
            'Max Return': f"{np.max(metrics['Returns']):.2f}%",
            'Avg Sharpe': f"{np.mean(metrics['Sharpe']):.2f}",
            'Sharpe Std': f"{np.std(metrics['Sharpe']):.2f}",
            'Avg Drawdown': f"{np.mean(metrics['Drawdowns']):.2f}%",
            'Avg Win Rate': f"{np.mean(metrics['Win_Rate']):.2f}%",
            'Avg Trades': f"{np.mean(metrics['Trades']):.0f}",
            'Avg Return/Std': f"{np.mean(metrics['Return_Std']):.4f}",
            'Avg Sharpe/Std': f"{np.mean(metrics['Sharpe_Std']):.4f}"
        }

        # Display results
        print(f"\n{' Random Walk Results (' + str(NUM_ITERATIONS) + ' runs) ':=^50}")
        for k, v in results.items():
            print(f"{k}: {v}")
        print("="*50)

        # Save results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        strategy_name = RandomWalkStrategy.__name__.replace('Strategy', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Construct filenames
        metrics_filename = f"{strategy_name}_{NUM_ITERATIONS}runs_metrics_{timestamp}.csv"
        summary_filename = f"{strategy_name}_{NUM_ITERATIONS}runs_summary_{timestamp}.csv"
        
        # Save full metrics and summary
        pd.DataFrame(metrics).to_csv(os.path.join(results_dir, metrics_filename))
        pd.DataFrame([results]).to_csv(os.path.join(results_dir, summary_filename))
        
        print(f"\n{' Results Saved ':=^50}")
        print(f"Metrics File: {os.path.abspath(os.path.join(results_dir, metrics_filename))}")
        print(f"Summary File: {os.path.abspath(os.path.join(results_dir, summary_filename))}")
        print("=" * 50) 

        gc.collect()
