"""
Cryptocurrency MACD Strategy Backtesting
Academic Research Implementation

This script implements and backtests a MACD crossover strategy
for cryptocurrency trading, following academic research best practices.
"""
import os
import gc
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from openbb import obb
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Suppress warnings
warnings.filterwarnings('ignore')

# ===========================
# Load Historical Crypto Data
# ===========================

def load_data(symbol='BTC-USD', start_date='2021-01-01', end_date='2024-01-01', interval='1d'):
    """
    Load historical cryptocurrency data using OpenBB's yfinance provider.
    
    Parameters:
    -----------
    symbol : str
        Cryptocurrency pair symbol (e.g., 'BTC-USD')
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    interval : str
        Data frequency ('1d', '1h', etc.)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and OHLCV columns
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
# MACD Calculation
# ===========================

def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD components.
    
    Parameters:
    -----------
    close_prices : pd.Series
        Series of closing prices
    fast_period : int
        Fast EMA period (default: 12)
    slow_period : int
        Slow EMA period (default: 26)
    signal_period : int
        Signal line EMA period (default: 9)
    
    Returns:
    --------
    tuple
        (macd_line, signal_line, histogram)
    """
    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ===========================
# MACD Crossover Strategy
# ===========================

class MACDStrategy(Strategy):
    """
    MACD Crossover Strategy implementation for Backtesting.py
    
    Parameters:
    -----------
    fast_period : int
        Fast EMA period (default: 12)
    slow_period : int
        Slow EMA period (default: 26)
    signal_period : int
        Signal line period (default: 9)
    """
    # Standard MACD parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        """
        Initialize indicators and prepare the strategy.
        """
        # Calculate MACD components
        close_prices = pd.Series(self.data.Close)
        self.macd_line, self.signal_line, _ = calculate_macd(
            close_prices,
            self.fast_period,
            self.slow_period,
            self.signal_period
        )
        
        # Plot MACD and Signal Line
        self.I(lambda x: x, self.macd_line, name='MACD')
        self.I(lambda x: x, self.signal_line, name='Signal')

    def next(self):
        """
        Generate buy/sell signals based on MACD crossovers.
        """
        if crossover(self.macd_line, self.signal_line):
            if not self.position:
                self.buy()
        elif crossover(self.signal_line, self.macd_line):
            if self.position:
                self.sell()

# ===========================
# Run Backtest
# ===========================

def run_backtest(df, optimize=False):
    """
    Run the backtest and return performance statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input OHLCV data
    optimize : bool
        Whether to run parameter optimization
    
    Returns:
    --------
    pd.Series
        Backtest performance statistics
    """
    bt = Backtest(
        df, 
        MACDStrategy, 
        cash=20000, 
        commission=.001,
        exclusive_orders=True
    )
    
    if optimize:
        stats = bt.optimize(
            fast_period=range(10, 15, 1),
            slow_period=range(20, 30, 1),
            signal_period=range(7, 12, 1),
            maximize='Sharpe Ratio',
            constraint=lambda p: p.fast_period < p.slow_period,
            max_tries=500,
            return_heatmap=False   
        )
    else:
        stats = bt.run()
    
    # Get the equity curve from the stats instead of bt._equity_curve
    equity_curve = stats._equity_curve
    returns = pd.Series(equity_curve['Equity'].pct_change().dropna())
    daily_return_mean = returns.mean() * 100
    daily_return_std = returns.std() * 100
    
    # Display key metrics
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
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
# Main Functionality for Multiple Iterations
# ===========================

if __name__ == "__main__":
    # Change these values as needed
    NUM_ITERATIONS = 1  # Set this to the number of runs you want
    OPTIMIZE = True      # Set to True if you want optimization (not recommended for multiple runs)
    # Load data (BTC-USD daily by default)
    df = load_data()
    
    if df is not None:
        # Print dataset information
        print(f"\n{' Backtest Summary ':=^50}")
        print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Calendar Days: {(df.index[-1] - df.index[0]).days}")
        print(f"Trading Days: {len(df)}")

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

        # Run identical backtests
        print(f"\n{' Running ' + str(NUM_ITERATIONS) + ' Backtests ':=^50}")
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
        print(f"\n{' Average Results (' + str(NUM_ITERATIONS) + ' runs) ':=^50}")
        for k, v in results.items():
            print(f"{k}: {v}")
        print("="*50)

        # Save results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        strategy_name = MACDStrategy.__name__.replace('Strategy', '')
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