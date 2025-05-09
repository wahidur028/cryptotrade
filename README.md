# Cryptocurrency Trading Strategies Backtesting Framework

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Strategy Details](#strategy-details)
- [Results Analysis](#results-analysis)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
A comprehensive backtesting framework for evaluating cryptocurrency trading strategies using academic research methodologies. Implements four distinct strategies with configurable parameters and robust performance metrics.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Backtesting](https://img.shields.io/badge/backtesting.py-0.3.3-orange)

## Features
### Core Functionality
- Multiple backtest iterations for statistical significance
- Parameter optimization with constraints
- Transaction cost modeling (commission/slippage)
- Walk-forward testing capability

### Performance Metrics
- **Absolute Metrics**:
  - Total Return (%)
  - Max Drawdown (%)
  - Win Rate (%)
  - Number of Trades

- **Risk-Adjusted Metrics**:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Risk-Return Ratio

### Data Handling
- Automated OHLCV data fetching via OpenBB
- Local caching of historical data
- Flexible timeframes (daily/hourly)

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/wahidur028/cryptotrade.git
cd cryptotrade

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up OpenBB credentials (if needed)
obb account login
```
## Usage
### Basic Execution
```bash
# Run SMA strategy with default parameters
python strategy_sma.py

# Run MACD strategy with optimization
python strategy_macd.py
```
### Configuration Options
Each strategy file contains these configurable parameters:
```bash
# Backtest Parameters
NUM_ITERATIONS = 10       # Number of independent runs
OPTIMIZE = False          # Enable parameter optimization
CASH = 20000              # Initial capital (USD)
COMMISSION = 0.001        # 0.1% per trade

# Data Parameters
SYMBOL = 'BTC-USD'        # Trading pair
START_DATE = '2021-01-01' # Backtest start
END_DATE = '2024-01-01'   # Backtest end
INTERVAL = '1d'           # Timeframe ('1d', '4h', '1h')
```