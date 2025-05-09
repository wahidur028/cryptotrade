# Cryptocurrency Trading Strategies Backtesting Framework

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features](#features)
    - [Core Functionality](#core-functionality)
    - [Performance Metrics](#performance-metrics)
    - [Data Handling](#data-handling)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
- [Usage](#usage)
    - [Basic Execution](#basic-execution)
    - [Configuration Options](#configuration-options)
- [Results Analysis](#results-analysis)
    - [Output Files](#output-files)
    - [Metrics Captured](#metrics-captured)
- [Contact](#contact)

## Project Overview
A comprehensive backtesting framework for evaluating cryptocurrency trading strategies using academic research methodologies. Implements four distinct strategies with configurable parameters and robust performance metrics.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Backtesting](https://img.shields.io/badge/backtesting.py-0.3.3-orange)

## Project Structure
```bash
cryptotrade/
├── strategies/               # Strategy implementations
│   ├── strategy_sma.py
│   ├── strategy_macd.py
│   ├── strategy_buyandhold.py
│   └── strategy_randomwalk.py
├── results/                  # Backtest outputs
│   ├── metrics/              # Detailed metrics
│   └── optimized/            # Optimization results
├── data/                     # Cached market data
├── tests/                    # Unit tests
├── .gitignore                # Version control exclusions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

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

## Results Analysis
### Output Files
```bash
results/
├── [Strategy]_[Iterations]runs_metrics_[timestamp].csv
├── [Strategy]_[Iterations]runs_summary_[timestamp].csv
├── [Strategy]_optimized_[params]_[timestamp].csv
```
### Metrics Captured
Metric	Description
Return [%]	Total percentage return
Sharpe Ratio	Risk-adjusted return
Max. Drawdown [%]	Largest peak-to-trough decline
Win Rate [%]	Percentage of winning trades

## Contact
Author: Wahidur Rahman
Email: sm.wahidur@gm.gist.ac.kr
GitHub: @wahidur028
Project Link: https://github.com/wahidur028/cryptotrade
