# MA Crossover Strategy Backtesting System

A quantitative trading backtesting system for analyzing MA (Moving Average) crossover strategies on tick-level data.

## 📊 Project Overview

This project implements a complete backtesting framework for evaluating moving average crossover strategies on XAUUSD (Gold) tick data from 2024. The system processes high-frequency tick data, generates trading signals, and calculates comprehensive performance metrics.

## 🎯 Strategy Description

**MA Crossover Strategy (Golden Cross / Death Cross)**

- **Golden Cross (Buy Signal)**: Fast MA crosses above Slow MA
- **Death Cross (Sell Signal)**: Fast MA crosses below Slow MA
- **Default Parameters**: MA(20/50) on 1-hour timeframe
- **Position**: Long-only (1 = in position, 0 = cash)

## 📈 Performance Metrics

Based on XAUUSD 2024 data (Jan - Nov):

| Metric | Value | Description |
|--------|-------|-------------|
| **Annual Return (AR)** | 21.36% | Annualized return rate |
| **Maximum Drawdown (MDD)** | -6.14% | Maximum peak-to-trough decline |
| **Sharpe Ratio** | 0.43 | Risk-adjusted return (all volatility) |
| **Sortino Ratio** | 1.61 | Risk-adjusted return (downside volatility only) |

### Trade Statistics

- **Total Trades**: 60
- **Win Rate**: 45.0%
- **Average Return per Trade**: 0.30%
- **Best Trade**: +6.13%
- **Worst Trade**: -1.79%
- **Total Cumulative Return**: 18.12%
- **Market Exposure**: 58.73%

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Tick-Backtest.git
cd Tick-Backtest

# Install dependencies using uv
uv sync

# Activate the virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/Mac:
source .venv/bin/activate
```

### Running the Backtest

```bash
python main.py
```

## 📁 Project Structure

```
Tick-Backtest/
├── data/
│   └── 2024/
│       └── XAUUSD_1y_24.csv       # Tick data (not included in repo)
├── main.py                         # Main backtesting script
├── pyproject.toml                  # Project dependencies (uv)
├── uv.lock                         # Lock file for reproducibility
├── .gitignore                      # Git ignore rules
├── .python-version                 # Python version specification
└── README.md                       # This file
```

## 🔧 Technical Details

### Data Processing Pipeline

1. **Phase 1: Data Exploration**
   - Load 1.7M+ tick records
   - Validate data quality
   - Check for missing values

2. **Phase 2: Data Preprocessing**
   - Convert Unix timestamps to datetime
   - Set datetime index
   - Resample to OHLC candlesticks

3. **Phase 3: Indicator Calculation**
   - Calculate fast/slow moving averages
   - Generate crossover signals
   - Detect golden/death crosses

4. **Phase 4: Backtesting**
   - Track position changes
   - Calculate trade returns
   - Analyze winning/losing trades

5. **Phase 5: Performance Metrics**
   - Build equity curve with compounding
   - Calculate Annual Return (AR)
   - Calculate Maximum Drawdown (MDD)
   - Calculate Sharpe Ratio
   - Calculate Sortino Ratio

### Key Implementation Details

- **Data Compression**: Tick data → OHLC (1.7M → 5.4K bars, 0.32% ratio)
- **Signal Generation**: Using pandas `.loc[]` for conditional assignment
- **Return Calculation**: Percentage returns with entry/exit price tracking
- **Equity Curve**: Compounding returns formula: `equity += equity * return / 100`
- **Annualization**:
  - AR: `((1 + total_return)^(365/days) - 1) * 100`
  - Sharpe/Sortino: `mean / std * sqrt(252 / num_trades)`

## 📊 Dependencies

```toml
numpy >= 1.26.0    # Numerical computations
pandas >= 3.0.1    # Data manipulation and time series
```

## 🎓 Key Learnings

This project demonstrates:
- ✅ Tick data processing and resampling
- ✅ Technical indicator implementation (MA crossover)
- ✅ Signal generation and position tracking
- ✅ Backtesting engine design
- ✅ Performance metrics calculation (AR, MDD, Sharpe, Sortino)
- ✅ Pandas advanced operations (`.loc`, `.iloc`, `.cummax()`, `.ffill()`)

## 📝 Notes

- **Sortino Ratio > Sharpe Ratio**: Indicates that most volatility comes from upward movements, with well-controlled downside risk
- **Win Rate < 50%**: Typical for trend-following strategies; relies on larger winners than losers (3.4:1 profit/loss ratio)
- **Low Maximum Drawdown**: -6.14% shows good risk management

## 🔍 Future Improvements

- [ ] Add visualization (equity curve, drawdown chart)
- [ ] Parameter optimization (grid search for MA periods)
- [ ] Multi-timeframe analysis (4h, daily)
- [ ] Additional indicators (RSI, MACD, Bollinger Bands)
- [ ] Walk-forward optimization
- [ ] Transaction cost modeling

## 📧 Contact

For questions or feedback about this project, please open an issue.

## 📄 License

This project is for educational and interview purposes.

---

**Built with Python 3.13 + pandas + numpy | Managed with uv**
