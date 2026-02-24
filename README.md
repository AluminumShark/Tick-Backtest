# Tick-Backtest

MA Crossover + RSI momentum filter strategy backtested on XAUUSD tick data.

**[View Notebook on Google Colab](https://colab.research.google.com/github/AluminumShark/Tick-Backtest/blob/main/notebooks/backtest.ipynb)**

## Strategy

- **Signal**: MA(12/34) Golden Cross / Death Cross
- **Filter**: RSI(14) blocks buy when overbought (>=70), blocks sell when oversold (<=30)
- **Position**: Long-only. Buy on Golden Cross, sell on Death Cross, flat in between.
- **Timeframe**: 1-hour OHLC resampled from tick data
- **Leverage**: 3.28x (GA-optimized)

## Results

Parameters optimized with Genetic Algorithm (64 population, early stop at gen 34/128).

| Metric | Training (2024) | Validation (2025) |
|--------|-----------------|-------------------|
| Annual Return | 151.84% | 104.87% |
| Max Drawdown | -14.85% | -22.24% |
| Sharpe Ratio | 2.81 | 2.00 |
| Sortino Ratio | 3.96 | 2.85 |

## Project Structure

```
tick-backtest/
├── src/
│   ├── data_loader.py        # Tick data loading and OHLC resampling
│   ├── strategy.py           # MA Crossover + RSI signal generation
│   ├── backtest_engine.py    # Position tracking and return calculation
│   └── metrics.py            # AR, MDD, Sharpe, Sortino
├── scripts/
│   ├── ga_optimizer.py       # Genetic algorithm parameter optimization
│   └── backtest_standalone.py
├── notebooks/
│   └── backtest.ipynb        # Full report with visualizations
├── data/                     # Tick data (not in repo)
├── output/                   # GA optimization results
└── main.py
```

## Quick Start

```bash
uv sync
uv run python main.py
```

## Tech Stack

Python 3.13 / pandas / numpy / matplotlib / uv
