# Tick-Backtest

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Asset-XAUUSD-FFD700?logo=bitcoin&logoColor=white)]()
[![uv](https://img.shields.io/badge/Package-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

**MA Crossover + RSI momentum filter strategy backtested on XAUUSD (Gold) tick-level data.**

Tick data (~3.6 M records) is resampled to 1-hour OHLC candles, then a Moving Average crossover strategy with RSI momentum filter generates long-only signals. Strategy parameters are optimized via a custom Genetic Algorithm and validated on out-of-sample data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AluminumShark/Tick-Backtest/blob/main/notebooks/backtest.ipynb)

---

## Highlights

- **Tick-level data** &mdash; Real bid/ask tick data (~3.6 M records) with realistic spread costs
- **Genetic Algorithm optimization** &mdash; Automated parameter search (64 pop, 128 gen, Sharpe fitness)
- **Out-of-sample validation** &mdash; Train on 2024, validate on 2025
- **Modular architecture** &mdash; Clean separation of data, strategy, engine, and metrics
- **No external trading libraries** &mdash; Built from scratch with pandas & numpy

---

## Strategy

| Component | Detail |
|-----------|--------|
| **Signal** | MA(12) / MA(34) Golden Cross & Death Cross |
| **Filter** | RSI(14) blocks buy when overbought (&ge;70), blocks sell when oversold (&le;30) |
| **Position** | Long-only &mdash; flat when no signal |
| **Timeframe** | 1H OHLC (resampled from tick data) |
| **Leverage** | 3.28x (GA-optimized) |
| **Costs** | Bid-ask spread deducted at entry & exit |

```
Buy:  Fast MA crosses above Slow MA  AND  RSI < 70
Sell: Fast MA crosses below Slow MA  AND  RSI > 30
```

---

## Results

Parameters optimized with Genetic Algorithm (64 population, early stopped at gen 34/128, MDD &le; -15% constraint).

| Metric | Training (2024) | Validation (2025) |
|--------|:---------------:|:-----------------:|
| **Annual Return** | 151.84 % | 104.87 % |
| **Max Drawdown** | -14.85 % | -22.24 % |
| **Sharpe Ratio** | 2.81 | 2.00 |
| **Sortino Ratio** | 3.96 | 2.85 |
| **Calmar Ratio** | 10.22 | 4.72 |
| **Win Rate** | 48.6 % | 47.0 % |
| **Profit Factor** | 2.62 | 1.80 |
| **Total Trades** | 72 | 66 |
| **Final Equity** | $23,108 | $17,960 |

> Initial capital: $10,000 &rarr; $23,108 (training) / $17,960 (validation)

---

## Architecture

```
tick-backtest/
├── src/
│   ├── data_loader.py          # Tick data I/O & OHLC resampling
│   ├── strategy.py             # MA Crossover + RSI signal generation
│   ├── backtest_engine.py      # Position tracking & return calculation
│   └── metrics.py              # AR, MDD, Sharpe, Sortino, Calmar
├── scripts/
│   ├── ga_optimizer.py         # Genetic Algorithm parameter optimizer
│   ├── backtest_standalone.py  # Standalone backtest runner
│   ├── test_rsi_working.py     # RSI verification
│   ├── test_rsi_enhancement.py # RSI enhancement tests
│   └── test_leverage.py        # Leverage sensitivity test
├── notebooks/
│   └── backtest.ipynb          # Full report with visualizations
├── data/                       # Tick CSVs (not tracked, ~190 MB)
├── output/                     # GA optimization results (JSON)
├── main.py                     # CLI entry point
├── pyproject.toml              # Project config & dependencies
└── uv.lock                     # Dependency lock file
```

### Pipeline

```
Tick CSV ──▶ data_loader ──▶ 1H OHLC ──▶ strategy ──▶ signals ──▶ backtest_engine ──▶ metrics
                                           MA + RSI      ▲                                │
                                                         │                                │
                                                   ga_optimizer ◄─────── fitness ◄────────┘
```

---

## Quick Start

### Prerequisites

- [Python 3.13+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager)

### Installation

```bash
git clone https://github.com/AluminumShark/Tick-Backtest.git
cd Tick-Backtest
uv sync
```

### Data

Place tick CSV files under `data/`:

```
data/
├── 2024/
│   └── XAUUSD_1y_24.csv
└── 2025/
    └── XAUUSD_1y_25.csv
```

> Tick data is not included in the repository due to file size (~190 MB).

### Run

```bash
# Run the backtest
uv run python main.py

# Run the GA optimizer
uv run python scripts/ga_optimizer.py
```

Or open the interactive notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AluminumShark/Tick-Backtest/blob/main/notebooks/backtest.ipynb)

---

## Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.13 |
| Data | pandas, numpy |
| Visualization | matplotlib |
| Package Manager | uv |
| Linting | Ruff |
| Type Checking | ty |

---

## License

This project is licensed under the [MIT License](LICENSE).
