"""
Tick Backtesting Framework

A modular backtesting system for quantitative trading strategies on tick data.
"""

from .data_loader import load_tick_data, preprocess_data, resample_to_kline
from .strategy import MACrossoverStrategy
from .backtest_engine import BacktestEngine
from .metrics import PerformanceMetrics

__version__ = "0.1.0"
__all__ = [
    "load_tick_data",
    "preprocess_data",
    "resample_to_kline",
    "MACrossoverStrategy",
    "BacktestEngine",
    "PerformanceMetrics",
]
