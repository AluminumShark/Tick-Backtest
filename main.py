"""
MA Crossover Strategy Backtest - Main Entry Point

This is the main entry point for running the backtest.
For modular code, see the src/ directory.
"""

from src.data_loader import load_tick_data, preprocess_data, resample_to_kline
from src.strategy import MACrossoverStrategy
from src.backtest_engine import BacktestEngine
from src.metrics import PerformanceMetrics


def main():
    """Run the complete MA Crossover backtest pipeline."""
    print("\n" + "=" * 60)
    print(" MA CROSSOVER STRATEGY BACKTEST - XAUUSD")
    print("=" * 60)

    # Configuration
    DATA_FILE = "data/2025/XAUUSD_1y_25.csv"  # Change to 2024 or 2025
    TIMEFRAME = '1h'
    FAST_PERIOD = 20
    SLOW_PERIOD = 50
    LEVERAGE = 2.0  # Leverage multiplier (1.0 = no leverage)
    INITIAL_CAPITAL = 10000

    # Phase 1-2: Load and preprocess data
    df = load_tick_data(DATA_FILE)
    df = preprocess_data(df)
    kline = resample_to_kline(df, timeframe=TIMEFRAME)

    # Phase 3: Calculate indicators and generate signals
    strategy = MACrossoverStrategy(fast_period=FAST_PERIOD, slow_period=SLOW_PERIOD)
    kline = strategy.calculate_indicators(kline)

    # Phase 4: Run backtest
    engine = BacktestEngine(leverage=LEVERAGE)
    kline = engine.run(kline)

    # Phase 5: Calculate performance metrics
    metrics = PerformanceMetrics(initial_capital=INITIAL_CAPITAL)
    results = metrics.calculate(kline)

    print("\n" + "=" * 60)
    print(" BACKTEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return kline, results


if __name__ == "__main__":
    kline, metrics = main()
