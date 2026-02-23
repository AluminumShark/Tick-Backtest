"""
Quick Test: Compare MA Crossover with and without RSI Filter

This script tests the impact of adding RSI filter to the MA Crossover strategy.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_tick_data, preprocess_data, resample_to_kline
from src.strategy import MACrossoverStrategy
from src.backtest_engine import BacktestEngine
from src.metrics import PerformanceMetrics


def test_strategy(data_file: str, fast_period: int, slow_period: int, leverage: float, use_rsi: bool):
    """
    Test strategy with or without RSI filter.

    Args:
        data_file: Path to data file
        fast_period: Fast MA period
        slow_period: Slow MA period
        leverage: Leverage multiplier
        use_rsi: Whether to use RSI filter (if False, RSI thresholds won't filter)

    Returns:
        Performance metrics dictionary
    """
    # Load data
    df = load_tick_data(data_file, verbose=False)
    df = preprocess_data(df, verbose=False)
    kline = resample_to_kline(df, timeframe='1h', verbose=False)

    # Calculate indicators
    strategy = MACrossoverStrategy(
        fast_period=fast_period,
        slow_period=slow_period,
        rsi_period=14
    )
    kline = strategy.calculate_indicators(kline, verbose=False)

    # If not using RSI, remove RSI filter by setting permissive thresholds
    if not use_rsi:
        # Regenerate signals without RSI filter
        kline['signal'] = 0

        # Death cross (sell signal) - no RSI filter
        kline.loc[
            (kline['ma_fast'] < kline['ma_slow']) &
            (kline['ma_fast'].shift(1) >= kline['ma_slow'].shift(1)),
            'signal'
        ] = -1

        # Golden cross (buy signal) - no RSI filter
        kline.loc[
            (kline['ma_fast'] > kline['ma_slow']) &
            (kline['ma_fast'].shift(1) <= kline['ma_slow'].shift(1)),
            'signal'
        ] = 1

    # Run backtest
    engine = BacktestEngine(leverage=leverage)
    kline = engine.run(kline, verbose=False)

    # Calculate metrics
    metrics_calc = PerformanceMetrics(initial_capital=10000)
    metrics = metrics_calc.calculate(kline, verbose=False)

    return metrics


def main():
    """Compare MA Crossover with and without RSI filter."""

    print("\n" + "=" * 80)
    print(" RSI ENHANCEMENT TEST - MA CROSSOVER STRATEGY")
    print("=" * 80)

    # Use current best parameters from GA
    FAST_PERIOD = 28
    SLOW_PERIOD = 100
    LEVERAGE = 1.72

    # Test on both 2024 and 2025 data
    datasets = {
        "2024 (Training)": "data/2024/XAUUSD_1y_24.csv",
        "2025 (Testing)": "data/2025/XAUUSD_1y_25.csv"
    }

    for dataset_name, data_file in datasets.items():
        print(f"\n{'=' * 80}")
        print(f" {dataset_name}")
        print("=" * 80)

        # Test without RSI filter
        print("\n[1] Original MA Crossover (No RSI Filter)")
        metrics_no_rsi = test_strategy(data_file, FAST_PERIOD, SLOW_PERIOD, LEVERAGE, use_rsi=False)
        print(f"   AR: {metrics_no_rsi['annual_return']:.2f}%")
        print(f"   MDD: {metrics_no_rsi['max_drawdown']:.2f}%")
        print(f"   Sharpe: {metrics_no_rsi['sharpe_ratio']:.2f}")
        print(f"   Sortino: {metrics_no_rsi['sortino_ratio']:.2f}")

        # Test with RSI filter
        print("\n[2] RSI-Enhanced MA Crossover (RSI Filter: 30-70)")
        metrics_with_rsi = test_strategy(data_file, FAST_PERIOD, SLOW_PERIOD, LEVERAGE, use_rsi=True)
        print(f"   AR: {metrics_with_rsi['annual_return']:.2f}%")
        print(f"   MDD: {metrics_with_rsi['max_drawdown']:.2f}%")
        print(f"   Sharpe: {metrics_with_rsi['sharpe_ratio']:.2f}")
        print(f"   Sortino: {metrics_with_rsi['sortino_ratio']:.2f}")

        # Compare improvements
        print("\n[3] Improvement (RSI vs No RSI)")
        ar_diff = metrics_with_rsi['annual_return'] - metrics_no_rsi['annual_return']
        mdd_diff = metrics_with_rsi['max_drawdown'] - metrics_no_rsi['max_drawdown']
        sharpe_diff = metrics_with_rsi['sharpe_ratio'] - metrics_no_rsi['sharpe_ratio']
        sortino_diff = metrics_with_rsi['sortino_ratio'] - metrics_no_rsi['sortino_ratio']

        print(f"   AR: {ar_diff:+.2f}% ({'+' if ar_diff > 0 else ''}{ar_diff / abs(metrics_no_rsi['annual_return']) * 100:.1f}%)")
        print(f"   MDD: {mdd_diff:+.2f}% ({'better' if mdd_diff > 0 else 'worse'})")
        print(f"   Sharpe: {sharpe_diff:+.2f} ({'+' if sharpe_diff > 0 else ''}{sharpe_diff / abs(metrics_no_rsi['sharpe_ratio']) * 100:.1f}%)")
        print(f"   Sortino: {sortino_diff:+.2f} ({'+' if sortino_diff > 0 else ''}{sortino_diff / abs(metrics_no_rsi['sortino_ratio']) * 100:.1f}%)")

        # Overall verdict
        if sharpe_diff > 0:
            print(f"\n   [+] RSI filter improves risk-adjusted returns!")
        else:
            print(f"\n   [-] RSI filter does not improve performance on this dataset")

    print("\n" + "=" * 80)
    print(" RECOMMENDATION")
    print("=" * 80)
    print("\nIf RSI filter improves Sharpe ratio on both datasets,")
    print("we should re-run GA optimization with the RSI-enhanced strategy.")
    print("\nIf RSI filter only helps on training set but hurts test set,")
    print("it's overfitting and we should stick with the original strategy.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
