"""
Test Leverage Effects on Different Datasets

This script tests various leverage levels on both 2024 and 2025 data
to find optimal parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_tick_data, preprocess_data, resample_to_kline
from src.strategy import MACrossoverStrategy
from src.backtest_engine import BacktestEngine
from src.metrics import PerformanceMetrics
import pandas as pd


def test_leverage_on_dataset(data_file: str, leverage: float, fast_period: int = 20, slow_period: int = 50):
    """
    Run backtest with specified leverage on a dataset.

    Args:
        data_file: Path to data file
        leverage: Leverage multiplier
        fast_period: Fast MA period
        slow_period: Slow MA period

    Returns:
        Dictionary with performance metrics
    """
    # Load and preprocess data
    df = load_tick_data(data_file, verbose=False)
    df = preprocess_data(df, verbose=False)
    kline = resample_to_kline(df, timeframe='1h', verbose=False)

    # Calculate indicators
    strategy = MACrossoverStrategy(fast_period=fast_period, slow_period=slow_period)
    kline = strategy.calculate_indicators(kline, verbose=False)

    # Run backtest
    engine = BacktestEngine(leverage=leverage)
    kline = engine.run(kline, verbose=False)

    # Calculate metrics
    metrics = PerformanceMetrics(initial_capital=10000)
    results = metrics.calculate(kline, verbose=False)

    return results


def main():
    """Test different leverage levels on both 2024 and 2025 data."""
    print("\n" + "=" * 80)
    print(" LEVERAGE TESTING - MA CROSSOVER STRATEGY")
    print("=" * 80)

    # Test configurations
    datasets = {
        "2024 (Training)": "data/2024/XAUUSD_1y_24.csv",
        "2025 (Testing)": "data/2025/XAUUSD_1y_25.csv"
    }

    leverages = [1.0, 1.5, 2.0, 2.5, 3.0]

    # Store results
    results = []

    print("\nTesting different leverage levels...\n")

    for dataset_name, data_file in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print('=' * 80)
        print(f"{'Leverage':<12} {'AR (%)':<12} {'MDD (%)':<12} {'Sharpe':<12} {'Sortino':<12}")
        print('-' * 80)

        for leverage in leverages:
            try:
                metrics = test_leverage_on_dataset(data_file, leverage)

                result = {
                    'dataset': dataset_name,
                    'leverage': leverage,
                    'ar': metrics['annual_return'],
                    'mdd': metrics['max_drawdown'],
                    'sharpe': metrics['sharpe_ratio'],
                    'sortino': metrics['sortino_ratio']
                }
                results.append(result)

                print(f"{leverage:<12.1f} {result['ar']:<12.2f} {result['mdd']:<12.2f} "
                      f"{result['sharpe']:<12.2f} {result['sortino']:<12.2f}")

            except Exception as e:
                print(f"{leverage:<12.1f} ERROR: {str(e)}")

    # Summary
    print("\n" + "=" * 80)
    print(" SUMMARY - BEST CONFIGURATIONS")
    print("=" * 80)

    df_results = pd.DataFrame(results)

    for dataset_name in datasets.keys():
        dataset_results = df_results[df_results['dataset'] == dataset_name]

        print(f"\n{dataset_name}:")
        best_ar = dataset_results.loc[dataset_results['ar'].idxmax()]
        best_sharpe = dataset_results.loc[dataset_results['sharpe'].idxmax()]
        best_sortino = dataset_results.loc[dataset_results['sortino'].idxmax()]

        print(f"  Best AR: {best_ar['leverage']:.1f}x leverage → AR: {best_ar['ar']:.2f}%, MDD: {best_ar['mdd']:.2f}%")
        print(f"  Best Sharpe: {best_sharpe['leverage']:.1f}x leverage → Sharpe: {best_sharpe['sharpe']:.2f}")
        print(f"  Best Sortino: {best_sortino['leverage']:.1f}x leverage → Sortino: {best_sortino['sortino']:.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
