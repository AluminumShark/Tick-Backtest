"""
Performance Metrics Module

Calculates trading performance metrics including AR, MDD, Sharpe, and Sortino ratios.
"""

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Performance metrics calculator for backtesting results.

    Calculates Annual Return, Maximum Drawdown, Sharpe Ratio, and Sortino Ratio.
    """

    def __init__(self, initial_capital: float = 10000):
        """
        Initialize the performance metrics calculator.

        Args:
            initial_capital: Starting capital for the backtest
        """
        self.initial_capital = initial_capital
        self.metrics = {}

    def calculate(self, kline: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Calculate all performance metrics.

        Args:
            kline: DataFrame with returns
            verbose: Whether to print metrics

        Returns:
            Dictionary with all calculated metrics
        """
        if verbose:
            print("\n" + "=" * 50)
            print("Phase 5: Calculating Metrics")
            print("=" * 50)

        # Build equity curve
        kline = self._build_equity_curve(kline, verbose)

        # Calculate metrics
        self._calculate_annual_return(kline, verbose)
        self._calculate_max_drawdown(kline, verbose)
        self._calculate_sharpe_ratio(kline, verbose)
        self._calculate_sortino_ratio(kline, verbose)

        return self.metrics

    def _build_equity_curve(self, kline: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Build equity curve with compounding returns.

        Args:
            kline: DataFrame with returns
            verbose: Whether to print progress

        Returns:
            DataFrame with equity curve
        """
        if verbose:
            print("\nTask 5.1: Building Equity Curve")

        equity = self.initial_capital
        equity_curve = []

        for i in range(len(kline)):
            if kline['returns'].iloc[i] != 0:  # only update equity when closing position
                equity += equity * kline['returns'].iloc[i] / 100
            equity_curve.append(equity)

        kline['equity'] = equity_curve

        return kline

    def _calculate_annual_return(self, kline: pd.DataFrame, verbose: bool = True):
        """Calculate Annual Return (AR)."""
        if verbose:
            print("\nTask 5.2: Calculating Metrics")

        final_equity = kline['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        days = (kline.index[-1] - kline.index[0]).days
        ar = ((1 + total_return) ** (365 / days) - 1) * 100

        self.metrics['annual_return'] = ar

        if verbose:
            print(f"Annual Return(AR): {ar:.2f}%")

    def _calculate_max_drawdown(self, kline: pd.DataFrame, verbose: bool = True):
        """Calculate Maximum Drawdown (MDD)."""
        running_max = kline['equity'].cummax()
        drawdown = (kline['equity'] - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        self.metrics['max_drawdown'] = max_drawdown

        if verbose:
            print(f"Maximum Drawdown(MDD): {max_drawdown:.2f}%")

    def _calculate_sharpe_ratio(self, kline: pd.DataFrame, verbose: bool = True):
        """Calculate Sharpe Ratio."""
        trades = kline[kline['returns'] != 0]
        returns = trades['returns']
        annualization_factor = 252 / len(trades)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(annualization_factor)

        self.metrics['sharpe_ratio'] = sharpe_ratio

        if verbose:
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    def _calculate_sortino_ratio(self, kline: pd.DataFrame, verbose: bool = True):
        """Calculate Sortino Ratio."""
        trades = kline[kline['returns'] != 0]
        returns = trades['returns']
        downside_returns = returns[returns < 0]
        annualization_factor = 252 / len(trades)
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)

        self.metrics['sortino_ratio'] = sortino_ratio

        if verbose:
            print(f"Sortino Ratio: {sortino_ratio:.2f}")
