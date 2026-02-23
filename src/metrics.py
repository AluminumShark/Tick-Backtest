"""
Performance Metrics Module

Calculates trading performance metrics including AR, MDD, Sharpe, and Sortino ratios.
Uses per-bar returns for accurate risk-adjusted metric calculation.
"""

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Performance metrics calculator for backtesting results.

    Calculates Annual Return, Maximum Drawdown, Sharpe Ratio, and Sortino Ratio
    from per-bar strategy returns.
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
            kline: DataFrame with per-bar returns
            verbose: Whether to print metrics

        Returns:
            Dictionary with all calculated metrics
        """
        if verbose:
            print("\n" + "=" * 50)
            print("Phase 5: Calculating Metrics")
            print("=" * 50)

        # Build equity curve from per-bar returns
        kline = self._build_equity_curve(kline, verbose)

        # Calculate metrics
        self._calculate_annual_return(kline, verbose)
        self._calculate_max_drawdown(kline, verbose)
        self._calculate_sharpe_ratio(kline, verbose)
        self._calculate_sortino_ratio(kline, verbose)

        return self.metrics

    def _build_equity_curve(self, kline: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Build equity curve by compounding per-bar returns.

        Args:
            kline: DataFrame with per-bar returns (in %)
            verbose: Whether to print progress

        Returns:
            DataFrame with equity curve
        """
        if verbose:
            print("\nTask 5.1: Building Equity Curve")

        # Compound per-bar returns into equity
        equity = self.initial_capital
        equity_curve = []

        for r in kline['returns']:
            equity *= (1 + r / 100)
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
        """
        Calculate annualized Sharpe Ratio from per-bar strategy returns.

        Sharpe = (mean_return / std_return) * sqrt(bars_per_year)
        """
        returns = kline['returns']

        # Annualization factor based on actual data frequency
        total_days = (kline.index[-1] - kline.index[0]).days
        bars_per_year = len(kline) / (total_days / 365.25)

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (mean_return / std_return) * np.sqrt(bars_per_year)

        self.metrics['sharpe_ratio'] = sharpe_ratio

        if verbose:
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    def _calculate_sortino_ratio(self, kline: pd.DataFrame, verbose: bool = True):
        """
        Calculate annualized Sortino Ratio from per-bar strategy returns.

        Sortino = (mean_return / downside_deviation) * sqrt(bars_per_year)
        Downside deviation uses all observations, replacing gains with 0.
        """
        returns = kline['returns']

        total_days = (kline.index[-1] - kline.index[0]).days
        bars_per_year = len(kline) / (total_days / 365.25)

        mean_return = returns.mean()

        # Proper downside deviation: clip positive returns to 0, then RMS
        downside = returns.clip(upper=0)
        downside_std = np.sqrt((downside ** 2).mean())

        if downside_std == 0:
            sortino_ratio = 0.0
        else:
            sortino_ratio = (mean_return / downside_std) * np.sqrt(bars_per_year)

        self.metrics['sortino_ratio'] = sortino_ratio

        if verbose:
            print(f"Sortino Ratio: {sortino_ratio:.2f}")
