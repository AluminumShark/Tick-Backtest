"""
Backtesting Engine Module

Executes trading strategies and tracks positions, trades, and returns.
"""

import pandas as pd


class BacktestEngine:
    """
    Backtesting engine for executing trading strategies.

    Tracks position changes, calculates returns, and analyzes trade statistics.
    """

    def __init__(self, leverage: float = 1.0):
        """
        Initialize the backtesting engine.

        Args:
            leverage: Leverage multiplier (default 1.0 = no leverage)
        """
        self.trades = None
        self.winning_trades = None
        self.losing_trades = None
        self.leverage = leverage

    def run(self, kline: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Execute backtest on kline data with signals.

        Args:
            kline: DataFrame with OHLC data and trading signals
            verbose: Whether to print backtest information

        Returns:
            DataFrame with positions and returns
        """
        if verbose:
            print("\n" + "=" * 50)
            print("Phase 4: Backtesting Strategy")
            print("=" * 50)
            print(f"Leverage: {self.leverage}x")

        # Task 4.1: Track positions
        kline = self._track_positions(kline, verbose)

        # Task 4.2: Calculate returns
        kline = self._calculate_returns(kline, verbose)

        return kline

    def _track_positions(self, kline: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Track position changes based on trading signals.

        Args:
            kline: DataFrame with trading signals
            verbose: Whether to print tracking information

        Returns:
            DataFrame with position column
        """
        position = 0
        positions = []

        for signal in kline['signal']:
            if signal == 1:  # Golden cross - buy
                position = 1
            elif signal == -1:  # Death cross - sell
                position = 0
            positions.append(position)

        kline['positions'] = positions

        if verbose:
            print(f"\nTask 4.1: Position Tracking")
            print(f"Total hours in market: {kline['positions'].sum():,.0f}")
            print(f"Market exposure: {kline['positions'].sum() / len(kline) * 100:.2f}%")

        return kline

    def _calculate_returns(self, kline: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Calculate returns for each trade with leverage applied.

        Args:
            kline: DataFrame with positions
            verbose: Whether to print return statistics

        Returns:
            DataFrame with returns (leveraged)
        """
        # Track position changes
        kline['position_change'] = kline['positions'].diff()

        # Track entry prices (forward fill until close)
        kline['entry_price'] = kline['close'].where(kline['position_change'] == 1).ffill()

        # Calculate returns only when closing position (in percentage) with leverage
        kline['returns'] = 0.0
        kline.loc[kline['position_change'] == -1, 'returns'] = (
            (kline['close'] - kline['entry_price']) / kline['entry_price'] * 100 * self.leverage
        )

        if verbose:
            self._print_trade_statistics(kline)

        return kline

    def _print_trade_statistics(self, kline: pd.DataFrame):
        """Print detailed trade statistics."""
        print(f"\nTask 4.2: Trade Statistics")

        self.trades = kline[kline['returns'] != 0]
        self.winning_trades = self.trades[self.trades['returns'] > 0]
        self.losing_trades = self.trades[self.trades['returns'] < 0]

        print(f"Total trades: {len(self.trades)}")
        print(f"Winning trades: {len(self.winning_trades)} ({len(self.winning_trades)/len(self.trades)*100:.1f}%)")
        print(f"Losing trades: {len(self.losing_trades)} ({len(self.losing_trades)/len(self.trades)*100:.1f}%)")
        print(f"\nAverage return per trade: {self.trades['returns'].mean():.2f}%")
        print(f"Best trade: {self.trades['returns'].max():.2f}%")
        print(f"Worst trade: {self.trades['returns'].min():.2f}%")
        print(f"Total cumulative return: {self.trades['returns'].sum():.2f}%")

        print(f"\nTop 5 Winning Trades:")
        print(self.trades.nlargest(5, 'returns')[['close', 'entry_price', 'returns']])

        print(f"\nTop 5 Losing Trades:")
        print(self.trades.nsmallest(5, 'returns')[['close', 'entry_price', 'returns']])
