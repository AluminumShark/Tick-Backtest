"""
Backtesting Engine Module

Executes trading strategies and tracks positions, trades, and returns.
Per-bar return calculation with bid-ask spread transaction costs.
"""

import pandas as pd


class BacktestEngine:
    """
    Backtesting engine for executing trading strategies.

    Tracks position changes, calculates per-bar returns with leverage
    and transaction costs from bid-ask spread.
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

        # Task 4.2: Calculate per-bar returns with transaction costs
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
            print("\nTask 4.1: Position Tracking")
            print(f"Total hours in market: {kline['positions'].sum():,.0f}")
            print(f"Market exposure: {kline['positions'].sum() / len(kline) * 100:.2f}%")

        return kline

    def _calculate_returns(self, kline: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Calculate per-bar returns with leverage and bid-ask spread costs.

        Returns are computed every bar (not just at trade close):
        - Holding bars: price change * leverage
        - Entry bars: deduct half-spread cost
        - Exit bars: deduct half-spread cost

        Args:
            kline: DataFrame with positions
            verbose: Whether to print return statistics

        Returns:
            DataFrame with per-bar returns (in percentage)
        """
        # Per-bar price change (%)
        kline['pct_change'] = kline['close'].pct_change().fillna(0)

        # Position changes for identifying entries/exits
        kline['position_change'] = kline['positions'].diff().fillna(0)

        # Per-bar strategy returns:
        # Use shifted position — signal fires at bar close, returns start next bar
        kline['returns'] = (
            kline['pct_change'] * kline['positions'].shift(1).fillna(0) * self.leverage * 100
        )

        # Deduct bid-ask spread on entry and exit
        if 'spread' in kline.columns:
            spread_pct = (kline['spread'] / kline['close']) * 100
            entry_mask = kline['position_change'] == 1
            exit_mask = kline['position_change'] == -1
            kline.loc[entry_mask, 'returns'] -= spread_pct[entry_mask] / 2
            kline.loc[exit_mask, 'returns'] -= spread_pct[exit_mask] / 2

        # Track entry prices for reference
        kline['entry_price'] = kline['close'].where(kline['position_change'] == 1).ffill()

        # Always build trade log (needed by callers even when verbose=False)
        self._build_trade_log(kline)

        if verbose:
            self._print_trade_statistics(kline)

        return kline

    def _build_trade_log(self, kline: pd.DataFrame):
        """Build a summary of individual trades from per-bar returns."""
        entries = kline.index[kline['position_change'] == 1]
        exits = kline.index[kline['position_change'] == -1]

        trade_list = []
        for entry_time in entries:
            exit_candidates = exits[exits > entry_time]
            if len(exit_candidates) == 0:
                continue
            exit_time = exit_candidates[0]

            # Sum per-bar returns from entry bar through exit bar
            mask = (kline.index >= entry_time) & (kline.index <= exit_time)
            trade_return = kline.loc[mask, 'returns'].sum()

            trade_list.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': kline.loc[entry_time, 'close'],
                'exit_price': kline.loc[exit_time, 'close'],
                'returns': trade_return,
                'duration': exit_time - entry_time,
            })

        if trade_list:
            self.trades = pd.DataFrame(trade_list)
            self.winning_trades = self.trades[self.trades['returns'] > 0]
            self.losing_trades = self.trades[self.trades['returns'] <= 0]
        else:
            self.trades = pd.DataFrame()
            self.winning_trades = pd.DataFrame()
            self.losing_trades = pd.DataFrame()

    def _print_trade_statistics(self, kline: pd.DataFrame):
        """Print detailed trade statistics."""
        print("\nTask 4.2: Trade Statistics")

        if self.trades is None or self.trades.empty:
            print("No trades executed")
            return

        trades = self.trades
        n_total = len(trades)
        n_wins = len(self.winning_trades)
        n_losses = len(self.losing_trades)

        print(f"Total trades: {n_total}")
        print(f"Winning trades: {n_wins} ({n_wins/n_total*100:.1f}%)")
        print(f"Losing trades: {n_losses} ({n_losses/n_total*100:.1f}%)")
        print(f"\nAverage return per trade: {trades['returns'].mean():.2f}%")
        print(f"Best trade: {trades['returns'].max():.2f}%")
        print(f"Worst trade: {trades['returns'].min():.2f}%")
        print(f"Total cumulative return: {trades['returns'].sum():.2f}%")

        if 'spread' in kline.columns:
            avg_spread_pct = (kline['spread'] / kline['close']).mean() * 100
            print(f"\nAvg spread cost per trade: {avg_spread_pct:.4f}%")
