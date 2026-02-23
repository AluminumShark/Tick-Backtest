"""
Trading Strategy Module

Implements various trading strategies and technical indicators.
"""

import pandas as pd


class MACrossoverStrategy:
    """
    Moving Average Crossover Strategy (Golden Cross / Death Cross)

    Generates buy signals when fast MA crosses above slow MA (Golden Cross)
    and sell signals when fast MA crosses below slow MA (Death Cross).
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50, rsi_period: int = 14):
        """
        Initialize the MA Crossover Strategy.

        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            rsi_period: Period for RSI indicator (default 14)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period

    def calculate_indicators(self, kline: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Calculate moving averages and generate trading signals.

        Args:
            kline: OHLC DataFrame
            verbose: Whether to print calculation information

        Returns:
            DataFrame with indicators and signals
        """
        if verbose:
            print("\n" + "=" * 50)
            print("Phase 3: Calculating Indicators")
            print("=" * 50)
            print(f"\nTask 3.1: Calculating MA({self.fast_period}/{self.slow_period})")

        # Calculate moving averages
        kline['ma_fast'] = kline['close'].rolling(window=self.fast_period).mean()
        kline['ma_slow'] = kline['close'].rolling(window=self.slow_period).mean()

        if verbose:
            print(f"\nTask 3.2: Calculating RSI({self.rsi_period})")

        # Calculate RSI
        kline['rsi'] = self._calculate_rsi(kline['close'], self.rsi_period)

        # Remove rows with NaN
        kline = kline.dropna()

        if verbose:
            print(f"Valid data points after indicator calculation: {len(kline):,}")
            print("\nTask 3.3: Generating Trading Signals with RSI Filter")

        # Initialize signal column
        kline['signal'] = 0

        # Death cross (sell signal) - only when RSI > 30 (avoid oversold)
        kline.loc[
            (kline['ma_fast'] < kline['ma_slow']) &
            (kline['ma_fast'].shift(1) >= kline['ma_slow'].shift(1)) &
            (kline['rsi'] > 30),  # RSI filter: avoid selling in oversold condition
            'signal'
        ] = -1

        # Golden cross (buy signal) - only when RSI < 70 (avoid overbought)
        kline.loc[
            (kline['ma_fast'] > kline['ma_slow']) &
            (kline['ma_fast'].shift(1) <= kline['ma_slow'].shift(1)) &
            (kline['rsi'] < 70),  # RSI filter: avoid buying in overbought condition
            'signal'
        ] = 1

        if verbose:
            # Display signal statistics
            print("\nSignal Statistics:")
            signal_counts = kline['signal'].value_counts().sort_index()
            print(f"Death Cross (Sell): {signal_counts.get(-1, 0)}")
            print(f"No Signal: {signal_counts.get(0, 0)}")
            print(f"Golden Cross (Buy): {signal_counts.get(1, 0)}")
            print(f"\nTotal crossover points: {len(kline[kline['signal'] != 0])}")
            print(f"Average RSI: {kline['rsi'].mean():.2f}")
            print(f"RSI range: [{kline['rsi'].min():.2f}, {kline['rsi'].max():.2f}]")

        return kline

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss

        Args:
            prices: Price series
            period: RSI period (default 14)

        Returns:
            RSI values (0-100)
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Calculate exponential moving average of gains and losses
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
