"""
MA Crossover Strategy Backtest - Standalone Version

This is a complete, self-contained backtest script with all functionality in one file.
Perfect for quick parameter testing and experimentation.
"""

import pandas as pd
import numpy as np


# ============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS
# ============================================================================

# Data Configuration
DATA_FILE = "data/2024/XAUUSD_1y_24.csv"  # Change to 2024 or 2025
TIMEFRAME = '1h'  # Candlestick timeframe

# Strategy Parameters
FAST_PERIOD = 20   # Fast MA period
SLOW_PERIOD = 50   # Slow MA period

# Risk Management
LEVERAGE = 2.0           # Leverage multiplier (1.0 = no leverage)
INITIAL_CAPITAL = 10000  # Starting capital

# ============================================================================


def load_and_preprocess_data(file_path: str, timeframe: str = '1h') -> pd.DataFrame:
    """
    Load tick data and convert to OHLC candlesticks.

    Args:
        file_path: Path to CSV file
        timeframe: Candlestick timeframe (e.g., '1h', '4h', '1d')

    Returns:
        OHLC DataFrame
    """
    print("\n" + "=" * 60)
    print("Phase 1-2: Loading and Preprocessing Data")
    print("=" * 60)

    # Load tick data
    print(f"\nLoading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} tick records")

    # Convert timestamp and set index
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('date', inplace=True)
    df.drop(columns=['time'], inplace=True)

    # Calculate mid price and resample to OHLC
    print(f"\nResampling to {timeframe} candlesticks...")
    df['price'] = (df['bid'] + df['ask']) / 2
    kline = df['price'].resample(timeframe).ohlc()
    kline = kline.dropna()

    print(f"Generated {len(kline):,} candlesticks")
    print(f"Period: {kline.index[0]} to {kline.index[-1]}")

    return kline


def calculate_ma_indicators(kline: pd.DataFrame, fast_period: int, slow_period: int) -> pd.DataFrame:
    """
    Calculate moving averages and generate trading signals.

    Args:
        kline: OHLC DataFrame
        fast_period: Fast MA period
        slow_period: Slow MA period

    Returns:
        DataFrame with indicators and signals
    """
    print("\n" + "=" * 60)
    print("Phase 3: Calculating Indicators and Signals")
    print("=" * 60)

    # Calculate moving averages
    print(f"\nCalculating MA({fast_period}/{slow_period})...")
    kline['ma_fast'] = kline['close'].rolling(window=fast_period).mean()
    kline['ma_slow'] = kline['close'].rolling(window=slow_period).mean()
    kline = kline.dropna()

    print(f"Valid data points: {len(kline):,}")

    # Generate signals
    print("\nGenerating trading signals...")
    kline['signal'] = 0

    # Death cross (sell signal)
    kline.loc[
        (kline['ma_fast'] < kline['ma_slow']) &
        (kline['ma_fast'].shift(1) >= kline['ma_slow'].shift(1)),
        'signal'
    ] = -1

    # Golden cross (buy signal)
    kline.loc[
        (kline['ma_fast'] > kline['ma_slow']) &
        (kline['ma_fast'].shift(1) <= kline['ma_slow'].shift(1)),
        'signal'
    ] = 1

    # Print signal statistics
    signal_counts = kline['signal'].value_counts().sort_index()
    print(f"\nSignal Statistics:")
    print(f"  Golden Cross (Buy):  {signal_counts.get(1, 0)}")
    print(f"  Death Cross (Sell):  {signal_counts.get(-1, 0)}")
    print(f"  Total crossovers:    {len(kline[kline['signal'] != 0])}")

    return kline


def run_backtest(kline: pd.DataFrame, leverage: float = 1.0) -> pd.DataFrame:
    """
    Execute backtest and calculate returns.

    Args:
        kline: DataFrame with signals
        leverage: Leverage multiplier

    Returns:
        DataFrame with positions and returns
    """
    print("\n" + "=" * 60)
    print("Phase 4: Running Backtest")
    print("=" * 60)
    print(f"Leverage: {leverage}x")

    # Track positions (dual-direction: long +1 / short -1)
    print("\nTracking positions...")
    position = 0
    positions = []

    for signal in kline['signal']:
        if signal == 1:  # Golden cross - go long
            position = 1
        elif signal == -1:  # Death cross - go short
            position = -1
        positions.append(position)

    kline['positions'] = positions

    long_bars = (kline['positions'] == 1).sum()
    short_bars = (kline['positions'] == -1).sum()
    in_market = (kline['positions'] != 0).sum()
    print(f"Long bars:  {long_bars:,}")
    print(f"Short bars: {short_bars:,}")
    print(f"Market exposure: {in_market / len(kline) * 100:.2f}%")

    # Calculate returns on position flips
    print("\nCalculating returns...")
    kline['position_change'] = kline['positions'].diff().fillna(0)

    returns = []
    entry_prices = []
    entry_price = 0.0
    prev_pos = 0

    for i in range(len(kline)):
        pos = kline['positions'].iloc[i]
        price = kline['close'].iloc[i]
        ret = 0.0

        if pos != prev_pos:
            if prev_pos != 0 and entry_price > 0:
                ret = prev_pos * (price - entry_price) / entry_price * 100 * leverage
            entry_price = price if pos != 0 else 0.0

        returns.append(ret)
        entry_prices.append(entry_price)
        prev_pos = pos

    kline['returns'] = returns
    kline['entry_price'] = entry_prices

    # Print trade statistics
    trades = kline[kline['returns'] != 0]
    winning_trades = trades[trades['returns'] > 0]
    losing_trades = trades[trades['returns'] < 0]

    print(f"\nTrade Statistics:")
    print(f"  Total trades:    {len(trades)}")
    print(f"  Winning trades:  {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    print(f"  Losing trades:   {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
    print(f"\n  Average return:  {trades['returns'].mean():.2f}%")
    print(f"  Best trade:      {trades['returns'].max():.2f}%")
    print(f"  Worst trade:     {trades['returns'].min():.2f}%")
    print(f"  Cumulative:      {trades['returns'].sum():.2f}%")

    return kline


def calculate_performance_metrics(kline: pd.DataFrame, initial_capital: float = 10000) -> dict:
    """
    Calculate performance metrics.

    Args:
        kline: DataFrame with returns
        initial_capital: Starting capital

    Returns:
        Dictionary with metrics
    """
    print("\n" + "=" * 60)
    print("Phase 5: Calculating Performance Metrics")
    print("=" * 60)

    # Build equity curve
    print("\nBuilding equity curve...")
    equity = initial_capital
    equity_curve = []

    for i in range(len(kline)):
        if kline['returns'].iloc[i] != 0:
            equity += equity * kline['returns'].iloc[i] / 100
        equity_curve.append(equity)

    kline['equity'] = equity_curve

    # Calculate metrics
    print("\nCalculating metrics...")

    # Annual Return (AR)
    final_equity = kline['equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    days = (kline.index[-1] - kline.index[0]).days
    ar = ((1 + total_return) ** (365 / days) - 1) * 100

    # Maximum Drawdown (MDD)
    running_max = kline['equity'].cummax()
    drawdown = (kline['equity'] - running_max) / running_max * 100
    mdd = drawdown.min()

    # Sharpe Ratio
    trades = kline[kline['returns'] != 0]
    returns = trades['returns']
    annualization_factor = 252 / len(trades)
    sharpe = returns.mean() / returns.std() * np.sqrt(annualization_factor)

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino = returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)

    metrics = {
        'annual_return': ar,
        'max_drawdown': mdd,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'final_equity': final_equity,
        'total_return': total_return * 100
    }

    # Print results
    print("\n" + "=" * 60)
    print(" PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\nAnnual Return (AR):      {ar:.2f}%")
    print(f"Maximum Drawdown (MDD):  {mdd:.2f}%")
    print(f"Sharpe Ratio:            {sharpe:.2f}")
    print(f"Sortino Ratio:           {sortino:.2f}")
    print(f"\nInitial Capital:         ${initial_capital:,.2f}")
    print(f"Final Equity:            ${final_equity:,.2f}")
    print(f"Total Return:            {total_return * 100:.2f}%")

    return metrics, kline


def main():
    """Run the complete backtest pipeline."""
    print("\n" + "=" * 70)
    print(" MA CROSSOVER STRATEGY BACKTEST - STANDALONE VERSION")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Data File:      {DATA_FILE}")
    print(f"  Timeframe:      {TIMEFRAME}")
    print(f"  MA Periods:     {FAST_PERIOD}/{SLOW_PERIOD}")
    print(f"  Leverage:       {LEVERAGE}x")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")

    # Run backtest pipeline
    kline = load_and_preprocess_data(DATA_FILE, TIMEFRAME)
    kline = calculate_ma_indicators(kline, FAST_PERIOD, SLOW_PERIOD)
    kline = run_backtest(kline, LEVERAGE)
    metrics, kline = calculate_performance_metrics(kline, INITIAL_CAPITAL)

    print("\n" + "=" * 70)
    print(" BACKTEST COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")

    return kline, metrics


if __name__ == "__main__":
    kline, metrics = main()
