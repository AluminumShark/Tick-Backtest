import numpy as np
import pandas as pd

def explore_data():
    """Phase 1: Explore Basic Information of Tick Data"""
    print("="*50)
    print("Phase 1: Data Exploration")
    print("="*50)

    # Read file
    print("\nReading tick data...")
    file_path = "data/2024/XAUUSD_1y_24.csv"
    df = pd.read_csv(file_path)

    print(f"Loaded {len(df):,} tick records")
    print(f"Date range: {pd.to_datetime(df['time'].min(), unit='s')} to {pd.to_datetime(df['time'].max(), unit='s')}")
    print(f"Columns: {', '.join(df.columns)}")

    # Check data quality
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")

    return df


def process_data(df):
    """Phase 2: Preprocess Tick Data"""
    print("\n" + "="*50)
    print("Phase 2: Data Preprocessing")
    print("="*50)

    # Convert time to datetime
    print("\nConverting timestamps to datetime...")
    df['date'] = pd.to_datetime(df['time'], unit='s')

    # Set index to datetime
    df.set_index('date', inplace=True)

    # Drop the time column
    df.drop(columns=['time'], inplace=True)

    print(f"Datetime index set successfully")

    return df

def resample_to_kline(df, timeframe='1h'):
    """Resample Tick Data to OHLC K-lines"""
    print(f"\nResampling to {timeframe} K-lines...")

    # Calculate mid-price (average of bid and ask)
    df['price'] = (df['bid'] + df['ask']) / 2

    # Resample to K-line with OHLC
    kline = df['price'].resample(timeframe).ohlc()

    # Remove rows with NaN (periods with no data)
    kline = kline.dropna()

    # Display results
    print(f"Tick data: {len(df):,} rows")
    print(f"K-lines: {len(kline):,} bars")
    print(f"Compression ratio: {len(kline)/len(df)*100:.2f}%")

    return kline

def calculate_indicators(kline, fast_period=20, slow_period=50):
    """Phase 3: Calculate Technical Indicators and Generate Signals"""
    print("\n" + "="*50)
    print(f"Phase 3: Calculating Indicators")
    print("="*50)

    # Task 3.1: Calculate moving averages
    print(f"\nTask 3.1: Calculating MA({fast_period}/{slow_period})")
    kline['ma_fast'] = kline['close'].rolling(window=fast_period).mean()
    kline['ma_slow'] = kline['close'].rolling(window=slow_period).mean()

    # Remove rows with NaN (first 50 rows won't have MA values)
    kline = kline.dropna()
    print(f"Valid data points after MA calculation: {len(kline):,}")

    # Task 3.2: Generate trading signals
    print(f"\nTask 3.2: Generating Trading Signals")
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

    # Display signal statistics
    print(f"\nSignal Statistics:")
    signal_counts = kline['signal'].value_counts().sort_index()
    print(f"Death Cross (Sell): {signal_counts.get(-1, 0)}")
    print(f"No Signal: {signal_counts.get(0, 0)}")
    print(f"Golden Cross (Buy): {signal_counts.get(1, 0)}")
    print(f"\nTotal crossover points: {len(kline[kline['signal'] != 0])}")

    return kline

def backtest_strategy(kline):
    """Phase 4: Backtest the MA Crossover Strategy"""
    print("\n" + "="*50)
    print("Phase 4: Backtesting Strategy")
    print("="*50)

    # Task 4.1: Track positions
    position = 0
    positions = []

    for signal in kline['signal']:
        if signal == 1:  # Golden cross - buy
            position = 1
        elif signal == -1:  # Death cross - sell
            position = 0
        positions.append(position)

    kline['positions'] = positions

    # Display position tracking results
    print(f"\nTask 4.1: Position Tracking")
    print(f"Total hours in market: {kline['positions'].sum():,.0f}")
    print(f"Market exposure: {kline['positions'].sum() / len(kline) * 100:.2f}%")

    # Task 4.2: Calculate returns
    kline['position_change'] = kline['positions'].diff()

    # Track entry prices (forward fill until close)
    kline['entry_price'] = kline['close'].where(kline['position_change'] == 1).ffill()

    # Calculate returns only when closing position (in percentage)
    kline['returns'] = 0.0
    kline.loc[kline['position_change'] == -1, 'returns'] = (
        (kline['close'] - kline['entry_price']) / kline['entry_price'] * 100
    )

    # Display trade statistics
    print(f"\nTask 4.2: Trade Statistics")
    trades = kline[kline['returns'] != 0]
    winning_trades = trades[trades['returns'] > 0]
    losing_trades = trades[trades['returns'] < 0]

    print(f"Total trades: {len(trades)}")
    print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
    print(f"\nAverage return per trade: {trades['returns'].mean():.2f}%")
    print(f"Best trade: {trades['returns'].max():.2f}%")
    print(f"Worst trade: {trades['returns'].min():.2f}%")
    print(f"Total cumulative return: {trades['returns'].sum():.2f}%")

    print(f"\nTop 5 Winning Trades:")
    print(trades.nlargest(5, 'returns')[['close', 'entry_price', 'returns']])

    print(f"\nTop 5 Losing Trades:")
    print(trades.nsmallest(5, 'returns')[['close', 'entry_price', 'returns']])

    return kline

def calculate_metrics(kline, initial_capital=10000):
    """Phase 5: Calculate Metrics"""
    print("\n" + "="*50)
    print("Phase 5: Calculating Metrics")
    print("="*50)

    # Task 5.1 Build equity curve
    print("\nTask 5.1: Building Equity Curve")
    equity = initial_capital
    equity_curve = []

    for i in range(len(kline)):
        if kline['returns'].iloc[i] != 0: # only update equity when closing position
            equity += equity * kline['returns'].iloc[i] / 100
        equity_curve.append(equity)
        
    kline['equity'] = equity_curve

    # Task 5.2 Calculate metrics
    print("\nTask 5.2: Calculating Metrics")

    # 1. Calculate Annual Return(AR)
    final_equity = kline['equity'].iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    days = (kline.index[-1] - kline.index[0]).days
    ar = ((1 + total_return)**(365/days) - 1) * 100

    print(f"Annual Return(AR): {ar:.2f}%")

    # 2. Calculate Maximum Drawdown(MDD)
    running_max = kline['equity'].cummax()
    drawdown = (kline['equity'] - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown(MDD): {max_drawdown:.2f}%")

    # 3. Calculate Sharpe Ratio
    trades = kline[kline['returns'] != 0]
    returns = trades['returns']
    annualization_factor = 252 / len(trades)
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(annualization_factor)
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # 4. Calculate Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    
    return kline

def main():
    """Main workflow for MA Crossover Strategy Backtest"""
    print("\n" + "="*60)
    print(" MA CROSSOVER STRATEGY BACKTEST - XAUUSD 2024")
    print("="*60)

    # Phase 1: Explore data
    df = explore_data()

    # Phase 2: Preprocess data
    df = process_data(df)
    kline = resample_to_kline(df, timeframe='1h')

    # Phase 3: Calculate indicators and generate signals
    kline = calculate_indicators(kline, fast_period=20, slow_period=50)

    # Phase 4: Backtest strategy
    kline = backtest_strategy(kline)

    # Phase 5: Calculate metrics
    kline = calculate_metrics(kline)

    print("\n" + "="*60)
    print(" BACKTEST COMPLETED SUCCESSFULLY!")
    print("="*60)

    return kline
        

if __name__ == "__main__":
    result = main()
