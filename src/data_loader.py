"""
Data Loading and Preprocessing Module

Handles tick data loading, validation, and resampling to OHLC candlesticks.
"""

import os

import pandas as pd


def load_tick_data(file_path: str | os.PathLike, verbose: bool = True) -> pd.DataFrame:
    """
    Load tick data from CSV file.

    Args:
        file_path: Path to the tick data CSV file
        verbose: Whether to print loading information

    Returns:
        DataFrame with tick data
    """
    if verbose:
        print("=" * 50)
        print("Phase 1: Data Exploration")
        print("=" * 50)
        print("\nReading tick data...")

    df = pd.read_csv(file_path)

    if verbose:
        print(f"Loaded {len(df):,} tick records")
        print(f"Date range: {pd.to_datetime(df['time'].min(), unit='s')} to {pd.to_datetime(df['time'].max(), unit='s')}")
        print(f"Columns: {', '.join(df.columns)}")

        # Check data quality
        missing = df.isnull().sum().sum()
        print(f"Missing values: {missing}")

    return df


def preprocess_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Preprocess tick data by converting timestamps and setting index.

    Args:
        df: Raw tick data DataFrame
        verbose: Whether to print processing information

    Returns:
        Preprocessed DataFrame with datetime index
    """
    if verbose:
        print("\n" + "=" * 50)
        print("Phase 2: Data Preprocessing")
        print("=" * 50)
        print("\nConverting timestamps to datetime...")

    # Convert time to datetime
    df['date'] = pd.to_datetime(df['time'], unit='s')

    # Set index to datetime
    df.set_index('date', inplace=True)

    # Drop the time column
    df.drop(columns=['time'], inplace=True)

    if verbose:
        print("Datetime index set successfully")

    return df


def resample_to_kline(df: pd.DataFrame, timeframe: str = '1h', verbose: bool = True) -> pd.DataFrame:
    """
    Resample tick data to OHLC candlesticks.

    Args:
        df: Preprocessed tick data with datetime index
        timeframe: Resampling timeframe (e.g., '1h', '4h', '1d')
        verbose: Whether to print resampling information

    Returns:
        DataFrame with OHLC data
    """
    if verbose:
        print(f"\nResampling to {timeframe} K-lines...")

    # Calculate mid-price (average of bid and ask)
    df['price'] = (df['bid'] + df['ask']) / 2

    # Resample to K-line with OHLC
    kline = df['price'].resample(timeframe).ohlc()

    # Resample bid/ask close for realistic transaction cost modeling
    kline['bid_close'] = df['bid'].resample(timeframe).last()
    kline['ask_close'] = df['ask'].resample(timeframe).last()
    kline['spread'] = kline['ask_close'] - kline['bid_close']

    # Remove rows with NaN (periods with no data)
    kline = kline.dropna()

    if verbose:
        tick_count = len(df)
        bar_count = len(kline)
        avg_spread = kline['spread'].mean()
        avg_spread_pct = (kline['spread'] / kline['close']).mean() * 100
        print(f"Tick data: {tick_count:,} rows")
        print(f"K-lines: {bar_count:,} bars")
        print(f"Compression ratio: {bar_count/tick_count*100:.3f}%")
        print(f"Average spread: {avg_spread:.4f} ({avg_spread_pct:.4f}% of price)")

    return kline
