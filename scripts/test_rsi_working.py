"""
Quick test to verify RSI is working in the strategy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_tick_data, preprocess_data, resample_to_kline
from src.strategy import MACrossoverStrategy

# Load some data
print("Loading data...")
df = load_tick_data("data/2025/XAUUSD_1y_25.csv", verbose=False)
df = preprocess_data(df, verbose=False)
kline = resample_to_kline(df, timeframe='1h', verbose=False)

# Create strategy WITHOUT specifying rsi_period
print("\n" + "=" * 60)
print("Testing: Does RSI work automatically?")
print("=" * 60)

strategy = MACrossoverStrategy(fast_period=28, slow_period=100)
# Note: We did NOT pass rsi_period!

print("\nStrategy parameters:")
print(f"  fast_period: {strategy.fast_period}")
print(f"  slow_period: {strategy.slow_period}")
print(f"  rsi_period: {strategy.rsi_period}")  # Should be 14 (default)

# Calculate indicators
kline = strategy.calculate_indicators(kline, verbose=True)

# Check if RSI column exists
if 'rsi' in kline.columns:
    print("\n" + "=" * 60)
    print("SUCCESS! RSI is working!")
    print("=" * 60)
    print("\nRSI Statistics:")
    print(f"  Mean: {kline['rsi'].mean():.2f}")
    print(f"  Min: {kline['rsi'].min():.2f}")
    print(f"  Max: {kline['rsi'].max():.2f}")

    # Show first few rows
    print("\nFirst 5 rows with RSI:")
    print(kline[['close', 'ma_fast', 'ma_slow', 'rsi', 'signal']].head())
else:
    print("\nERROR: RSI is NOT working!")
