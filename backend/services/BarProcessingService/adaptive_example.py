import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from _bar_processor import BarType
from bar_calculator_wrapper import BarCalculator

# Set a consistent random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create sample data with a price shock in the middle
n_samples = 1000
timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')

# Create price series with a jump in the middle to test adaptivity
prices = np.zeros(n_samples)
for i in range(n_samples):
    if i < 300:
        # Low volatility region
        prices[i] = 100 + np.cumsum(np.random.normal(0, 0.03, i+1))[-1]
    elif i < 700:
        # High volatility region
        if i == 300:
            prices[i] = prices[i-1] + 1.0  # Price jump
        else:
            prices[i] = prices[i-1] + np.random.normal(0, 0.2, 1)[0]  # Higher volatility
    else:
        # Back to low volatility
        prices[i] = prices[i-1] + np.random.normal(0, 0.03, 1)[0]

# Generate volumes that also reflect volatility
volumes = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    if i < 300:
        volumes[i] = int(np.random.normal(50, 10))
    elif i < 700:
        volumes[i] = int(np.random.normal(150, 30))  # Higher volume during volatile periods
    else:
        volumes[i] = int(np.random.normal(50, 10))
        
volumes = np.abs(volumes)  # Ensure positive volumes

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'open': prices,
    'high': prices * 1.001,
    'low': prices * 0.999,
    'close': prices,
    'volume': volumes
})

# Print data summary
print(f"Created price series with {n_samples} samples")
print(f"Average volume in low volatility: ~50 units")
print(f"Average volume in high volatility: ~150 units")

# Create a BarCalculator instance
calculator = BarCalculator()

# Set up data
calculator.set_data(df)

# Parameters
lookback = 20  # Lookback window
volume_ratio = 2.5  # Higher ratio (> 1) for volume bars
tick_ratio = 3.0    # Higher ratio (> 1) for tick bars

# Function to run calculations and get results
def calculate_adaptive_bars(lookback, v_ratio, t_ratio):
    print("\nRunning calculations with adaptive thresholds...")
    
    # Calculate volume bars with adaptive threshold
    volume_bars = calculator.calculate_volume_bars(
        volume_ratio=v_ratio,
        lookback_window=lookback
    )
    
    # Calculate tick bars with adaptive threshold
    tick_bars = calculator.calculate_tick_bars(
        tick_ratio=t_ratio,
        lookback_window=lookback
    )
    
    # Print bar counts
    print(f"\nNumber of volume bars: {len(volume_bars)}")
    print(f"Number of tick bars: {len(tick_bars)}")
    
    return {
        'volume_bars': volume_bars,
        'tick_bars': tick_bars
    }

# Calculate bars
results = calculate_adaptive_bars(lookback, volume_ratio, tick_ratio)

# Visualize the results
print("\nCreating visualizations...")

# Create plots with 2 subplots in 1 row
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle(f'Adaptive Threshold Bars (Lookback={lookback})', fontsize=16)

# Plot 1: Volume Bars
ax1 = axes[0]
ax1.set_title(f'Volume Bars with Ratio={volume_ratio} (n={len(results["volume_bars"])})')
ax1.plot(df['timestamp'], df['close'], 'b-', linewidth=0.5, alpha=0.3)

# Highlight bar points
ax1.scatter(results['volume_bars']['timestamp'], 
          results['volume_bars']['close'], 
          c='g', s=25, alpha=0.8, marker='o', label='Bar Close')

# Mark volatility regions
ax1.axvspan(timestamps[300], timestamps[700], alpha=0.2, color='red', label='High Volatility')
ax1.set_ylabel('Price')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Tick Bars
ax2 = axes[1]
ax2.set_title(f'Tick Bars with Ratio={tick_ratio} (n={len(results["tick_bars"])})')
ax2.plot(df['timestamp'], df['close'], 'b-', linewidth=0.5, alpha=0.3)

# Highlight bar points
ax2.scatter(results['tick_bars']['timestamp'], 
          results['tick_bars']['close'], 
          c='r', s=25, alpha=0.8, marker='o', label='Bar Close')

# Mark volatility regions
ax2.axvspan(timestamps[300], timestamps[700], alpha=0.2, color='red', label='High Volatility')
ax2.set_ylabel('Price')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('adaptive_bars.png')
plt.show()

# Additional visualization: bar spacing in different volatility regions
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle('Bar Spacing in Different Volatility Regions', fontsize=16)

# Function to calculate time differences between consecutive bars
def calc_time_diffs(bars):
    bars['timestamp_dt'] = pd.to_datetime(bars['timestamp'])
    bars['time_diff'] = bars['timestamp_dt'].diff().dt.total_seconds() / 60  # in minutes
    return bars

# Process all bar sets
volume_with_diffs = calc_time_diffs(results['volume_bars'])
tick_with_diffs = calc_time_diffs(results['tick_bars'])

# Plot histograms of bar spacing
# Volume Bars
ax1 = axes[0]
ax1.set_title(f'Volume Bars: Time Between Bars (Ratio={volume_ratio})')
ax1.hist(volume_with_diffs['time_diff'].dropna(), bins=30, alpha=0.7)
ax1.set_xlabel('Minutes')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Tick Bars
ax2 = axes[1]
ax2.set_title(f'Tick Bars: Time Between Bars (Ratio={tick_ratio})')
ax2.hist(tick_with_diffs['time_diff'].dropna(), bins=30, alpha=0.7)
ax2.set_xlabel('Minutes')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adaptive_bars_spacing.png')
plt.show()

print("Visualizations complete. Images saved as 'adaptive_bars.png' and 'adaptive_bars_spacing.png'.") 