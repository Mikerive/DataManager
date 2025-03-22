"""
Example usage of the BarProcessingService with C++ implementation.

This script demonstrates how to:
1. Load data
2. Calculate different types of bars
3. Compare performance with Python implementation
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from bar_processing_service import BarProcessingService

def generate_sample_data(n_samples=10000):
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate timestamps
    start_date = pd.Timestamp('2023-01-01')
    timestamps = [start_date + pd.Timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate random price data with a trend
    closes = np.cumsum(np.random.normal(0, 1, n_samples)) + 100
    
    # Generate open, high, low based on close
    opens = closes - np.random.normal(0, 0.5, n_samples)
    highs = np.maximum(opens, closes) + np.random.normal(0, 0.5, n_samples)
    lows = np.minimum(opens, closes) - np.random.normal(0, 0.5, n_samples)
    
    # Generate random volume data
    volumes = np.abs(np.random.normal(1000, 300, n_samples))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df

def plot_bars(original_df, bars_df, title):
    """Plot the original data and the calculated bars."""
    plt.figure(figsize=(12, 6))
    
    # Plot original data
    plt.plot(original_df['timestamp'], original_df['close'], 'k-', alpha=0.3, label='Original')
    
    # Plot bar data
    plt.plot(bars_df['timestamp'], bars_df['close'], 'ro-', label='Bars')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data(50000)
    print(f"Generated {len(df)} data points")
    
    # Create a BarProcessingService instance
    service = BarProcessingService()
    
    # Load the data
    print("Loading data into service...")
    service.load_data(df)
    
    # Calculate volume bars
    print("\nCalculating volume bars...")
    volume_threshold = 5000
    start_time = time.time()
    volume_bars = service.calculate_volume_bars(volume_threshold)
    elapsed = time.time() - start_time
    print(f"Calculated {len(volume_bars)} volume bars in {elapsed:.4f} seconds")
    
    # Calculate tick bars
    print("\nCalculating tick bars...")
    tick_count = 100
    start_time = time.time()
    tick_bars = service.calculate_tick_bars(tick_count)
    elapsed = time.time() - start_time
    print(f"Calculated {len(tick_bars)} tick bars in {elapsed:.4f} seconds")
    
    # Calculate time bars
    print("\nCalculating time bars...")
    time_interval = 60  # 1 hour
    start_time = time.time()
    time_bars = service.calculate_time_bars(time_interval)
    elapsed = time.time() - start_time
    print(f"Calculated {len(time_bars)} time bars in {elapsed:.4f} seconds")
    
    # Calculate entropy bars
    print("\nCalculating entropy bars...")
    entropy_threshold = 0.8
    start_time = time.time()
    entropy_bars = service.calculate_entropy_bars(
        entropy_threshold, window_size=100, method="shannon"
    )
    elapsed = time.time() - start_time
    print(f"Calculated {len(entropy_bars)} entropy bars in {elapsed:.4f} seconds")
    
    # Batch calculation
    print("\nPerforming batch calculation...")
    params_list = [
        {'bar_type': 'volume', 'ratio': 5000},
        {'bar_type': 'tick', 'ratio': 100},
        {'bar_type': 'time', 'ratio': 60},
        {'bar_type': 'entropy', 'ratio': 0.8, 'window_size': 100, 'method': 'shannon'}
    ]
    start_time = time.time()
    batch_results = service.batch_calculate(params_list)
    elapsed = time.time() - start_time
    print(f"Batch calculated multiple bar types in {elapsed:.4f} seconds")
    for key, bars in batch_results.items():
        print(f"  {key}: {len(bars)} bars")
    
    # Compare results of individual calculations vs batch
    print("\nComparing individual vs batch results...")
    print(f"Volume bars: {len(volume_bars)} vs {len(batch_results['volume_5000.0'])}")
    print(f"Tick bars: {len(tick_bars)} vs {len(batch_results['tick_100.0'])}")
    print(f"Time bars: {len(time_bars)} vs {len(batch_results['time_60.0'])}")
    print(f"Entropy bars: {len(entropy_bars)} vs {len(batch_results['entropy_0.8'])}")
    
    # Plot some results
    try:
        # Plot a subset of the data for clarity
        plot_sample = df.iloc[:1000]
        
        # Plot volume bars
        volume_bars_sample = volume_bars[volume_bars['timestamp'] <= plot_sample['timestamp'].iloc[-1]]
        plot_bars(plot_sample, volume_bars_sample, f'Volume Bars (threshold={volume_threshold})')
        
        # Plot tick bars
        tick_bars_sample = tick_bars[tick_bars['timestamp'] <= plot_sample['timestamp'].iloc[-1]]
        plot_bars(plot_sample, tick_bars_sample, f'Tick Bars (ticks={tick_count})')
        
        # Plot time bars
        time_bars_sample = time_bars[time_bars['timestamp'] <= plot_sample['timestamp'].iloc[-1]]
        plot_bars(plot_sample, time_bars_sample, f'Time Bars (seconds={time_interval})')
        
        # Plot entropy bars
        entropy_bars_sample = entropy_bars[entropy_bars['timestamp'] <= plot_sample['timestamp'].iloc[-1]]
        plot_bars(plot_sample, entropy_bars_sample, f'Entropy Bars (threshold={entropy_threshold})')
    except ImportError:
        print("Matplotlib not available, skipping plots")

if __name__ == "__main__":
    main() 