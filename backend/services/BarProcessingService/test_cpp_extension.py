"""
Test script to verify the C++ extension works correctly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# Import the bar processing service
from bar_processing_service import BarProcessingService

def create_sample_data(num_samples=2880):  # Default to 2 days (2880 minutes)
    """Create sample OHLCV data for testing with minute-level granularity."""
    # Create timestamps with minute-level granularity over 2 days
    start_date = datetime.now() - timedelta(days=2)
    timestamps = [start_date + timedelta(minutes=i) for i in range(num_samples)]
    
    # Seed the random number generator for reproducibility
    np.random.seed(42)
    
    # Create price data using pure random walk with no drift
    close_prices = [100.0]
    for i in range(1, num_samples):
        # True random walk (no drift)
        rnd_change = np.random.normal(0, 0.1)  # Lower volatility
        close_prices.append(close_prices[-1] + rnd_change)  # Simple addition instead of multiplication
    
    # Create OHLCV data
    data = {
        'timestamp': timestamps,
        'close': close_prices,
        'open': [price * (1 - 0.001 * np.random.random()) for price in close_prices],
        'high': [price * (1 + 0.0015 * np.random.random()) for price in close_prices],
        'low': [price * (1 - 0.0015 * np.random.random()) for price in close_prices],
        'volume': [100 * (1 + np.random.random()) for _ in range(num_samples)]
    }
    
    return pd.DataFrame(data)

def test_cpp_integration():
    """Test the C++ extension integration."""
    print("Creating sample data...")
    # Use fixed seed for reproducible results
    np.random.seed(42)
    df = create_sample_data()  # Use default 2-day minute data (2880 minutes)
    print(f"Sample data created with {len(df)} rows")
    
    # For debugging, print the range of close prices in the original data
    print(f"Original data close price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
    
    # Verify timestamp intervals in the original data
    if len(df) > 1:
        time_diff = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
        print(f"Time interval between data points: {time_diff} seconds (should be 60 for minute data)")
    
    # Initialize the bar processing service
    print("Initializing BarProcessingService...")
    service = BarProcessingService()
    
    # Load data
    print("Loading data into the service...")
    service.load_data(df)
    
    # Test time bars calculation first (should be most reliable)
    # For minute data, using 60 data points gives 1-hour bars
    print("\nCalculating time bars...")
    start_time = datetime.now()
    time_bar_interval = 30  # 30 minutes per bar (half hour bars)
    time_bars = service.calculate_time_bars(time_bar_interval)
    end_time = datetime.now()
    print(f"Time bars calculation completed in {(end_time - start_time).total_seconds():.4f} seconds")
    print(f"Generated {len(time_bars)} time bars")
    print("\nSample time bars:")
    pd.set_option('display.max_columns', None)
    # Print price range for debugging
    if len(time_bars) > 0:
        print(f"Time bars close price range: {time_bars['close'].min():.4f} to {time_bars['close'].max():.4f}")
    print(time_bars.head())
    
    # Test volume bars calculation
    print("\nCalculating volume bars...")
    start_time = datetime.now()
    volume_bars = service.calculate_volume_bars(1000)  # Reduced threshold for minute data
    end_time = datetime.now()
    print(f"Volume bars calculation completed in {(end_time - start_time).total_seconds():.4f} seconds")
    print(f"Generated {len(volume_bars)} volume bars")
    # Print price range for debugging
    print(f"Volume bars close price range: {volume_bars['close'].min():.4f} to {volume_bars['close'].max():.4f}")
    print("\nSample volume bars:")
    print(volume_bars.head())
    
    # Test tick bars calculation
    print("\nCalculating tick bars...")
    start_time = datetime.now()
    tick_bars = service.calculate_tick_bars(60)  # 60 ticks (minutes) per bar
    end_time = datetime.now()
    print(f"Tick bars calculation completed in {(end_time - start_time).total_seconds():.4f} seconds")
    print(f"Generated {len(tick_bars)} tick bars")
    # Print price range for debugging
    print(f"Tick bars close price range: {tick_bars['close'].min():.4f} to {tick_bars['close'].max():.4f}")
    print("\nSample tick bars:")
    print(tick_bars.head())
    
    # Test entropy bars calculation
    print("\nCalculating entropy bars...")
    start_time = datetime.now()
    entropy_bars = service.calculate_entropy_bars(0.2, window_size=60)  # Reduced threshold for minute data
    end_time = datetime.now()
    print(f"Entropy bars calculation completed in {(end_time - start_time).total_seconds():.4f} seconds")
    print(f"Generated {len(entropy_bars)} entropy bars")
    # Print price range for debugging
    print(f"Entropy bars close price range: {entropy_bars['close'].min():.4f} to {entropy_bars['close'].max():.4f}")
    print("\nSample entropy bars:")
    print(entropy_bars.head())
    
    # Test batch calculation
    print("\nTesting batch calculation...")
    params_list = [
        {'bar_type': 'volume', 'ratio': 1000},
        {'bar_type': 'tick', 'ratio': 60},
        {'bar_type': 'time', 'ratio': time_bar_interval},  # Same as above
        {'bar_type': 'entropy', 'ratio': 0.2, 'window_size': 60}
    ]
    
    start_time = datetime.now()
    batch_results = service.batch_calculate(params_list)
    end_time = datetime.now()
    print(f"Batch calculation completed in {(end_time - start_time).total_seconds():.4f} seconds")
    
    for key, bars in batch_results.items():
        print(f"Generated {len(bars)} bars for {key}")
    
    print("\nTest completed successfully!")
    return volume_bars, tick_bars, time_bars, entropy_bars, batch_results

def generate_realistic_price_charts(volume_bars, tick_bars, time_bars, entropy_bars):
    """Generate better visualizations of the bar data."""
    # Get sample data to show original data
    np.random.seed(42)  # Use same seed as in test_cpp_integration
    original_data = create_sample_data()
    
    # For debugging, ensure all datasets refer to the same original data
    print("Visualization data ranges:")
    print(f"Original data: {min(original_data['timestamp'])} to {max(original_data['timestamp'])}")
    print(f"Volume bars: {min(volume_bars['timestamp'])} to {max(volume_bars['timestamp'])}")
    print(f"Tick bars: {min(tick_bars['timestamp'])} to {max(tick_bars['timestamp'])}")
    if len(time_bars) > 0:
        print(f"Time bars: {min(time_bars['timestamp'])} to {max(time_bars['timestamp'])}")
    print(f"Entropy bars: {min(entropy_bars['timestamp'])} to {max(entropy_bars['timestamp'])}")
    
    # Get y-axis limits for consistent scaling
    y_min = min(original_data['close'].min(), volume_bars['close'].min(), 
                tick_bars['close'].min(), entropy_bars['close'].min())
    if len(time_bars) > 0:
        y_min = min(y_min, time_bars['close'].min())
        
    y_max = max(original_data['close'].max(), volume_bars['close'].max(), 
                tick_bars['close'].max(), entropy_bars['close'].max())
    if len(time_bars) > 0:
        y_max = max(y_max, time_bars['close'].max())
    
    # Add padding
    y_padding = (y_max - y_min) * 0.1
    y_min -= y_padding
    y_max += y_padding
    
    # Create figure with more subplots to include original data
    plt.figure(figsize=(15, 16))
    
    # Original data at the top - price
    ax1 = plt.subplot(5, 2, 1)
    plt.title("Original Data - Price")
    plt.plot(original_data['timestamp'], original_data['close'])
    plt.xticks(rotation=45)
    plt.ylabel('Price')
    plt.grid(True)
    plt.ylim(y_min, y_max)  # Consistent y-axis scaling
    
    # Original data at the top - volume (sample a portion to avoid overcrowding)
    plt.subplot(5, 2, 2)
    plt.title("Original Data - Volume")
    # Only show 6 hours of data (360 minutes) for the volume chart
    sample_size = min(360, len(original_data))
    plt.bar(original_data['timestamp'][:sample_size], original_data['volume'][:sample_size], width=0.01, alpha=0.7)
    plt.xticks(rotation=45)
    plt.ylabel('Volume')
    plt.grid(True)
    
    # Volume bars - price chart
    ax2 = plt.subplot(5, 2, 3)
    plt.title("Volume Bars - Price")
    plt.plot(volume_bars['timestamp'], volume_bars['close'])
    plt.xticks(rotation=45)
    plt.ylabel('Price')
    plt.grid(True)
    plt.ylim(y_min, y_max)  # Consistent y-axis scaling
    
    # Add reference line for original data close price at start
    first_timestamp = volume_bars['timestamp'].iloc[0]
    original_at_first = original_data[original_data['timestamp'] <= first_timestamp].iloc[-1]['close']
    plt.axhline(y=original_at_first, color='r', linestyle='--', alpha=0.5, 
                label=f'Original Price at Start: {original_at_first:.2f}')
    plt.legend()
    
    # Volume bars - volume chart
    plt.subplot(5, 2, 4)
    plt.title("Volume Bars - Volume")
    plt.bar(volume_bars['timestamp'], volume_bars['volume'], width=0.02)
    plt.xticks(rotation=45)
    plt.ylabel('Volume')
    plt.grid(True)
    
    # Tick bars
    ax3 = plt.subplot(5, 2, 5)
    plt.title("Tick Bars - Price (60 min/ticks per bar)")
    plt.plot(tick_bars['timestamp'], tick_bars['close'])
    plt.xticks(rotation=45)
    plt.ylabel('Price')
    plt.grid(True)
    plt.ylim(y_min, y_max)  # Consistent y-axis scaling
    
    # Add reference line for original data close price at start
    first_timestamp = tick_bars['timestamp'].iloc[0]
    original_at_first = original_data[original_data['timestamp'] <= first_timestamp].iloc[-1]['close']
    plt.axhline(y=original_at_first, color='r', linestyle='--', alpha=0.5, 
                label=f'Original Price at Start: {original_at_first:.2f}')
    plt.legend()
    
    # Tick bars - volume
    plt.subplot(5, 2, 6)
    plt.title("Tick Bars - Volume (60 min/ticks per bar)")
    plt.bar(tick_bars['timestamp'], tick_bars['volume'], width=0.02)
    plt.xticks(rotation=45)
    plt.ylabel('Volume')
    plt.grid(True)
    
    # Time bars
    ax4 = plt.subplot(5, 2, 7)
    plt.title("Time Bars - Price (30 min)")
    if len(time_bars) > 0:
        plt.plot(time_bars['timestamp'], time_bars['close'])
        
        # Add reference line for original data close price at start
        first_timestamp = time_bars['timestamp'].iloc[0]
        original_at_first = original_data[original_data['timestamp'] <= first_timestamp].iloc[-1]['close']
        plt.axhline(y=original_at_first, color='r', linestyle='--', alpha=0.5, 
                    label=f'Original Price at Start: {original_at_first:.2f}')
        plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Price')
    plt.grid(True)
    plt.ylim(y_min, y_max)  # Consistent y-axis scaling
    
    # Time bars - volume
    plt.subplot(5, 2, 8)
    plt.title("Time Bars - Volume (30 min)")
    if len(time_bars) > 0:
        plt.bar(time_bars['timestamp'], time_bars['volume'], width=0.02)
    plt.xticks(rotation=45)
    plt.ylabel('Volume')
    plt.grid(True)
    
    # Entropy bars
    ax5 = plt.subplot(5, 2, 9)
    plt.title("Entropy Bars - Price")
    plt.plot(entropy_bars['timestamp'], entropy_bars['close'])
    plt.xticks(rotation=45)
    plt.ylabel('Price')
    plt.grid(True)
    plt.ylim(y_min, y_max)  # Consistent y-axis scaling
    
    # Add reference line for original data close price at start
    first_timestamp = entropy_bars['timestamp'].iloc[0]
    original_at_first = original_data[original_data['timestamp'] <= first_timestamp].iloc[-1]['close']
    plt.axhline(y=original_at_first, color='r', linestyle='--', alpha=0.5, 
                label=f'Original Price at Start: {original_at_first:.2f}')
    plt.legend()
    
    # Entropy bars - volume
    plt.subplot(5, 2, 10)
    plt.title("Entropy Bars - Volume")
    plt.bar(entropy_bars['timestamp'], entropy_bars['volume'], width=0.02)
    plt.xticks(rotation=45)
    plt.ylabel('Volume')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("bar_types_detailed_comparison.png")
    plt.show()
    
    # Also create candlestick charts for each bar type
    from mplfinance.original_flavor import candlestick_ohlc
    import matplotlib.dates as mdates
    
    # Convert to OHLC format
    def prepare_ohlc_data(bars, title):
        # Convert datetime to float for plotting
        bars_plot = bars.copy()
        bars_plot['date_float'] = mdates.date2num(bars_plot['timestamp'])
        
        # Select relevant columns
        ohlc = bars_plot[['date_float', 'open', 'high', 'low', 'close']].values
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f"{title} - Candlestick Chart", fontsize=16)
        
        # Plot candlestick
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='g', colordown='r')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.set_ylabel('Price')
        ax1.grid(True)
        
        # Plot volume
        ax2.bar(bars_plot['date_float'], bars_plot['volume'], width=0.6, color='b', alpha=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(f"{title.lower().replace(' ', '_')}_candlestick.png")
        plt.close()
    
    # Create candlestick charts for each bar type (using a sample of 50 bars for clarity)
    prepare_ohlc_data(volume_bars.iloc[:50], "Volume Bars")
    prepare_ohlc_data(tick_bars.iloc[:50], "Tick Bars")
    if len(time_bars) > 0:
        prepare_ohlc_data(time_bars.iloc[:min(50, len(time_bars))], "Time Bars")
    prepare_ohlc_data(entropy_bars.iloc[:50], "Entropy Bars")
    
    print("All charts have been generated and saved.")

if __name__ == "__main__":
    volume_bars, tick_bars, time_bars, entropy_bars, batch_results = test_cpp_integration()
    
    # Generate detailed visualizations
    generate_realistic_price_charts(volume_bars, tick_bars, time_bars, entropy_bars) 