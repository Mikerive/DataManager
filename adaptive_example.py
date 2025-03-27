import numpy as np
import pandas as pd
import time
from backend.services.BarProcessingService.bar_processing import BarProcessor, BarType

def generate_sample_data(num_points=1000, seed=42):
    """Generate synthetic price data for testing"""
    np.random.seed(seed)
    
    # Start with a random walk
    returns = np.random.normal(0, 0.01, num_points)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    timestamps = np.arange(num_points) * 60 * 1000  # 1-minute intervals in milliseconds
    opens = prices
    highs = prices * (1 + np.random.uniform(0, 0.005, num_points))
    lows = prices * (1 - np.random.uniform(0, 0.005, num_points))
    closes = prices * (1 + np.random.normal(0, 0.002, num_points))
    volumes = np.random.lognormal(10, 1, num_points)
    
    return timestamps, opens, highs, lows, closes, volumes

def main():
    # Create a bar processor
    processor = BarProcessor()
    
    # Generate sample data
    timestamps, opens, highs, lows, closes, volumes = generate_sample_data(num_points=10000)
    
    print("=== Volume Bars with Caching Example ===")
    print("Generating volume bars with adaptive thresholds...")
    
    # First run without caching (baseline)
    ticker_id = ""
    start_time = time.time()
    result = processor.calculate_bars(
        timestamps, opens, highs, lows, closes, volumes,
        BarType.VOLUME, threshold=1000.0,
        use_adaptive_threshold=True, lookback_window=50,
        ticker_id=ticker_id
    )
    elapsed_no_cache = time.time() - start_time
    
    print(f"Generated {len(result['timestamps'])} bars without caching in {elapsed_no_cache:.4f} seconds")
    
    # Run again with the same data, but now with caching
    ticker_id = "AAPL"
    start_time = time.time()
    result_cached_1 = processor.calculate_bars(
        timestamps, opens, highs, lows, closes, volumes,
        BarType.VOLUME, threshold=1000.0,
        use_adaptive_threshold=True, lookback_window=50,
        ticker_id=ticker_id
    )
    elapsed_first_cache = time.time() - start_time
    
    print(f"Generated {len(result_cached_1['timestamps'])} bars with first cache run in {elapsed_first_cache:.4f} seconds")
    
    # Run a third time with caching to see the speedup
    start_time = time.time()
    result_cached_2 = processor.calculate_bars(
        timestamps, opens, highs, lows, closes, volumes,
        BarType.VOLUME, threshold=1000.0,
        use_adaptive_threshold=True, lookback_window=50,
        ticker_id=ticker_id
    )
    elapsed_second_cache = time.time() - start_time
    
    print(f"Generated {len(result_cached_2['timestamps'])} bars with second cache run in {elapsed_second_cache:.4f} seconds")
    print(f"Cache speedup factor: {elapsed_no_cache / elapsed_second_cache:.2f}x\n")
    
    # Clear cache for specific ticker
    processor.clear_cache_for_ticker(ticker_id)
    print(f"Cleared cache for ticker {ticker_id}")
    
    # Run again after clearing the cache
    start_time = time.time()
    result_after_clear = processor.calculate_bars(
        timestamps, opens, highs, lows, closes, volumes,
        BarType.VOLUME, threshold=1000.0,
        use_adaptive_threshold=True, lookback_window=50,
        ticker_id=ticker_id
    )
    elapsed_after_clear = time.time() - start_time
    
    print(f"Generated {len(result_after_clear['timestamps'])} bars after clearing cache in {elapsed_after_clear:.4f} seconds")
    
    # Try a different ticker
    ticker_id_2 = "MSFT"
    start_time = time.time()
    result_diff_ticker = processor.calculate_bars(
        timestamps, opens, highs, lows, closes, volumes,
        BarType.VOLUME, threshold=1000.0,
        use_adaptive_threshold=True, lookback_window=50,
        ticker_id=ticker_id_2
    )
    elapsed_diff_ticker = time.time() - start_time
    
    print(f"Generated {len(result_diff_ticker['timestamps'])} bars for a different ticker in {elapsed_diff_ticker:.4f} seconds")
    
    # Clear all caches
    processor.clear_cache()
    print("Cleared all caches")
    
    print("\n=== Testing Different Bar Types with Caching ===")
    # Try tick bars
    ticker_id = "AAPL"
    start_time = time.time()
    result_tick = processor.calculate_bars(
        timestamps, opens, highs, lows, closes, volumes,
        BarType.TICK, threshold=100.0,
        use_adaptive_threshold=True, lookback_window=50,
        ticker_id=ticker_id
    )
    elapsed_tick = time.time() - start_time
    
    print(f"Generated {len(result_tick['timestamps'])} tick bars in {elapsed_tick:.4f} seconds")
    
    # Try entropy bars
    start_time = time.time()
    result_entropy = processor.calculate_bars(
        timestamps, opens, highs, lows, closes, volumes,
        BarType.ENTROPY, threshold=0.5,
        use_adaptive_threshold=True, lookback_window=50,
        window_size=20, entropy_method="shannon",
        ticker_id=ticker_id
    )
    elapsed_entropy = time.time() - start_time
    
    print(f"Generated {len(result_entropy['timestamps'])} entropy bars in {elapsed_entropy:.4f} seconds")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 