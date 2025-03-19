# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
import pandas as pd
from datetime import datetime
from libc.math cimport log, fabs
from cpython.datetime cimport datetime
import cython

# Type definitions
ctypedef double float_t
ctypedef long long int64_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_volume_bars_cy(np.ndarray[float_t, ndim=1] prices, 
                             np.ndarray[float_t, ndim=1] volumes, 
                             np.ndarray timestamps,  # Cannot type this directly in Cython
                             float_t ratio, 
                             int avg_window=200):
    """
    Calculate volume bars from raw price data - Cython optimized version.
    
    Args:
        prices: NumPy array of prices
        volumes: NumPy array of volumes
        timestamps: NumPy array of timestamps (datetime64)
        ratio: Volume threshold as a multiple of average volume
        avg_window: Window size for calculating average volume
        
    Returns:
        DataFrame with volume bars
    """
    cdef int n_points = len(prices)
    if n_points == 0:
        return pd.DataFrame()
    
    # Initialize arrays for bar data
    cdef list timestamps_out = []
    cdef list start_times = []
    cdef list end_times = []
    cdef list opens = []
    cdef list highs = []
    cdef list lows = []
    cdef list closes = []
    cdef list volumes_out = []
    
    # Initialize bar tracking variables
    cdef float_t cum_volume = 0.0
    cdef float_t bar_open = prices[0]
    cdef float_t bar_high = prices[0]
    cdef float_t bar_low = prices[0]
    cdef float_t bar_volume = 0.0
    cdef float_t threshold = 0.0
    cdef float_t vol_mavg = 0.0
    cdef int bar_start_idx = 0
    cdef int i, j
    
    # Calculate initial moving average of volume
    cdef int vol_window = min(avg_window, n_points)
    
    # Calculate initial volume moving average
    vol_mavg = 0.0
    for i in range(vol_window):
        vol_mavg += volumes[i]
    vol_mavg /= vol_window
    threshold = vol_mavg * ratio
    
    # Process each data point
    for i in range(n_points):
        # Update bar stats
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])
        bar_volume += volumes[i]
        cum_volume += volumes[i]
        
        # Update moving average periodically
        if i % vol_window == 0 and i > 0:
            vol_mavg = 0.0
            for j in range(max(0, i-vol_window), i):
                vol_mavg += volumes[j]
            vol_mavg /= vol_window
            threshold = vol_mavg * ratio
        
        # Check if volume threshold is reached
        if cum_volume >= threshold and threshold > 0:
            # Add bar to results
            timestamps_out.append(timestamps[i])
            start_times.append(timestamps[bar_start_idx])
            end_times.append(timestamps[i])
            opens.append(bar_open)
            highs.append(bar_high)
            lows.append(bar_low)
            closes.append(prices[i])
            volumes_out.append(bar_volume)
            
            # Reset bar tracking for next bar
            if i < n_points - 1:
                bar_start_idx = i + 1
                bar_open = prices[bar_start_idx]
                bar_high = bar_open
                bar_low = bar_open
                bar_volume = 0.0
                cum_volume = 0.0
    
    # Create DataFrame from collected data
    if len(timestamps_out) > 0:
        result = pd.DataFrame({
            'timestamp': timestamps_out,
            'start_time': start_times,
            'end_time': end_times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes_out,
            'bar_type': 'volume',
            'ratio': ratio
        })
        # Make timestamp the index and return
        result.set_index('timestamp', inplace=True)
        return result
    
    return pd.DataFrame()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_tick_bars_cy(np.ndarray[float_t, ndim=1] prices, 
                          np.ndarray[float_t, ndim=1] volumes, 
                          np.ndarray timestamps,
                          int tick_size):
    """
    Calculate tick bars from raw price data - Cython optimized version.
    
    Args:
        prices: NumPy array of prices
        volumes: NumPy array of volumes
        timestamps: NumPy array of timestamps
        tick_size: Number of ticks to aggregate into one bar
        
    Returns:
        DataFrame with tick bars
    """
    cdef int n_points = len(prices)
    if n_points == 0 or tick_size <= 0:
        return pd.DataFrame()
    
    # Initialize arrays for bar data
    cdef list timestamps_out = []
    cdef list start_times = []
    cdef list end_times = []
    cdef list opens = []
    cdef list highs = []
    cdef list lows = []
    cdef list closes = []
    cdef list volumes_out = []
    
    # Initialize bar tracking variables
    cdef int tick_count = 0
    cdef float_t bar_open = prices[0]
    cdef float_t bar_high = prices[0]
    cdef float_t bar_low = prices[0]
    cdef float_t bar_volume = 0.0
    cdef int bar_start_idx = 0
    cdef int i
    
    # Process each data point
    for i in range(n_points):
        # Update bar stats
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])
        bar_volume += volumes[i]
        tick_count += 1
        
        # Check if tick threshold is reached
        if tick_count >= tick_size:
            # Add bar to results
            timestamps_out.append(timestamps[i])
            start_times.append(timestamps[bar_start_idx])
            end_times.append(timestamps[i])
            opens.append(bar_open)
            highs.append(bar_high)
            lows.append(bar_low)
            closes.append(prices[i])
            volumes_out.append(bar_volume)
            
            # Reset bar tracking for next bar
            if i < n_points - 1:
                bar_start_idx = i + 1
                bar_open = prices[bar_start_idx]
                bar_high = bar_open
                bar_low = bar_open
                bar_volume = 0.0
                tick_count = 0
    
    # Create DataFrame from collected data
    if len(timestamps_out) > 0:
        result = pd.DataFrame({
            'timestamp': timestamps_out,
            'start_time': start_times,
            'end_time': end_times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes_out,
            'bar_type': 'tick',
            'ratio': float(tick_size)
        })
        # Make timestamp the index and return
        result.set_index('timestamp', inplace=True)
        return result
    
    return pd.DataFrame()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_price_bars_cy(np.ndarray[float_t, ndim=1] prices, 
                           np.ndarray[float_t, ndim=1] volumes, 
                           np.ndarray timestamps,
                           int price_threshold):
    """
    Calculate price bars from raw price data - Cython optimized version.
    
    Args:
        prices: NumPy array of prices
        volumes: NumPy array of volumes
        timestamps: NumPy array of timestamps
        price_threshold: Minimum absolute price change to create a new bar
        
    Returns:
        DataFrame with price bars
    """
    cdef int n_points = len(prices)
    if n_points == 0 or price_threshold <= 0:
        return pd.DataFrame()
    
    # Initialize arrays for bar data
    cdef list timestamps_out = []
    cdef list start_times = []
    cdef list end_times = []
    cdef list opens = []
    cdef list highs = []
    cdef list lows = []
    cdef list closes = []
    cdef list volumes_out = []
    
    # Initialize bar tracking variables
    cdef float_t bar_open = prices[0]
    cdef float_t bar_high = prices[0]
    cdef float_t bar_low = prices[0]
    cdef float_t bar_volume = 0.0
    cdef float_t last_bar_price = prices[0]
    cdef int bar_start_idx = 0
    cdef int i
    
    # Process each data point
    for i in range(n_points):
        # Update bar stats
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])
        bar_volume += volumes[i]
        
        # Check if price threshold is reached
        if fabs(prices[i] - last_bar_price) >= price_threshold:
            # Add bar to results
            timestamps_out.append(timestamps[i])
            start_times.append(timestamps[bar_start_idx])
            end_times.append(timestamps[i])
            opens.append(bar_open)
            highs.append(bar_high)
            lows.append(bar_low)
            closes.append(prices[i])
            volumes_out.append(bar_volume)
            
            # Reset bar tracking for next bar
            if i < n_points - 1:
                bar_start_idx = i + 1
                bar_open = prices[bar_start_idx]
                bar_high = bar_open
                bar_low = bar_open
                bar_volume = 0.0
                last_bar_price = prices[i]
    
    # Create DataFrame from collected data
    if len(timestamps_out) > 0:
        result = pd.DataFrame({
            'timestamp': timestamps_out,
            'start_time': start_times,
            'end_time': end_times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes_out,
            'bar_type': 'price',
            'ratio': float(price_threshold)
        })
        # Make timestamp the index and return
        result.set_index('timestamp', inplace=True)
        return result
    
    return pd.DataFrame()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_dollar_bars_cy(np.ndarray[float_t, ndim=1] prices, 
                            np.ndarray[float_t, ndim=1] volumes, 
                            np.ndarray timestamps,
                            int dollar_threshold):
    """
    Calculate dollar bars from raw price data - Cython optimized version.
    
    Args:
        prices: NumPy array of prices
        volumes: NumPy array of volumes
        timestamps: NumPy array of timestamps
        dollar_threshold: Dollar volume threshold to create a new bar
        
    Returns:
        DataFrame with dollar bars
    """
    cdef int n_points = len(prices)
    if n_points == 0 or dollar_threshold <= 0:
        return pd.DataFrame()
    
    # Initialize arrays for bar data
    cdef list timestamps_out = []
    cdef list start_times = []
    cdef list end_times = []
    cdef list opens = []
    cdef list highs = []
    cdef list lows = []
    cdef list closes = []
    cdef list volumes_out = []
    
    # Initialize bar tracking variables
    cdef float_t cum_dollar_volume = 0.0
    cdef float_t bar_open = prices[0]
    cdef float_t bar_high = prices[0]
    cdef float_t bar_low = prices[0]
    cdef float_t bar_volume = 0.0
    cdef float_t dollar_volume
    cdef int bar_start_idx = 0
    cdef int i
    
    # Process each data point
    for i in range(n_points):
        # Calculate dollar volume for this tick
        dollar_volume = prices[i] * volumes[i]
        
        # Update bar stats
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])
        bar_volume += volumes[i]
        cum_dollar_volume += dollar_volume
        
        # Check if dollar threshold is reached
        if cum_dollar_volume >= dollar_threshold:
            # Add bar to results
            timestamps_out.append(timestamps[i])
            start_times.append(timestamps[bar_start_idx])
            end_times.append(timestamps[i])
            opens.append(bar_open)
            highs.append(bar_high)
            lows.append(bar_low)
            closes.append(prices[i])
            volumes_out.append(bar_volume)
            
            # Reset bar tracking for next bar
            if i < n_points - 1:
                bar_start_idx = i + 1
                bar_open = prices[bar_start_idx]
                bar_high = bar_open
                bar_low = bar_open
                bar_volume = 0.0
                cum_dollar_volume = 0.0
    
    # Create DataFrame from collected data
    if len(timestamps_out) > 0:
        result = pd.DataFrame({
            'timestamp': timestamps_out,
            'start_time': start_times,
            'end_time': end_times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes_out,
            'bar_type': 'dollar',
            'ratio': float(dollar_threshold)
        })
        # Make timestamp the index and return
        result.set_index('timestamp', inplace=True)
        return result
    
    return pd.DataFrame()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float_t calculate_shannon_entropy_cy(np.ndarray[float_t, ndim=1] price_changes, int window_size):
    """Optimized Shannon entropy calculation using Cython"""
    cdef int n = len(price_changes)
    if n < window_size:
        return 0.0
    
    # Use the last window_size elements
    cdef np.ndarray[float_t, ndim=1] window = price_changes[n-window_size:n]
    
    # Calculate absolute values and sum
    cdef np.ndarray[float_t, ndim=1] abs_window = np.abs(window)
    cdef float_t sum_abs = np.sum(abs_window)
    
    if sum_abs == 0:
        return 0.0
    
    # Normalize and calculate entropy
    cdef np.ndarray[float_t, ndim=1] prob = abs_window / sum_abs
    cdef np.ndarray[float_t, ndim=1] nonzero_prob = prob[prob > 0]
    
    if len(nonzero_prob) == 0:
        return 0.0
    
    cdef np.ndarray[float_t, ndim=1] log_prob = np.log2(nonzero_prob)
    return -np.sum(nonzero_prob * log_prob)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float_t calculate_tsallis_entropy_cy(np.ndarray[float_t, ndim=1] price_changes, int window_size, float_t q=1.5):
    """Optimized Tsallis entropy calculation using Cython"""
    cdef int n = len(price_changes)
    if n < window_size:
        return 0.0
    
    # Use the last window_size elements
    cdef np.ndarray[float_t, ndim=1] window = price_changes[n-window_size:n]
    
    # Calculate absolute values and sum
    cdef np.ndarray[float_t, ndim=1] abs_window = np.abs(window)
    cdef float_t sum_abs = np.sum(abs_window)
    
    if sum_abs == 0:
        return 0.0
    
    # Normalize and calculate entropy
    cdef np.ndarray[float_t, ndim=1] prob = abs_window / sum_abs
    cdef np.ndarray[float_t, ndim=1] nonzero_prob = prob[prob > 0]
    
    if len(nonzero_prob) == 0:
        return 0.0
    
    cdef np.ndarray[float_t, ndim=1] prob_q = np.power(nonzero_prob, q)
    return (1.0 - np.sum(prob_q)) / (q - 1.0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_entropy_bars_cy(np.ndarray[float_t, ndim=1] prices, 
                              np.ndarray[float_t, ndim=1] volumes, 
                              np.ndarray timestamps,
                              int entropy_threshold, 
                              int window_size=100, 
                              str method='shannon', 
                              int avg_window=200):
    """
    Calculate entropy bars from raw price data - Cython optimized version.
    
    Args:
        prices: NumPy array of prices
        volumes: NumPy array of volumes
        timestamps: NumPy array of timestamps
        entropy_threshold: Entropy threshold as a multiple of average entropy
        window_size: Window size for entropy calculation
        method: Entropy calculation method ('shannon' or 'tsallis')
        avg_window: Window size for calculating average entropy
        
    Returns:
        DataFrame with entropy bars
    """
    cdef int n_points = len(prices)
    if n_points == 0 or entropy_threshold <= 0:
        return pd.DataFrame()
    
    # Calculate price changes
    cdef np.ndarray[float_t, ndim=1] price_changes = np.zeros(n_points, dtype=np.float64)
    cdef int i, j
    
    for i in range(1, n_points):
        price_changes[i] = prices[i] - prices[i-1]
    
    # Calculate entropy values
    cdef np.ndarray[float_t, ndim=1] entropy_values = np.zeros(n_points, dtype=np.float64)
    
    for i in range(1, n_points):
        if method == 'shannon':
            entropy_values[i] = calculate_shannon_entropy_cy(price_changes[:i+1], window_size)
        else:
            entropy_values[i] = calculate_tsallis_entropy_cy(price_changes[:i+1], window_size)
    
    # Initialize arrays for bar data
    cdef list timestamps_out = []
    cdef list start_times = []
    cdef list end_times = []
    cdef list opens = []
    cdef list highs = []
    cdef list lows = []
    cdef list closes = []
    cdef list volumes_out = []
    cdef list entropy_out = []
    
    # Initialize bar tracking variables
    cdef float_t bar_open = prices[0]
    cdef float_t bar_high = prices[0]
    cdef float_t bar_low = prices[0]
    cdef float_t bar_volume = 0.0
    cdef int bar_start_idx = 0
    
    # Calculate initial average entropy
    cdef float_t entropy_avg = 0.0
    cdef int avg_points = min(avg_window, n_points)
    cdef float_t threshold = 0.0
    
    for i in range(avg_points):
        entropy_avg += entropy_values[i]
    entropy_avg /= avg_points
    threshold = entropy_avg * entropy_threshold
    
    # Process each data point
    for i in range(1, n_points):
        # Update bar stats
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])
        bar_volume += volumes[i]
        
        # Update moving average of entropy periodically
        if i % avg_window == 0:
            entropy_avg = 0.0
            for j in range(max(0, i-avg_window), i):
                entropy_avg += entropy_values[j]
            entropy_avg /= min(avg_window, i)
            threshold = entropy_avg * entropy_threshold
        
        # Check if entropy threshold is reached
        if entropy_values[i] >= threshold and threshold > 0:
            # Add bar to results
            timestamps_out.append(timestamps[i])
            start_times.append(timestamps[bar_start_idx])
            end_times.append(timestamps[i])
            opens.append(bar_open)
            highs.append(bar_high)
            lows.append(bar_low)
            closes.append(prices[i])
            volumes_out.append(bar_volume)
            entropy_out.append(entropy_values[i])
            
            # Reset bar tracking for next bar
            if i < n_points - 1:
                bar_start_idx = i + 1
                bar_open = prices[bar_start_idx]
                bar_high = bar_open
                bar_low = bar_open
                bar_volume = 0.0
    
    # Create DataFrame from collected data
    if len(timestamps_out) > 0:
        result = pd.DataFrame({
            'timestamp': timestamps_out,
            'start_time': start_times,
            'end_time': end_times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes_out,
            'entropy': entropy_out,
            'bar_type': 'entropy',
            'ratio': float(entropy_threshold)
        })
        # Make timestamp the index and return
        result.set_index('timestamp', inplace=True)
        return result
    
    return pd.DataFrame()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calculate_information_bars_cy(np.ndarray[float_t, ndim=1] prices, 
                                 np.ndarray[float_t, ndim=1] volumes, 
                                 np.ndarray timestamps,
                                 int info_threshold):
    """
    Calculate information bars from raw price data - Cython optimized version.
    
    Args:
        prices: NumPy array of prices
        volumes: NumPy array of volumes
        timestamps: NumPy array of timestamps
        info_threshold: Information content threshold to create a new bar
        
    Returns:
        DataFrame with information bars
    """
    cdef int n_points = len(prices)
    if n_points == 0 or info_threshold <= 0:
        return pd.DataFrame()
    
    # Calculate log returns
    cdef np.ndarray[float_t, ndim=1] log_returns = np.zeros(n_points, dtype=np.float64)
    cdef int i
    
    for i in range(1, n_points):
        if prices[i-1] > 0:
            log_returns[i] = log(prices[i] / prices[i-1])
    
    # Initialize arrays for bar data
    cdef list timestamps_out = []
    cdef list start_times = []
    cdef list end_times = []
    cdef list opens = []
    cdef list highs = []
    cdef list lows = []
    cdef list closes = []
    cdef list volumes_out = []
    cdef list info_content_out = []
    
    # Initialize bar tracking variables
    cdef float_t cum_abs_info = 0.0
    cdef float_t bar_open = prices[0]
    cdef float_t bar_high = prices[0]
    cdef float_t bar_low = prices[0]
    cdef float_t bar_volume = 0.0
    cdef float_t info_content
    cdef int bar_start_idx = 0
    
    # Process each data point
    for i in range(1, n_points):
        # Calculate information content
        info_content = fabs(log_returns[i])
        
        # Update bar stats
        bar_high = max(bar_high, prices[i])
        bar_low = min(bar_low, prices[i])
        bar_volume += volumes[i]
        cum_abs_info += info_content
        
        # Check if information threshold is reached
        if cum_abs_info >= info_threshold:
            # Add bar to results
            timestamps_out.append(timestamps[i])
            start_times.append(timestamps[bar_start_idx])
            end_times.append(timestamps[i])
            opens.append(bar_open)
            highs.append(bar_high)
            lows.append(bar_low)
            closes.append(prices[i])
            volumes_out.append(bar_volume)
            info_content_out.append(cum_abs_info)
            
            # Reset bar tracking for next bar
            if i < n_points - 1:
                bar_start_idx = i + 1
                bar_open = prices[bar_start_idx]
                bar_high = bar_open
                bar_low = bar_open
                bar_volume = 0.0
                cum_abs_info = 0.0
    
    # Create DataFrame from collected data
    if len(timestamps_out) > 0:
        result = pd.DataFrame({
            'timestamp': timestamps_out,
            'start_time': start_times,
            'end_time': end_times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes_out,
            'info_content': info_content_out,
            'bar_type': 'information',
            'ratio': float(info_threshold)
        })
        # Make timestamp the index and return
        result.set_index('timestamp', inplace=True)
        return result
    
    return pd.DataFrame()

# Python wrapper functions that take DataFrames as input and extract arrays
def calculate_volume_bars(df, ratio, avg_window=200):
    """DataFrame interface for Cython-optimized volume bars calculation"""
    if df.empty:
        return pd.DataFrame()
    
    prices = df['price'].values
    volumes = df['volume'].values
    timestamps = df.index.values
    
    return calculate_volume_bars_cy(prices, volumes, timestamps, ratio, avg_window)

def calculate_tick_bars(df, tick_size):
    """DataFrame interface for Cython-optimized tick bars calculation"""
    if df.empty:
        return pd.DataFrame()
    
    prices = df['price'].values
    volumes = df['volume'].values
    timestamps = df.index.values
    
    return calculate_tick_bars_cy(prices, volumes, timestamps, tick_size)

def calculate_price_bars(df, price_threshold):
    """DataFrame interface for Cython-optimized price bars calculation"""
    if df.empty:
        return pd.DataFrame()
    
    prices = df['price'].values
    volumes = df['volume'].values
    timestamps = df.index.values
    
    return calculate_price_bars_cy(prices, volumes, timestamps, price_threshold)

def calculate_dollar_bars(df, dollar_threshold):
    """DataFrame interface for Cython-optimized dollar bars calculation"""
    if df.empty:
        return pd.DataFrame()
    
    prices = df['price'].values
    volumes = df['volume'].values
    timestamps = df.index.values
    
    return calculate_dollar_bars_cy(prices, volumes, timestamps, dollar_threshold)

def calculate_entropy_bars(df, entropy_threshold, window_size=100, method='shannon', avg_window=200):
    """DataFrame interface for Cython-optimized entropy bars calculation"""
    if df.empty:
        return pd.DataFrame()
    
    prices = df['price'].values
    volumes = df['volume'].values
    timestamps = df.index.values
    
    return calculate_entropy_bars_cy(prices, volumes, timestamps, entropy_threshold, window_size, method, avg_window)

def calculate_information_bars(df, info_threshold):
    """DataFrame interface for Cython-optimized information bars calculation"""
    if df.empty:
        return pd.DataFrame()
    
    prices = df['price'].values
    volumes = df['volume'].values
    timestamps = df.index.values
    
    return calculate_information_bars_cy(prices, volumes, timestamps, info_threshold)

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_time_bars(df, time_interval_minutes):
    """
    Calculate time bars (regular OHLCV bars) from raw price data.
    
    This function uses pandas.resample() which cannot be directly implemented in Cython,
    so it's a wrapper around the pure Python implementation.
    
    Args:
        df: DataFrame with raw price data (must have 'timestamp', 'price', 'volume' columns)
        time_interval_minutes: Time interval in minutes
        
    Returns:
        DataFrame with time bars
    """
    if df.empty or time_interval_minutes <= 0:
        return pd.DataFrame()
    
    try:
        # Make sure timestamp is the index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
        
        # Use pandas resample to create time bars
        resampled = df.resample(f'{time_interval_minutes}T')
        
        # Create OHLCV aggregate
        bars_df = pd.DataFrame({
            'open': resampled['price'].first(),
            'high': resampled['price'].max(),
            'low': resampled['price'].min(),
            'close': resampled['price'].last(),
            'volume': resampled['volume'].sum()
        })
        
        # Drop rows with NaN values (empty bars)
        bars_df = bars_df.dropna()
        
        # Add bar metadata
        bars_df['bar_type'] = 'time'
        bars_df['ratio'] = float(time_interval_minutes)
        
        # Add start_time and end_time
        bars_df['start_time'] = bars_df.index
        bars_df['end_time'] = bars_df.index + pd.Timedelta(minutes=time_interval_minutes)
        
        return bars_df
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error calculating time bars: {str(e)}")
        return pd.DataFrame() 