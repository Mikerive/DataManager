import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

def calculate_volume_bars(df: pd.DataFrame, ratio: float, avg_window: int = 200) -> pd.DataFrame:
    """
    Calculate volume bars from raw price data.
    
    Args:
        df: DataFrame with raw price data (must have 'timestamp', 'price', 'volume' columns)
        ratio: Volume threshold as a multiple of average volume
        avg_window: Window size for calculating average volume
        
    Returns:
        DataFrame with volume bars
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        # Extract data
        df = df.copy()
        prices = df['price'].values
        volumes = df['volume'].values
        timestamps = df.index.values
        n_points = len(prices)
        
        # Initialize bar tracking
        bars = []
        cum_volume = 0
        bar_open = prices[0]
        bar_high = prices[0]
        bar_low = prices[0]
        bar_volume = 0
        bar_start_idx = 0
        
        # Calculate moving average of volume
        vol_window = min(avg_window, n_points)
        vol_mavg = np.mean(volumes[:vol_window])
        threshold = vol_mavg * ratio
        
        # Process each data point
        for i in range(n_points):
            price = prices[i]
            volume = volumes[i]
            timestamp = timestamps[i]
            
            # Update bar stats
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_volume += volume
            cum_volume += volume
            
            # Update moving average periodically
            if i % vol_window == 0 and i > 0:
                vol_mavg = np.mean(volumes[max(0, i-vol_window):i])
                threshold = vol_mavg * ratio
            
            # Check if volume threshold is reached
            if cum_volume >= threshold and threshold > 0:
                # Create a bar
                bars.append({
                    'timestamp': timestamps[i],
                    'start_time': timestamps[bar_start_idx],
                    'end_time': timestamps[i],
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': price,
                    'volume': bar_volume,
                    'bar_type': 'volume',
                    'ratio': ratio
                })
                
                # Reset bar tracking
                if i < n_points - 1:
                    bar_start_idx = i + 1
                    bar_open = prices[bar_start_idx]
                    bar_high = bar_open
                    bar_low = bar_open
                    bar_volume = 0
                    cum_volume = 0
        
        # Convert to DataFrame
        if bars:
            result_df = pd.DataFrame(bars)
            # Make timestamp the index
            result_df.set_index('timestamp', inplace=True)
            return result_df
        
        return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error calculating volume bars: {str(e)}")
        return pd.DataFrame()

def calculate_tick_bars(df: pd.DataFrame, tick_size: int) -> pd.DataFrame:
    """
    Calculate tick bars from raw price data.
    
    Args:
        df: DataFrame with raw price data (must have 'timestamp', 'price', 'volume' columns)
        tick_size: Number of ticks to aggregate into one bar
        
    Returns:
        DataFrame with tick bars
    """
    if df.empty or tick_size <= 0:
        return pd.DataFrame()
    
    try:
        # Extract data
        df = df.copy()
        prices = df['price'].values
        volumes = df['volume'].values
        timestamps = df.index.values
        n_points = len(prices)
        
        # Initialize bar tracking
        bars = []
        tick_count = 0
        bar_open = prices[0]
        bar_high = prices[0]
        bar_low = prices[0]
        bar_volume = 0
        bar_start_idx = 0
        
        # Process each data point
        for i in range(n_points):
            price = prices[i]
            volume = volumes[i]
            
            # Update bar stats
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_volume += volume
            tick_count += 1
            
            # Check if tick threshold is reached
            if tick_count >= tick_size:
                # Create a bar
                bars.append({
                    'timestamp': timestamps[i],
                    'start_time': timestamps[bar_start_idx],
                    'end_time': timestamps[i],
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': price,
                    'volume': bar_volume,
                    'bar_type': 'tick',
                    'ratio': float(tick_size)
                })
                
                # Reset bar tracking
                if i < n_points - 1:
                    bar_start_idx = i + 1
                    bar_open = prices[bar_start_idx]
                    bar_high = bar_open
                    bar_low = bar_open
                    bar_volume = 0
                    tick_count = 0
        
        # Convert to DataFrame
        if bars:
            result_df = pd.DataFrame(bars)
            # Make timestamp the index
            result_df.set_index('timestamp', inplace=True)
            return result_df
        
        return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error calculating tick bars: {str(e)}")
        return pd.DataFrame()

def calculate_time_bars(df: pd.DataFrame, time_interval_minutes: int) -> pd.DataFrame:
    """
    Calculate time bars (regular OHLCV bars) from raw price data.
    
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
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
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
        logger.error(f"Error calculating time bars: {str(e)}")
        return pd.DataFrame()

def calculate_price_bars(df: pd.DataFrame, price_threshold: float) -> pd.DataFrame:
    """
    Calculate price bars from raw price data.
    
    Args:
        df: DataFrame with raw price data (must have 'timestamp', 'price', 'volume' columns)
        price_threshold: Minimum absolute price change to create a new bar
        
    Returns:
        DataFrame with price bars
    """
    if df.empty or price_threshold <= 0:
        return pd.DataFrame()
    
    try:
        # Extract data
        df = df.copy()
        prices = df['price'].values
        volumes = df['volume'].values
        timestamps = df.index.values
        n_points = len(prices)
        
        # Initialize bar tracking
        bars = []
        bar_open = prices[0]
        bar_high = prices[0]
        bar_low = prices[0]
        bar_volume = 0
        bar_start_idx = 0
        last_bar_price = prices[0]
        
        # Process each data point
        for i in range(n_points):
            price = prices[i]
            volume = volumes[i]
            
            # Update bar stats
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_volume += volume
            
            # Check if price threshold is reached
            if abs(price - last_bar_price) >= price_threshold:
                # Create a bar
                bars.append({
                    'timestamp': timestamps[i],
                    'start_time': timestamps[bar_start_idx],
                    'end_time': timestamps[i],
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': price,
                    'volume': bar_volume,
                    'bar_type': 'price',
                    'ratio': float(price_threshold)
                })
                
                # Reset bar tracking
                if i < n_points - 1:
                    bar_start_idx = i + 1
                    bar_open = prices[bar_start_idx]
                    bar_high = bar_open
                    bar_low = bar_open
                    bar_volume = 0
                    last_bar_price = price
        
        # Convert to DataFrame
        if bars:
            result_df = pd.DataFrame(bars)
            # Make timestamp the index
            result_df.set_index('timestamp', inplace=True)
            return result_df
        
        return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error calculating price bars: {str(e)}")
        return pd.DataFrame()

def calculate_dollar_bars(df: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    """
    Calculate dollar bars from raw price data.
    
    Args:
        df: DataFrame with raw price data (must have 'timestamp', 'price', 'volume' columns)
        dollar_threshold: Dollar volume threshold to create a new bar
        
    Returns:
        DataFrame with dollar bars
    """
    if df.empty or dollar_threshold <= 0:
        return pd.DataFrame()
    
    try:
        # Extract data
        df = df.copy()
        prices = df['price'].values
        volumes = df['volume'].values
        timestamps = df.index.values
        n_points = len(prices)
        
        # Initialize bar tracking
        bars = []
        cum_dollar_volume = 0
        bar_open = prices[0]
        bar_high = prices[0]
        bar_low = prices[0]
        bar_volume = 0
        bar_start_idx = 0
        
        # Process each data point
        for i in range(n_points):
            price = prices[i]
            volume = volumes[i]
            dollar_volume = price * volume
            
            # Update bar stats
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_volume += volume
            cum_dollar_volume += dollar_volume
            
            # Check if dollar threshold is reached
            if cum_dollar_volume >= dollar_threshold:
                # Create a bar
                bars.append({
                    'timestamp': timestamps[i],
                    'start_time': timestamps[bar_start_idx],
                    'end_time': timestamps[i],
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': price,
                    'volume': bar_volume,
                    'bar_type': 'dollar',
                    'ratio': float(dollar_threshold)
                })
                
                # Reset bar tracking
                if i < n_points - 1:
                    bar_start_idx = i + 1
                    bar_open = prices[bar_start_idx]
                    bar_high = bar_open
                    bar_low = bar_open
                    bar_volume = 0
                    cum_dollar_volume = 0
        
        # Convert to DataFrame
        if bars:
            result_df = pd.DataFrame(bars)
            # Make timestamp the index
            result_df.set_index('timestamp', inplace=True)
            return result_df
        
        return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error calculating dollar bars: {str(e)}")
        return pd.DataFrame()

def calculate_shannon_entropy(price_changes: np.ndarray, window_size: int) -> float:
    """
    Calculate Shannon entropy for a window of price changes.
    
    Args:
        price_changes: Array of price changes
        window_size: Window size for entropy calculation
        
    Returns:
        float: Shannon entropy value
    """
    # Use a rolling window
    if len(price_changes) < window_size:
        return 0
    
    window = price_changes[-window_size:]
    # Check for all zeros to avoid division by zero
    sum_abs_window = np.sum(np.abs(window))
    if sum_abs_window == 0:
        return 0
        
    # Normalize price changes
    prob = np.abs(window) / sum_abs_window
    # Remove zeros to avoid log(0)
    prob = prob[prob > 0]
    if len(prob) == 0:
        return 0
        
    return -np.sum(prob * np.log2(prob))

def calculate_tsallis_entropy(price_changes: np.ndarray, window_size: int, q: float = 1.5) -> float:
    """
    Calculate Tsallis entropy for a window of price changes.
    
    Args:
        price_changes: Array of price changes
        window_size: Window size for entropy calculation
        q: Tsallis q-parameter (default: 1.5)
        
    Returns:
        float: Tsallis entropy value
    """
    if len(price_changes) < window_size:
        return 0
    
    window = price_changes[-window_size:]
    # Check for all zeros to avoid division by zero
    sum_abs_window = np.sum(np.abs(window))
    if sum_abs_window == 0:
        return 0
        
    # Normalize price changes
    prob = np.abs(window) / sum_abs_window
    # Remove zeros
    prob = prob[prob > 0]
    if len(prob) == 0:
        return 0
        
    return (1 - np.sum(prob**q)) / (q - 1)

def calculate_entropy_bars(df: pd.DataFrame, entropy_threshold: float, window_size: int = 100, 
                          method: str = 'shannon', avg_window: int = 200) -> pd.DataFrame:
    """
    Calculate entropy bars from raw price data.
    
    Args:
        df: DataFrame with raw price data (must have 'timestamp', 'price', 'volume' columns)
        entropy_threshold: Entropy threshold as a multiple of average entropy
        window_size: Window size for entropy calculation
        method: Entropy calculation method ('shannon' or 'tsallis')
        avg_window: Window size for calculating average entropy
        
    Returns:
        DataFrame with entropy bars
    """
    if df.empty or entropy_threshold <= 0:
        return pd.DataFrame()
    
    try:
        # Extract data
        df = df.copy()
        prices = df['price'].values
        volumes = df['volume'].values
        timestamps = df.index.values
        n_points = len(prices)
        
        # Calculate price changes
        price_changes = np.zeros(n_points)
        price_changes[1:] = np.diff(prices)
        
        # Initialize entropy values
        entropy_values = np.zeros(n_points)
        for i in range(1, n_points):
            if method == 'shannon':
                entropy_values[i] = calculate_shannon_entropy(price_changes[:i+1], window_size)
            else:
                entropy_values[i] = calculate_tsallis_entropy(price_changes[:i+1], window_size)
        
        # Initialize bar tracking
        bars = []
        bar_open = prices[0]
        bar_high = prices[0]
        bar_low = prices[0]
        bar_volume = 0
        bar_start_idx = 0
        
        # Calculate initial average entropy
        entropy_avg = np.mean(entropy_values[:min(avg_window, n_points)])
        threshold = entropy_avg * entropy_threshold
        
        # Process each data point
        for i in range(1, n_points):
            price = prices[i]
            volume = volumes[i]
            
            # Update bar stats
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_volume += volume
            
            # Update moving average of entropy periodically
            if i % avg_window == 0:
                entropy_avg = np.mean(entropy_values[max(0, i-avg_window):i])
                threshold = entropy_avg * entropy_threshold
            
            # Check if entropy threshold is reached
            if entropy_values[i] >= threshold and threshold > 0:
                # Create a bar
                bars.append({
                    'timestamp': timestamps[i],
                    'start_time': timestamps[bar_start_idx],
                    'end_time': timestamps[i],
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': price,
                    'volume': bar_volume,
                    'bar_type': 'entropy',
                    'ratio': float(entropy_threshold),
                    'entropy': float(entropy_values[i])
                })
                
                # Reset bar tracking
                if i < n_points - 1:
                    bar_start_idx = i + 1
                    bar_open = prices[bar_start_idx]
                    bar_high = bar_open
                    bar_low = bar_open
                    bar_volume = 0
        
        # Convert to DataFrame
        if bars:
            result_df = pd.DataFrame(bars)
            # Make timestamp the index
            result_df.set_index('timestamp', inplace=True)
            return result_df
        
        return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error calculating entropy bars: {str(e)}")
        return pd.DataFrame()

def calculate_information_bars(df: pd.DataFrame, info_threshold: float) -> pd.DataFrame:
    """
    Calculate information bars from raw price data based on absolute change in information content.
    
    Args:
        df: DataFrame with raw price data (must have 'timestamp', 'price', 'volume' columns)
        info_threshold: Information content threshold to create a new bar
        
    Returns:
        DataFrame with information bars
    """
    if df.empty or info_threshold <= 0:
        return pd.DataFrame()
    
    try:
        # Extract data
        df = df.copy()
        prices = df['price'].values
        volumes = df['volume'].values
        timestamps = df.index.values
        n_points = len(prices)
        
        # Calculate log returns as a proxy for information content
        log_returns = np.zeros(n_points)
        log_returns[1:] = np.log(prices[1:] / prices[:-1])
        
        # Initialize bar tracking
        bars = []
        cum_abs_info = 0
        bar_open = prices[0]
        bar_high = prices[0]
        bar_low = prices[0]
        bar_volume = 0
        bar_start_idx = 0
        
        # Process each data point
        for i in range(1, n_points):
            price = prices[i]
            volume = volumes[i]
            info_content = abs(log_returns[i])
            
            # Update bar stats
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_volume += volume
            cum_abs_info += info_content
            
            # Check if information threshold is reached
            if cum_abs_info >= info_threshold:
                # Create a bar
                bars.append({
                    'timestamp': timestamps[i],
                    'start_time': timestamps[bar_start_idx],
                    'end_time': timestamps[i],
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': price,
                    'volume': bar_volume,
                    'bar_type': 'information',
                    'ratio': float(info_threshold),
                    'info_content': float(cum_abs_info)
                })
                
                # Reset bar tracking
                if i < n_points - 1:
                    bar_start_idx = i + 1
                    bar_open = prices[bar_start_idx]
                    bar_high = bar_open
                    bar_low = bar_open
                    bar_volume = 0
                    cum_abs_info = 0
        
        # Convert to DataFrame
        if bars:
            result_df = pd.DataFrame(bars)
            # Make timestamp the index
            result_df.set_index('timestamp', inplace=True)
            return result_df
        
        return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error calculating information bars: {str(e)}")
        return pd.DataFrame() 