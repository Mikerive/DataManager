import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for bar processing by ensuring it has the correct format.
    
    Args:
        df: Raw data DataFrame
        
    Returns:
        Properly formatted DataFrame
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Rename 'close' to 'price' if needed
    if 'close' in df.columns and 'price' not in df.columns:
        df = df.rename(columns={'close': 'price'})
    
    # Ensure timestamp is the index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Ensure required columns exist
    required_columns = ['price', 'volume']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"DataFrame missing required column: {col}")
            return pd.DataFrame()
    
    return df

def validate_bar_parameters(bar_type: str, ratio: float, **kwargs) -> Tuple[bool, str]:
    """
    Validate parameters for bar calculation.
    
    Args:
        bar_type: Type of bar to calculate
        ratio: The ratio/threshold parameter (should be an integer)
        **kwargs: Additional parameters specific to certain bar types
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Ensure ratio is a positive integer or can be converted to one
    try:
        ratio_int = int(ratio)
        if ratio_int <= 0:
            return False, f"Invalid ratio: {ratio}. Must be greater than 0."
        if ratio_int != ratio:
            return False, f"Ratio must be an integer, got {ratio}"
    except (ValueError, TypeError):
        return False, f"Ratio must be convertible to an integer, got {ratio} of type {type(ratio)}"
    
    if bar_type not in ['volume', 'tick', 'price', 'dollar', 'time', 'entropy', 'information']:
        return False, f"Invalid bar type: {bar_type}"
    
    # Specific validations for certain bar types
    if bar_type == 'tick':
        if not isinstance(ratio_int, int) or ratio_int <= 0:
            return False, f"Tick size must be a positive integer, got {ratio}"
    
    elif bar_type == 'time':
        if not isinstance(ratio_int, int) or ratio_int <= 0:
            return False, f"Time interval must be a positive integer (minutes), got {ratio}"
    
    elif bar_type == 'entropy':
        window_size = kwargs.get('window_size', 100)
        if not isinstance(window_size, int) or window_size <= 0:
            return False, f"Window size must be a positive integer, got {window_size}"
        
        method = kwargs.get('method', 'shannon')
        if method not in ['shannon', 'tsallis']:
            return False, f"Entropy method must be 'shannon' or 'tsallis', got {method}"
    
    return True, ""

def enrich_bar_dataframe(bars_df: pd.DataFrame, bar_type: str, ratio: float) -> pd.DataFrame:
    """
    Add additional metadata to the bar DataFrame.
    
    Args:
        bars_df: DataFrame with calculated bars
        bar_type: Type of bar that was calculated
        ratio: The ratio/threshold parameter used
        
    Returns:
        Enriched DataFrame with additional metadata
    """
    if bars_df.empty:
        return bars_df
    
    # Add bar type and ratio if not already present
    if 'bar_type' not in bars_df.columns:
        bars_df['bar_type'] = bar_type
    
    if 'ratio' not in bars_df.columns:
        bars_df['ratio'] = float(ratio)
    
    # Calculate additional metrics that might be useful
    if all(col in bars_df.columns for col in ['open', 'close']):
        # Calculate returns
        bars_df['returns'] = (bars_df['close'] / bars_df['open']) - 1
        
        # Calculate log returns
        bars_df['log_returns'] = np.log(bars_df['close'] / bars_df['open'])
    
    if 'start_time' in bars_df.columns and 'end_time' in bars_df.columns:
        # Calculate duration in seconds if not already present
        if 'duration' not in bars_df.columns:
            bars_df['duration'] = (bars_df['end_time'] - bars_df['start_time']).dt.total_seconds()
    
    return bars_df

def merge_bar_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple bar DataFrames into a single DataFrame.
    
    Args:
        dfs: List of DataFrames with bar data
        
    Returns:
        Merged DataFrame
    """
    # Filter out empty DataFrames
    valid_dfs = [df for df in dfs if not df.empty]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    # Concatenate DataFrames
    result = pd.concat(valid_dfs)
    
    # Sort by timestamp
    if isinstance(result.index, pd.DatetimeIndex):
        result = result.sort_index()
    elif 'timestamp' in result.columns:
        result = result.sort_values('timestamp')
    
    return result 