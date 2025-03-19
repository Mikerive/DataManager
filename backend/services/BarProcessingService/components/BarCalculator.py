import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import logging


class BarCalculator:
    """
    Utility class for specialized financial bar calculations.
    
    This class provides focused methods for calculating different types of bars:
    - Volume bars
    - Tick bars
    - Price bars
    - Dollar bars
    - Time bars
    - Entropy bars (Shannon and Tsallis)
    - Information content bars
    
    All methods are stateless and work with DataFrame/NumPy inputs.
    """
    
    def __init__(self):
        """Initialize the BarCalculator."""
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
    def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw data for bar calculation by ensuring correct format.
        
        Args:
            df: DataFrame with raw price data
                
        Returns:
            Properly formatted DataFrame
        """
        if df.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have a proper datetime index and sorted data
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df.index = pd.to_datetime(df.index)
        
        # Sort by timestamp to ensure proper sequential processing
        df = df.sort_index()
        
        # Rename 'close' to 'price' if needed
        if 'close' in df.columns and 'price' not in df.columns:
            df = df.rename(columns={'close': 'price'})
        
        return df
    
    @staticmethod
    def calculate_price_changes(prices: np.ndarray) -> np.ndarray:
        """
        Calculate price changes from an array of prices.
        
        Args:
            prices: Array of prices
            
        Returns:
            Array of price changes
        """
        if len(prices) <= 1:
            return np.zeros(len(prices))
            
        changes = np.zeros(len(prices))
        changes[1:] = np.diff(prices)
        return changes