"""
Bar Processing Service

This module provides a service for processing different types of bars from raw market data.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any

# Add the current directory to the path so the extension can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class BarProcessingService:
    """
    Service for processing different types of bars from raw market data.
    This service provides methods for calculating various types of bars,
    including volume bars, tick bars, time bars, and entropy bars.
    """
    
    def __init__(self):
        """
        Initialize the BarProcessingService.
        Tries to use the C++ implementation if available, falls back to Python otherwise.
        """
        self.use_cpp = False
        self.calculator = None
        self.data = None
        
        try:
            # Try to import and use the C++ implementation
            try:
                # First try to import from the cpp_ext directory
                from cpp_ext import bar_calculator_cpp
                from bar_calculator_wrapper import BarCalculator
            except ImportError:
                # Fall back to using the fully qualified module path
                from backend.services.BarProcessingService.cpp_ext import bar_calculator_cpp
                from backend.services.BarProcessingService.bar_calculator_wrapper import BarCalculator
                
            self.calculator = BarCalculator()
            self.use_cpp = True
            print("Using C++ implementation for bar processing")
        except ImportError as e:
            # Fall back to Python implementation
            print(f"C++ implementation not available: {str(e)}, using Python implementation")
            self.use_cpp = False
    
    def load_data(self, df: pd.DataFrame) -> None:
        """
        Load price data for bar processing.
        
        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"DataFrame missing required columns: {missing}")
        
        # Store the data
        self.data = df.copy()
        
        # Ensure timestamp is datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Set the data in the C++ calculator if available
        if self.use_cpp:
            self.calculator.set_data(self.data)
    
    def calculate_volume_bars(self, volume_threshold: float) -> pd.DataFrame:
        """
        Calculate volume bars from the loaded data.
        
        Args:
            volume_threshold: Volume threshold for bar formation
            
        Returns:
            DataFrame with volume bar data
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.use_cpp:
            return self.calculator.calculate_volume_bars(volume_threshold)
        else:
            # Python fallback implementation
            return self._calculate_volume_bars_python(volume_threshold)
    
    def calculate_tick_bars(self, tick_count: int) -> pd.DataFrame:
        """
        Calculate tick bars from the loaded data.
        
        Args:
            tick_count: Number of ticks per bar
            
        Returns:
            DataFrame with tick bar data
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.use_cpp:
            return self.calculator.calculate_tick_bars(tick_count)
        else:
            # Python fallback implementation
            return self._calculate_tick_bars_python(tick_count)
    
    def calculate_time_bars(self, seconds: float) -> pd.DataFrame:
        """
        Calculate time bars from the loaded data.
        
        Args:
            seconds: Time interval in seconds
            
        Returns:
            DataFrame with time bar data
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.use_cpp:
            return self.calculator.calculate_time_bars(seconds)
        else:
            # Python fallback implementation
            return self._calculate_time_bars_python(seconds)
    
    def calculate_entropy_bars(self, entropy_threshold: float, 
                              window_size: int = 100, 
                              method: str = "shannon",
                              q_param: float = 1.5) -> pd.DataFrame:
        """
        Calculate entropy bars from the loaded data.
        
        Args:
            entropy_threshold: Entropy threshold for bar formation
            window_size: Size of rolling window for entropy calculation
            method: Entropy calculation method ('shannon' or 'tsallis')
            q_param: Tsallis q-parameter (used only with 'tsallis' method)
            
        Returns:
            DataFrame with entropy bar data
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.use_cpp:
            return self.calculator.calculate_entropy_bars(
                entropy_threshold, window_size, method, q_param
            )
        else:
            # Python fallback implementation
            return self._calculate_entropy_bars_python(
                entropy_threshold, window_size, method, q_param
            )
    
    def batch_calculate(self, params_list: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate multiple bar types in a single pass.
        
        Args:
            params_list: List of parameter dictionaries:
                         [{'bar_type': str, 'ratio': float, 'window_size': int, 'method': str, 'q_param': float}, ...]
            
        Returns:
            Dictionary mapping bar type to DataFrame with bar data
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.use_cpp:
            return self.calculator.batch_calculate(params_list)
        else:
            # Python fallback implementation
            results = {}
            for params in params_list:
                bar_type = params.get('bar_type')
                ratio = params.get('ratio')
                
                if not bar_type or ratio is None:
                    raise ValueError("Each parameter set must have 'bar_type' and 'ratio'")
                
                key = f"{bar_type}_{ratio}"
                
                if bar_type == 'volume':
                    results[key] = self._calculate_volume_bars_python(ratio)
                elif bar_type == 'tick':
                    results[key] = self._calculate_tick_bars_python(int(ratio))
                elif bar_type == 'time':
                    results[key] = self._calculate_time_bars_python(ratio)
                elif bar_type == 'entropy':
                    window_size = params.get('window_size', 100)
                    method = params.get('method', 'shannon')
                    q_param = params.get('q_param', 1.5)
                    results[key] = self._calculate_entropy_bars_python(
                        ratio, window_size, method, q_param
                    )
                else:
                    raise ValueError(f"Unknown bar type: {bar_type}")
            
            return results
    
    # Python fallback implementations
    
    def _calculate_volume_bars_python(self, volume_threshold: float) -> pd.DataFrame:
        """Python implementation of volume bars calculation."""
        bars = []
        cumulative_volume = 0
        bar_start_idx = 0
        df = self.data
        
        for i in range(len(df)):
            cumulative_volume += df.iloc[i]['volume']
            
            if cumulative_volume >= volume_threshold or i == len(df) - 1:
                # Create a bar
                bar = {
                    'timestamp': df.iloc[i]['timestamp'],
                    'start_time': df.iloc[bar_start_idx]['timestamp'],
                    'end_time': df.iloc[i]['timestamp'],
                    'open': df.iloc[bar_start_idx]['open'],
                    'high': df.iloc[bar_start_idx:i+1]['high'].max(),
                    'low': df.iloc[bar_start_idx:i+1]['low'].min(),
                    'close': df.iloc[i]['close'],
                    'volume': cumulative_volume
                }
                bars.append(bar)
                
                # Reset for the next bar
                cumulative_volume = 0
                bar_start_idx = i + 1
        
        return pd.DataFrame(bars)
    
    def _calculate_tick_bars_python(self, tick_count: int) -> pd.DataFrame:
        """Python implementation of tick bars calculation."""
        bars = []
        ticks = 0
        cumulative_volume = 0
        bar_start_idx = 0
        df = self.data
        
        for i in range(len(df)):
            ticks += 1
            cumulative_volume += df.iloc[i]['volume']
            
            if ticks >= tick_count or i == len(df) - 1:
                # Create a bar
                bar = {
                    'timestamp': df.iloc[i]['timestamp'],
                    'start_time': df.iloc[bar_start_idx]['timestamp'],
                    'end_time': df.iloc[i]['timestamp'],
                    'open': df.iloc[bar_start_idx]['open'],
                    'high': df.iloc[bar_start_idx:i+1]['high'].max(),
                    'low': df.iloc[bar_start_idx:i+1]['low'].min(),
                    'close': df.iloc[i]['close'],
                    'volume': cumulative_volume
                }
                bars.append(bar)
                
                # Reset for the next bar
                ticks = 0
                cumulative_volume = 0
                bar_start_idx = i + 1
        
        return pd.DataFrame(bars)
    
    def _calculate_time_bars_python(self, seconds: float) -> pd.DataFrame:
        """Python implementation of time bars calculation."""
        df = self.data
        # Calculate the time delta in nanoseconds
        time_delta = pd.Timedelta(seconds=seconds)
        
        # Group by time interval
        df['time_group'] = df['timestamp'].dt.floor(time_delta)
        
        # Aggregate the groups
        bars = df.groupby('time_group').agg({
            'timestamp': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # Add start_time and end_time
        bars = bars.rename(columns={'time_group': 'start_time'})
        bars['end_time'] = bars['timestamp']
        
        # Reorder columns
        return bars[['timestamp', 'start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume']]
    
    def _calculate_entropy_bars_python(self, entropy_threshold: float, 
                                      window_size: int = 100, 
                                      method: str = "shannon",
                                      q_param: float = 1.5) -> pd.DataFrame:
        """Python implementation of entropy bars calculation."""
        # Calculate price changes
        df = self.data
        price_changes = df['close'].diff().fillna(0).values
        
        # Calculate rolling entropy
        entropy_values = np.zeros(len(df))
        
        for i in range(window_size, len(df)):
            window = np.abs(price_changes[i-window_size+1:i+1])
            sum_abs = np.sum(window)
            
            if sum_abs == 0:
                entropy_values[i] = 0
                continue
            
            probs = window / sum_abs
            probs = probs[probs > 0]  # Remove zeros
            
            if method == 'shannon':
                entropy_values[i] = -np.sum(probs * np.log2(probs))
            elif method == 'tsallis':
                entropy_values[i] = (1 - np.sum(probs ** q_param)) / (q_param - 1)
            else:
                raise ValueError(f"Unknown entropy method: {method}")
        
        # Generate bars based on entropy threshold
        bars = []
        bar_start_idx = window_size  # Start after we have enough data for entropy
        cumulative_volume = 0
        
        for i in range(window_size, len(df)):
            cumulative_volume += df.iloc[i]['volume']
            
            if entropy_values[i] >= entropy_threshold or i == len(df) - 1:
                # Create a bar
                bar = {
                    'timestamp': df.iloc[i]['timestamp'],
                    'start_time': df.iloc[bar_start_idx]['timestamp'],
                    'end_time': df.iloc[i]['timestamp'],
                    'open': df.iloc[bar_start_idx]['open'],
                    'high': df.iloc[bar_start_idx:i+1]['high'].max(),
                    'low': df.iloc[bar_start_idx:i+1]['low'].min(),
                    'close': df.iloc[i]['close'],
                    'volume': cumulative_volume
                }
                bars.append(bar)
                
                # Reset for the next bar
                cumulative_volume = 0
                bar_start_idx = i + 1
        
        return pd.DataFrame(bars) 