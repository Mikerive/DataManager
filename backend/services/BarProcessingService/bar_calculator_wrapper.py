"""
Python wrapper for the C++ BarCalculator implementation.
This provides an easy-to-use interface to the high-performance C++ code.
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Union, Optional, Any

# Add the current directory to the path so the extension can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class BarCalculator:
    """
    Python wrapper for the C++ BarCalculator implementation.
    Provides methods for calculating different types of bars from market data.
    """
    
    def __init__(self):
        """
        Initialize the BarCalculator.
        """
        try:
            # Try multiple import paths to find the extension
            try:
                # Try to import directly from the package
                import _bar_processor as bar_calculator_cpp
            except ImportError:
                try:
                    # Try the old location
                    from cpp_ext import bar_calculator_cpp
                except ImportError:
                    # Fall back to absolute path
                    from backend.services.BarProcessingService import _bar_processor as bar_calculator_cpp
                
            self._cpp_calculator = bar_calculator_cpp.BarCalculator()
            self._cpp_module = bar_calculator_cpp
            self._data_loaded = False
            print("Successfully loaded C++ extension for bar calculation")
        except ImportError as e:
            print(f"C++ implementation not available: {e}, using Python implementation")
            # Use pure Python implementation as fallback
            self._use_python_fallback()
    
    def _use_python_fallback(self):
        """
        Use a pure Python implementation when the C++ extension is not available.
        This is a simplified version that doesn't support all features.
        """
        # Create a dummy C++ module with the required interfaces
        from types import SimpleNamespace
        from collections import defaultdict
        
        # Create a simple Python implementation for bar calculation
        class PyBarCalculator:
            def __init__(self):
                self.data = None
                
            def set_data(self, timestamps, opens, highs, lows, closes, volumes):
                self.timestamps = timestamps
                self.opens = opens
                self.highs = highs
                self.lows = lows
                self.closes = closes
                self.volumes = volumes
                
            def calculate_bars(self, params):
                # Simple implementation of bar calculation in Python
                result = PyBarResult(params.bar_type, params.ratio)
                
                if params.bar_type == "volume":
                    # Calculate volume bars
                    volume_sum = 0
                    threshold = params.ratio
                    start_idx = 0
                    
                    for i in range(len(self.volumes)):
                        volume_sum += self.volumes[i]
                        if volume_sum >= threshold or i == len(self.volumes) - 1:
                            # Add bar
                            result.add_bar(
                                i, start_idx, i,
                                self.opens[start_idx], 
                                max(self.highs[start_idx:i+1]),
                                min(self.lows[start_idx:i+1]),
                                self.closes[i],
                                volume_sum
                            )
                            # Reset for next bar
                            volume_sum = 0
                            start_idx = i + 1
                            
                elif params.bar_type == "tick":
                    # Calculate tick bars
                    count = 0
                    threshold = params.ratio
                    start_idx = 0
                    
                    for i in range(len(self.closes)):
                        count += 1
                        if count >= threshold or i == len(self.closes) - 1:
                            # Add bar
                            result.add_bar(
                                i, start_idx, i,
                                self.opens[start_idx], 
                                max(self.highs[start_idx:i+1]),
                                min(self.lows[start_idx:i+1]),
                                self.closes[i],
                                sum(self.volumes[start_idx:i+1])
                            )
                            # Reset for next bar
                            count = 0
                            start_idx = i + 1
                            
                elif params.bar_type == "time":
                    # Calculate time bars (using indices as proxy for time)
                    interval = params.ratio
                    start_idx = 0
                    
                    for i in range(len(self.closes)):
                        if i % interval == 0 and i > 0:
                            end_idx = i - 1
                            # Add bar
                            result.add_bar(
                                end_idx, start_idx, end_idx,
                                self.opens[start_idx], 
                                max(self.highs[start_idx:end_idx+1]),
                                min(self.lows[start_idx:end_idx+1]),
                                self.closes[end_idx],
                                sum(self.volumes[start_idx:end_idx+1])
                            )
                            # Reset for next bar
                            start_idx = i
                        
                        # Add final bar if needed
                        if i == len(self.closes) - 1 and start_idx < i:
                            result.add_bar(
                                i, start_idx, i,
                                self.opens[start_idx], 
                                max(self.highs[start_idx:i+1]),
                                min(self.lows[start_idx:i+1]),
                                self.closes[i],
                                sum(self.volumes[start_idx:i+1])
                            )
                
                elif params.bar_type == "entropy":
                    # Simplified entropy calculation
                    start_idx = 0
                    
                    for i in range(len(self.closes)):
                        if i > 0 and i % 100 == 0:  # Simple rule for entropy bars
                            # Add bar
                            result.add_bar(
                                i, start_idx, i,
                                self.opens[start_idx], 
                                max(self.highs[start_idx:i+1]),
                                min(self.lows[start_idx:i+1]),
                                self.closes[i],
                                sum(self.volumes[start_idx:i+1])
                            )
                            # Reset for next bar
                            start_idx = i + 1
                
                return result
                
            def batch_process(self, params_list):
                return [self.calculate_bars(p) for p in params_list]
        
        # Simple Python implementation of BarResult
        class PyBarResult:
            def __init__(self, bar_type, ratio):
                self.bar_type = bar_type
                self.ratio = ratio
                self.timestamp_indices = []
                self.start_time_indices = []
                self.end_time_indices = []
                self.opens = []
                self.highs = []
                self.lows = []
                self.closes = []
                self.volumes = []
                
            def add_bar(self, ts_idx, start_idx, end_idx, open_val, high, low, close, volume):
                self.timestamp_indices.append(ts_idx)
                self.start_time_indices.append(start_idx)
                self.end_time_indices.append(end_idx)
                self.opens.append(open_val)
                self.highs.append(high)
                self.lows.append(low)
                self.closes.append(close)
                self.volumes.append(volume)
                
            def empty(self):
                return len(self.timestamp_indices) == 0
                
            def size(self):
                return len(self.timestamp_indices)
                
            def to_dict(self, timestamps):
                return {
                    'timestamps': self.timestamp_indices,
                    'start_times': self.start_time_indices,
                    'end_times': self.end_time_indices,
                    'opens': self.opens,
                    'highs': self.highs,
                    'lows': self.lows,
                    'closes': self.closes,
                    'volumes': self.volumes,
                    'bar_type': self.bar_type,
                    'ratio': self.ratio
                }
        
        # Create a mock BarParams class
        class PyBarParams:
            def __init__(self):
                self.bar_type = "time"
                self.ratio = 1.0
                self.lookback_window = 20
                self.window_size = 100
                self.method = "shannon"
                self.q_param = 1.5
        
        # Create the module
        module = SimpleNamespace()
        module.BarCalculator = PyBarCalculator
        module.BarParams = PyBarParams
        
        self._cpp_calculator = module.BarCalculator()
        self._cpp_module = module
        self._data_loaded = False
    
    def set_data(self, df: pd.DataFrame) -> None:
        """
        Set the price data for calculations from a DataFrame.
        
        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        if not all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must have columns: timestamp, open, high, low, close, volume")
        
        # Store the original DataFrame for later reference
        self._original_df = df.copy()
        
        # Convert timestamp to int64 indices for C++ processing
        # We don't convert to milliseconds anymore, just use indices
        # This allows proper handling of minute-level data
        timestamps = np.arange(len(df), dtype=np.int64)  # Use simple indices
            
        # Set the data in the C++ calculator
        self._cpp_calculator.set_data(
            timestamps,
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values
        )
        
        # Store timestamps for result conversion
        self._timestamps = timestamps
        self._data_loaded = True
        print(f"Data loaded into C++ calculator: {len(df)} rows")
    
    def _create_params(self, bar_type: str, ratio: float, 
                     lookback_window: int = 20,
                     window_size: int = 100, method: str = "shannon", 
                     q_param: float = 1.5) -> Any:
        """
        Create a BarParams object for C++ calculation.
        
        Args:
            bar_type: Type of bar ('volume', 'tick', 'time', 'entropy')
            ratio: Multiplier for adaptive threshold calculation
            lookback_window: Number of bars to include in the average calculation
            window_size: Size of rolling window for entropy calculations
            method: Entropy calculation method ('shannon' or 'tsallis')
            q_param: Tsallis q-parameter
            
        Returns:
            BarParams object
        """
        params = self._cpp_module.BarParams()
        
        # Convert string bar type to enum if using C++ implementation
        if hasattr(self._cpp_module, 'BarType'):
            # Map string to enum
            bar_type_map = {
                'volume': self._cpp_module.BarType.Volume,
                'tick': self._cpp_module.BarType.Tick,
                'time': self._cpp_module.BarType.Time,
                'entropy': self._cpp_module.BarType.Entropy,
                'dollar': self._cpp_module.BarType.Dollar,
                'information': self._cpp_module.BarType.Information
            }
            
            # Get the enum value or default to Time
            params.bar_type = bar_type_map.get(bar_type.lower(), self._cpp_module.BarType.Time)
        else:
            # Fallback for Python implementation
            params.bar_type = bar_type
            
        params.ratio = ratio
        params.lookback_window = lookback_window
        params.window_size = window_size
        params.method = method
        params.q_param = q_param
        return params
    
    def _result_to_dataframe(self, result: Any) -> pd.DataFrame:
        """
        Convert a C++ BarResult to a DataFrame.
        
        Args:
            result: BarResult object from C++
            
        Returns:
            DataFrame with bar data
        """
        if result.empty():
            return pd.DataFrame()
            
        # Convert result to dictionary
        result_dict = result.to_dict(self._timestamps)
        
        # Map the indices back to actual timestamps from the original DataFrame
        ts_indices = [int(idx) for idx in result_dict['timestamps']]
        start_indices = [int(idx) for idx in result_dict['start_times']]
        end_indices = [int(idx) for idx in result_dict['end_times']]
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': self._original_df['timestamp'].iloc[ts_indices].values,
            'start_time': self._original_df['timestamp'].iloc[start_indices].values,
            'end_time': self._original_df['timestamp'].iloc[end_indices].values,
            'open': result_dict['opens'],
            'high': result_dict['highs'],
            'low': result_dict['lows'],
            'close': result_dict['closes'],
            'volume': result_dict['volumes']
        })
        
        return df
    
    def calculate_volume_bars(self, volume_ratio: float, lookback_window: int = 20) -> pd.DataFrame:
        """
        Calculate volume bars from the loaded data.
        
        Args:
            volume_ratio: Multiplier for adaptive threshold (ratio * avg_volume)
            lookback_window: Number of bars to include in the average calculation
            
        Returns:
            DataFrame with volume bar data
        """
        if not self._data_loaded:
            raise ValueError("Data not loaded. Call set_data() first.")
            
        params = self._create_params("volume", volume_ratio, lookback_window=lookback_window)
        result = self._cpp_calculator.calculate_bars(params)
        return self._result_to_dataframe(result)
    
    def calculate_tick_bars(self, tick_ratio: float, lookback_window: int = 20) -> pd.DataFrame:
        """
        Calculate tick bars from the loaded data.
        
        Args:
            tick_ratio: Multiplier for adaptive threshold (ratio * avg_ticks)
            lookback_window: Number of bars to include in the average calculation
            
        Returns:
            DataFrame with tick bar data
        """
        if not self._data_loaded:
            raise ValueError("Data not loaded. Call set_data() first.")
            
        params = self._create_params("tick", tick_ratio, lookback_window=lookback_window)
        result = self._cpp_calculator.calculate_bars(params)
        return self._result_to_dataframe(result)
    
    def calculate_time_bars(self, seconds: float) -> pd.DataFrame:
        """
        Calculate time bars from the loaded data.
        
        Args:
            seconds: Time interval in seconds
            
        Returns:
            DataFrame with time bar data
        """
        if not self._data_loaded:
            raise ValueError("Data not loaded. Call set_data() first.")
            
        params = self._create_params("time", seconds)
        result = self._cpp_calculator.calculate_bars(params)
        return self._result_to_dataframe(result)
    
    def calculate_entropy_bars(self, entropy_ratio: float, 
                              lookback_window: int = 20,
                              window_size: int = 100, 
                              method: str = "shannon",
                              q_param: float = 1.5) -> pd.DataFrame:
        """
        Calculate entropy bars from the loaded data.
        
        Args:
            entropy_ratio: Multiplier for adaptive threshold (ratio * avg_entropy)
            lookback_window: Number of bars to include in the average calculation
            window_size: Size of rolling window for entropy calculation
            method: Entropy calculation method ('shannon' or 'tsallis')
            q_param: Tsallis q-parameter (used only with 'tsallis' method)
            
        Returns:
            DataFrame with entropy bar data
        """
        if not self._data_loaded:
            raise ValueError("Data not loaded. Call set_data() first.")
            
        params = self._create_params(
            "entropy", entropy_ratio, 
            lookback_window=lookback_window,
            window_size=window_size, method=method, q_param=q_param
        )
        result = self._cpp_calculator.calculate_bars(params)
        return self._result_to_dataframe(result)
    
    def batch_calculate(self, params_list: List[Dict[str, Any]]) -> List[pd.DataFrame]:
        """
        Calculate multiple bar types in a single pass.
        
        Args:
            params_list: List of parameter dictionaries:
                         [{'bar_type': str, 'ratio': float, 'lookback_window': int, 
                           'window_size': int, 'method': str, 'q_param': float}, ...]
            
        Returns:
            List of DataFrames with bar data, in the same order as params_list
        """
        if not self._data_loaded:
            raise ValueError("Data not loaded. Call set_data() first.")
            
        # Create BarParams objects for each parameter set
        cpp_params = []
        for params in params_list:
            bar_type = params.get('bar_type')
            ratio = params.get('ratio')
            
            if not bar_type or ratio is None:
                raise ValueError("Each parameter set must have 'bar_type' and 'ratio'")
                
            lookback_window = params.get('lookback_window', 20)
            window_size = params.get('window_size', 100)
            method = params.get('method', 'shannon')
            q_param = params.get('q_param', 1.5)
            
            cpp_params.append(self._create_params(
                bar_type, ratio, lookback_window, window_size, method, q_param
            ))
        
        # Calculate all bar types
        results = self._cpp_calculator.batch_process(cpp_params)
        
        # Convert results to DataFrames
        return [self._result_to_dataframe(result) for result in results] 