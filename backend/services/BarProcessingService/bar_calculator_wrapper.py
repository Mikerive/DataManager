"""
Python wrapper for the C++ BarCalculator implementation.
This provides an easy-to-use interface to the high-performance C++ code.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any

# This will be imported when the C++ extension is built
# bar_calculator_cpp = None  # Will be set when extension is built

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
            from backend.services.BarProcessingService.cpp_ext import bar_calculator_cpp
            self._cpp_calculator = bar_calculator_cpp.BarCalculator()
            self._cpp_module = bar_calculator_cpp
            self._data_loaded = False
        except ImportError:
            raise ImportError("C++ extension not built. Run setup.py to build the extension.")
    
    def set_data(self, df: pd.DataFrame) -> None:
        """
        Set the price data for calculations from a DataFrame.
        
        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        if not all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must have columns: timestamp, open, high, low, close, volume")
        
        # Convert timestamp to int64 (milliseconds)
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            timestamps = df['timestamp'].astype(np.int64) // 10**6  # nanoseconds to milliseconds
        else:
            timestamps = df['timestamp'].astype(np.int64)
            
        # Set the data in the C++ calculator
        self._cpp_calculator.set_data(
            timestamps.values,
            df['open'].values,
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values
        )
        
        # Store timestamps for result conversion
        self._timestamps = timestamps.values
        self._data_loaded = True
    
    def _create_params(self, bar_type: str, ratio: float, 
                     window_size: int = 100, method: str = "shannon", 
                     q_param: float = 1.5) -> Any:
        """
        Create a BarParams object for C++ calculation.
        
        Args:
            bar_type: Type of bar ('volume', 'tick', 'time', 'entropy')
            ratio: Threshold for bar formation
            window_size: Size of rolling window for calculations
            method: Entropy calculation method ('shannon' or 'tsallis')
            q_param: Tsallis q-parameter
            
        Returns:
            BarParams object
        """
        params = self._cpp_module.BarParams()
        params.bar_type = bar_type
        params.ratio = ratio
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
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(result_dict['timestamps'], unit='ms'),
            'start_time': pd.to_datetime(result_dict['start_times'], unit='ms'),
            'end_time': pd.to_datetime(result_dict['end_times'], unit='ms'),
            'open': result_dict['opens'],
            'high': result_dict['highs'],
            'low': result_dict['lows'],
            'close': result_dict['closes'],
            'volume': result_dict['volumes']
        })
        
        return df
    
    def calculate_volume_bars(self, volume_threshold: float) -> pd.DataFrame:
        """
        Calculate volume bars from the loaded data.
        
        Args:
            volume_threshold: Volume threshold for bar formation
            
        Returns:
            DataFrame with volume bar data
        """
        if not self._data_loaded:
            raise ValueError("Data not loaded. Call set_data() first.")
            
        params = self._create_params("volume", volume_threshold)
        result = self._cpp_calculator.calculate_bars(params)
        return self._result_to_dataframe(result)
    
    def calculate_tick_bars(self, tick_count: int) -> pd.DataFrame:
        """
        Calculate tick bars from the loaded data.
        
        Args:
            tick_count: Number of ticks per bar
            
        Returns:
            DataFrame with tick bar data
        """
        if not self._data_loaded:
            raise ValueError("Data not loaded. Call set_data() first.")
            
        params = self._create_params("tick", float(tick_count))
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
        if not self._data_loaded:
            raise ValueError("Data not loaded. Call set_data() first.")
            
        params = self._create_params(
            "entropy", entropy_threshold, 
            window_size=window_size, method=method, q_param=q_param
        )
        result = self._cpp_calculator.calculate_bars(params)
        return self._result_to_dataframe(result)
    
    def batch_calculate(self, params_list: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate multiple bar types in a single pass.
        
        Args:
            params_list: List of parameter dictionaries:
                         [{'bar_type': str, 'ratio': float, 'window_size': int, 'method': str, 'q_param': float}, ...]
            
        Returns:
            Dictionary mapping bar type to DataFrame with bar data
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
                
            window_size = params.get('window_size', 100)
            method = params.get('method', 'shannon')
            q_param = params.get('q_param', 1.5)
            
            cpp_params.append(self._create_params(
                bar_type, ratio, window_size, method, q_param
            ))
        
        # Calculate all bar types
        results = self._cpp_calculator.batch_process(cpp_params)
        
        # Convert results to DataFrames
        return {key: self._result_to_dataframe(result) for key, result in results.items()} 