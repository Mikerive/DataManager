import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import os
import importlib.util
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from backend.db.models.ProcessedData import ProcessedData
from backend.db.models.RawData import RawData
from backend.db.models.Tickers import Tickers
from backend.db.Database import Database
from backend.db.utils.db_utils import log_db_error, log_db_success

# Import all utility functions from our new modules
from backend.services.BarProcessingService.utils import (
    calculate_volume_bars,
    calculate_tick_bars,
    calculate_time_bars,
    calculate_price_bars,
    calculate_dollar_bars,
    calculate_entropy_bars,
    calculate_information_bars,
    prepare_dataframe,
    validate_bar_parameters,
    enrich_bar_dataframe,
    merge_bar_dataframes
)


class BarProcessingService:
    """
    Service for processing raw financial data into different types of bars for a specific ticker.
    
    This service is responsible for:
    1. Fetching raw data from the database for a single ticker
    2. Processing it into various bar types using specialized calculation functions
    3. Storing processed bars back to the database or file system
    4. Managing the processing pipeline and workflow
    
    The service delegates specific bar calculation logic to utility functions while
    handling the high-level workflow, caching, and persistence.
    
    By default, it uses Cython-optimized functions when available for significantly
    improved performance, especially for computationally intensive operations like
    entropy calculations.
    """

    def __init__(
        self,
        ticker: str,
        output_dir: str = "bar_data",
        window_size: int = 100,
        calculation_method: str = "shannon",
        avg_window: int = 200,
        force_python_impl: bool = False
    ):
        """
        Initialize the BarProcessingService with a specific ticker and configuration.
        
        Args:
            ticker: The ticker symbol to process
            output_dir: Directory to store processed bar data
            window_size: Window size for rolling window calculations (default: 100)
            calculation_method: Method used for calculations (e.g., "shannon" for entropy)
            avg_window: Window size for running average (default: 200)
            force_python_impl: If True, force using Python implementations even if Cython is available
        """
        self.ticker = ticker
        self.output_dir = output_dir
        self.window_size = window_size
        self.calculation_method = calculation_method
        self.avg_window = avg_window
        self.logger = logging.getLogger(__name__)
        
        # Check if Cython is available and log accordingly
        if force_python_impl:
            self._force_python_implementations()
            self.logger.info("Forced using pure Python implementations for bar calculations")
        elif self.is_cython_available():
            self.logger.info("Using Cython-optimized implementations for bar calculations")
        else:
            self.logger.warning("Cython-optimized implementations not available, using pure Python")

        # Initialize data cache
        self.raw_data = None  # DataFrame for the ticker's raw data
        self.processed_data_cache = {}  # Dict mapping bar_type_ratio to processed DataFrames
        
        # Ensure output directory exists
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

    def is_cython_available(self) -> bool:
        """
        Check if Cython-optimized implementations are available.
        
        Returns:
            True if Cython module is available, False otherwise
        """
        try:
            # Check if the Cython module is already loaded or can be imported
            if importlib.util.find_spec("backend.services.BarProcessingService.utils.bar_types_cy"):
                return True
                
            # Alternative check: try to import it directly
            import backend.services.BarProcessingService.utils.bar_types_cy
            return True
        except (ImportError, ModuleNotFoundError):
            return False
            
    def _force_python_implementations(self):
        """
        Force the service to use pure Python implementations even if Cython is available.
        This is useful for benchmarking or troubleshooting.
        
        Note: This modifies the global module imports, so it affects all instances.
        """
        try:
            import sys
            from backend.services.BarProcessingService.utils import bar_types
            
            # Replace the imported functions with Python implementations
            global calculate_volume_bars, calculate_tick_bars, calculate_time_bars
            global calculate_price_bars, calculate_dollar_bars
            global calculate_entropy_bars, calculate_information_bars
            
            calculate_volume_bars = bar_types.calculate_volume_bars
            calculate_tick_bars = bar_types.calculate_tick_bars
            calculate_time_bars = bar_types.calculate_time_bars
            calculate_price_bars = bar_types.calculate_price_bars
            calculate_dollar_bars = bar_types.calculate_dollar_bars
            calculate_entropy_bars = bar_types.calculate_entropy_bars
            calculate_information_bars = bar_types.calculate_information_bars
            
            self.logger.info("Successfully forced Python implementations")
        except Exception as e:
            self.logger.error(f"Failed to force Python implementations: {str(e)}")

    def __del__(self):
        """
        Destructor to ensure cache is cleared when object is garbage collected.
        """
        self.clear_cache()
        self.logger.debug(f"BarProcessingService for {self.ticker} destroyed and cache cleared")

    async def change_ticker(self, new_ticker: str, load_data: bool = True, start_time: datetime = None, end_time: datetime = None) -> bool:
        """
        Change the ticker symbol, clear all cached data, and optionally load data for the new ticker.
        
        Args:
            new_ticker: The new ticker symbol to switch to
            load_data: Whether to automatically load raw data for the new ticker
            start_time: Start datetime for loading data (required if load_data is True)
            end_time: End datetime for loading data (required if load_data is True)
            
        Returns:
            Boolean indicating success
        """
        try:
            # First clear the cache for the current ticker
            self.clear_cache()
            
            self.logger.info(f"Changing ticker from {self.ticker} to {new_ticker}")
            
            # Update the ticker
            self.ticker = new_ticker
            
            # Create directory for the new ticker if it doesn't exist
            ticker_dir = os.path.join(self.output_dir, new_ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            
            # Optionally load raw data for the new ticker
            if load_data:
                if not start_time or not end_time:
                    self.logger.error("Cannot load data: start_time and end_time are required when load_data is True")
                    return False
                
                await self.load_data(start_time, end_time)
                
                if not await self.has_cached_data():
                    self.logger.warning(f"Changed to ticker {new_ticker} but failed to load any data")
                else:
                    self.logger.info(f"Changed to ticker {new_ticker} and loaded {len(self.raw_data)} data points")
            else:
                self.logger.info(f"Successfully changed to ticker: {new_ticker}")
                
            return True
        except Exception as e:
            self.logger.error(f"Error changing ticker to {new_ticker}: {str(e)}")
            return False

    async def load_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load raw data for the ticker and cache it.
        
        Args:
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with raw price data
        """
        try:
            # Get raw data using RawData class method
            df = await RawData.get_price_data(
                ticker=self.ticker, 
                start_date=start_time, 
                end_date=end_time
            )
            
            if not df.empty:
                # Prepare for processing using utility function
                df = prepare_dataframe(df)
                
                # Cache the data if valid
                if not df.empty:
                    self.raw_data = df
                    self.logger.info(f"Loaded and cached {len(df)} data points for {self.ticker}")
                else:
                    self.logger.warning(f"Data validation failed for {self.ticker}")
            else:
                self.logger.warning(f"No data found for {self.ticker}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {self.ticker}: {str(e)}")
            return pd.DataFrame()

    async def has_cached_data(self) -> bool:
        """
        Check if raw data is cached for the ticker.
        
        Returns:
            Boolean indicating if data is cached
        """
        return self.raw_data is not None and not self.raw_data.empty

    async def process_bar(self, bar_type: str, ratio: float, **kwargs) -> pd.DataFrame:
        """
        Process raw data into a specific bar type with a given ratio.
        
        Args:
            bar_type: Type of bar ('volume', 'tick', 'entropy', 'time', 'dollar', etc.)
            ratio: The parameter value for the bar calculation (will be converted to int)
            **kwargs: Additional parameters specific to the bar type
            
        Returns:
            DataFrame: Processed bars or empty DataFrame if unsuccessful
        """
        if not await self.has_cached_data():
            self.logger.warning(f"No cached data found for {self.ticker}")
            return pd.DataFrame()

        try:
            # Ensure ratio is an integer
            ratio = int(ratio)
            
            # Validate parameters
            is_valid, error_msg = validate_bar_parameters(bar_type, ratio, **kwargs)
            if not is_valid:
                self.logger.error(error_msg)
                return pd.DataFrame()
            
            # Call the specific processing method based on bar type
            bar_type_processors = {
                'volume': self._process_volume_bars,
                'tick': self._process_tick_bars,
                'time': self._process_time_bars,
                'price': self._process_price_bars,
                'dollar': self._process_dollar_bars,
                'entropy': self._process_entropy_bars,
                'information': self._process_information_bars
            }
            
            if bar_type not in bar_type_processors:
                self.logger.error(f"Unsupported bar type: {bar_type}")
                return pd.DataFrame()
            
            # Process bars with the appropriate calculation function
            bars_df = await bar_type_processors[bar_type](ratio, **kwargs)

            if bars_df.empty:
                self.logger.info(f"No {bar_type} bars generated for {self.ticker} with ratio {ratio}")
                return pd.DataFrame()

            # Enrich the DataFrame with additional metadata
            bars_df = enrich_bar_dataframe(bars_df, bar_type, ratio)
            
            # Store the processed bars in cache
            cache_key = f"{bar_type}_{ratio}"
            self.processed_data_cache[cache_key] = bars_df
            
            # Store in database and file system
            await self.store_bars(bars_df, bar_type, ratio)

            self.logger.info(f"Successfully processed {len(bars_df)} {bar_type} bars with ratio {ratio} for {self.ticker}")
            return bars_df

        except ValueError as e:
            self.logger.error(f"Invalid ratio for {bar_type} bars: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error processing {bar_type} bars for {self.ticker}: {str(e)}")
            return pd.DataFrame()

    async def _process_volume_bars(self, ratio: float, **kwargs) -> pd.DataFrame:
        """
        Process raw data into volume bars with a given ratio.
        
        Args:
            ratio: The volume ratio parameter (will be converted to int)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with processed volume bars
        """
        return calculate_volume_bars(
            self.raw_data, 
            int(ratio), 
            avg_window=kwargs.get('avg_window', self.avg_window)
        )

    async def _process_tick_bars(self, ratio: float, **kwargs) -> pd.DataFrame:
        """
        Process raw data into tick bars with a given tick size.
        
        Args:
            ratio: The tick size parameter (will be converted to int)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with processed tick bars
        """
        return calculate_tick_bars(
            self.raw_data, 
            tick_size=int(ratio)
        )

    async def _process_time_bars(self, ratio: float, **kwargs) -> pd.DataFrame:
        """
        Process raw data into time bars with a given interval.
        
        Args:
            ratio: The time interval in minutes (will be converted to int)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with processed time bars
        """
        return calculate_time_bars(
            self.raw_data, 
            time_interval_minutes=int(ratio)
        )

    async def _process_price_bars(self, ratio: float, **kwargs) -> pd.DataFrame:
        """
        Process raw data into price bars with a given threshold.
        
        Args:
            ratio: The price threshold parameter (will be converted to int)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with processed price bars
        """
        return calculate_price_bars(
            self.raw_data, 
            price_threshold=int(ratio)
        )

    async def _process_dollar_bars(self, ratio: float, **kwargs) -> pd.DataFrame:
        """
        Process raw data into dollar bars with a given threshold.
        
        Args:
            ratio: The dollar threshold parameter (will be converted to int)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with processed dollar bars
        """
        return calculate_dollar_bars(
            self.raw_data, 
            dollar_threshold=int(ratio)
        )

    async def _process_entropy_bars(self, ratio: float, **kwargs) -> pd.DataFrame:
        """
        Process raw data into entropy bars with a given threshold.
        
        Args:
            ratio: The entropy threshold parameter (will be converted to int)
            **kwargs: Additional parameters including window_size, method, and avg_window
            
        Returns:
            DataFrame with processed entropy bars
        """
        return calculate_entropy_bars(
            self.raw_data, 
            entropy_threshold=int(ratio),
            window_size=kwargs.get('window_size', self.window_size),
            method=kwargs.get('method', self.calculation_method),
            avg_window=kwargs.get('avg_window', self.avg_window)
        )

    async def _process_information_bars(self, ratio: float, **kwargs) -> pd.DataFrame:
        """
        Process raw data into information bars with a given threshold.
        
        Args:
            ratio: The information threshold parameter (will be converted to int)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with processed information bars
        """
        return calculate_information_bars(
            self.raw_data, 
            info_threshold=int(ratio)
        )

    async def store_bars(self, bars_df: pd.DataFrame, bar_type: str, ratio: float):
        """
        Store processed bars to database and file system.
        
        Args:
            bars_df: DataFrame with processed bars
            bar_type: Type of bar
            ratio: The ratio/threshold value for this bar type
        """
        if bars_df.empty:
            return

        # Store in database
        await self.store_bars_in_db(bars_df, bar_type, ratio)

        # Store in file system
        self.store_bars_in_file(bars_df, bar_type, ratio)

    async def store_bars_in_db(self, df: pd.DataFrame, bar_type: str, ratio: float):
        """
        Store processed bars in the database.
        
        Args:
            df: DataFrame with bar data
            bar_type: Type of bar ('volume', 'tick', 'entropy', 'price')
            ratio: The ratio/threshold value for this bar type
        """
        try:
            # Store data using class method
            await ProcessedData.add_dataframe_by_symbol(
                ticker=self.ticker,
                bar_type=bar_type,
                df=df,
                ratio=ratio
            )

            self.logger.info(f"Stored {len(df)} {bar_type} bars (ratio: {ratio}) for {self.ticker} in database")

        except Exception as e:
            self.logger.error(f"Error storing {bar_type} bars for {self.ticker} in database: {str(e)}")

    def store_bars_in_file(self, df: pd.DataFrame, bar_type: str, ratio: float):
        """
        Store processed bars in parquet files.
        
        Args:
            df: DataFrame with bar data
            bar_type: Type of bar ('volume', 'tick', 'entropy', 'price')
            ratio: The ratio/threshold value for this bar type
        """
        try:
            # Create directory structure
            symbol_dir = os.path.join(self.output_dir, self.ticker)
            os.makedirs(symbol_dir, exist_ok=True)

            # Create filename
            file_name = f"{bar_type}_{ratio}.parquet"
            file_path = os.path.join(symbol_dir, file_name)

            # Save to parquet
            df.to_parquet(file_path, index=False)

            self.logger.info(f"Stored {len(df)} {bar_type} bars (ratio: {ratio}) for {self.ticker} in {file_path}")

        except Exception as e:
            self.logger.error(f"Error storing {bar_type} bars for {self.ticker} in file: {str(e)}")

    async def process_multiple_bars(self, bar_types_ratios: Dict[str, List[float]], **kwargs) -> Dict[str, Dict[float, pd.DataFrame]]:
        """
        Process raw data into multiple bar types with various ratios.
        
        Args:
            bar_types_ratios: Dictionary mapping bar types to lists of ratios
            **kwargs: Additional parameters to pass to bar calculation functions
            
        Returns:
            Nested dictionary: {bar_type: {ratio: DataFrame}}
        """
        if not await self.has_cached_data():
            self.logger.warning(f"No cached data found for {self.ticker}")
            return {}

        results = {}
        for bar_type, ratios in bar_types_ratios.items():
            results[bar_type] = {}
            for ratio in ratios:
                bars_df = await self.process_bar(bar_type, ratio, **kwargs)
                results[bar_type][ratio] = bars_df

        return results

    def clear_cache(self):
        """Clear all data caches to free memory."""
        self.raw_data = None
        self.processed_data_cache = {}
        self.logger.info(f"Cleared all cached data for {self.ticker}")

    async def update_params(self, window_size: int = None, 
                             avg_window: int = None, 
                             calculation_method: str = None):
        """
        Update calculation parameters.
        
        Args:
            window_size: New window size for calculations
            avg_window: New window size for running average
            calculation_method: New calculation method
            
        Returns:
            Boolean indicating success
        """
        try:
            if window_size is not None:
                self.window_size = window_size
                
            if avg_window is not None:
                self.avg_window = avg_window
                
            if calculation_method is not None:
                self.calculation_method = calculation_method
                
            self.logger.info(f"Updated parameters: window_size={self.window_size}, "
                           f"avg_window={self.avg_window}, calculation_method={self.calculation_method}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating parameters: {str(e)}")
            return False

    async def get_available_bar_types(self) -> List[str]:
        """
        Get list of all available bar types for this ticker.
        
        Returns:
            List of bar type strings
        """
        return ["volume", "tick", "price", "dollar", "time", "entropy", "information"]

    async def get_bar_type_parameters(self, bar_type: str) -> Dict[str, Any]:
        """
        Get parameter information for a specific bar type.
        
        Args:
            bar_type: Type of bar
            
        Returns:
            Dictionary with parameter information
        """
        param_info = {
            "volume": {
                "name": "Volume Ratio",
                "range": [1, 100],
                "default": 10
            },
            "tick": {
                "name": "Tick Ratio",
                "range": [1, 1000],
                "default": 100
            },
            "price": {
                "name": "Price Ratio",
                "range": [1, 100],
                "default": 10
            },
            "dollar": {
                "name": "Dollar Ratio",
                "range": [1, 1000],
                "default": 100
            },
            "entropy": {
                "name": "Entropy Ratio",
                "range": [1, 50],
                "default": 5,
                "additional": {
                    "window_size": {
                        "name": "Window Size",
                        "range": [10, 500],
                        "default": 100
                    },
                    "avg_window": {
                        "name": "Average Window",
                        "range": [10, 500],
                        "default": 200
                    },
                    "method": {
                        "name": "Calculation Method",
                        "options": ["shannon", "tsallis"],
                        "default": "shannon"
                    }
                }
            },
            "information": {
                "name": "Information Ratio",
                "range": [1, 50],
                "default": 5
            },
            "time": {
                "name": "Timeframe (minutes)",
                "options": [1, 5, 15, 30, 60, 240, 1440],
                "default": 15
            }
        }
        
        return param_info.get(bar_type, {})

    async def generate_bars(
        self, 
        bar_type: str, 
        ratio: float, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None, 
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate bars with specified parameters.
        If no raw data is cached, it will load data from the database.
        
        Args:
            bar_type: Type of bar to generate
            ratio: Parameter value for the bar generation (will be converted to int)
            start_time: Start time for data range (if data needs to be loaded)
            end_time: End time for data range (if data needs to be loaded)
            **kwargs: Additional parameters specific to bar types
            
        Returns:
            DataFrame with generated bars
        """
        try:
            # Ensure ratio is an integer
            ratio = int(ratio)
            
            # Check if we need to load data first
            if not await self.has_cached_data():
                if not start_time or not end_time:
                    self.logger.error("No cached data and no date range provided")
                    return pd.DataFrame()
                    
                # Load data for the ticker
                await self.load_data(start_time, end_time)
                
                if not await self.has_cached_data():
                    self.logger.warning(f"Failed to load data for {self.ticker}")
                    return pd.DataFrame()
                
            # Check the cache first
            cache_key = f"{bar_type}_{ratio}"
            if cache_key in self.processed_data_cache:
                self.logger.info(f"Using cached {bar_type} bars for {self.ticker}")
                return self.processed_data_cache[cache_key]
            
            # Generate the bars through process_bar
            bars_df = await self.process_bar(bar_type, ratio, **kwargs)
            
            return bars_df
        except ValueError as e:
            self.logger.error(f"Invalid ratio for {bar_type} bars: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error generating {bar_type} bars for {self.ticker}: {str(e)}")
            return pd.DataFrame()

    async def save_generated_bars(self, bar_type: str, bars_df: pd.DataFrame, ratio: float) -> bool:
        """
        Save generated bars to database and file system.
        
        Args:
            bar_type: Type of bar
            bars_df: DataFrame with generated bars
            ratio: Parameter value used to generate bars
            
        Returns:
            Success status
        """
        try:
            if bars_df.empty:
                self.logger.warning("No bars to save")
                return False
                
            # Store in database and file system
            await self.store_bars(bars_df, bar_type, ratio)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {bar_type} bars for {self.ticker}: {str(e)}")
            return False

    async def get_available_processed_data(self) -> Dict[str, List[str]]:
        """
        Get information about available processed data for this ticker.
        
        Returns:
            Dictionary with available bar types
        """
        try:
            # Get all available data from ProcessedData for this ticker
            available_data = await ProcessedData.list_available_data(self.ticker)
            
            # Extract just the bar types for this ticker
            if self.ticker in available_data:
                return available_data[self.ticker]
            else:
                # Try with ticker_id lookup
                ticker_info = await Tickers.get_ticker(self.ticker)
                if ticker_info:
                    ticker_id = ticker_info['id']
                    ticker_key = f"ticker_id_{ticker_id}"
                    if ticker_key in available_data:
                        return available_data[ticker_key]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting available processed data for {self.ticker}: {str(e)}")
            return []
            
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the cached raw data.
        
        Returns:
            Dictionary with statistics like count, date range, etc.
        """
        if not await self.has_cached_data():
            return {
                "ticker": self.ticker,
                "cached": False,
                "count": 0,
                "start_date": None,
                "end_date": None,
                "avg_volume": None
            }
            
        df = self.raw_data
        return {
            "ticker": self.ticker,
            "cached": True,
            "count": len(df),
            "start_date": df.index.min(),
            "end_date": df.index.max(),
            "avg_volume": df['volume'].mean() if 'volume' in df.columns else None,
            "bar_types_cached": list(self.processed_data_cache.keys())
        }

    async def benchmark_performance(self, bar_type: str, ratio: int, sample_size: int = 10000, runs: int = 3) -> Dict:
        """
        Benchmark the performance difference between Cython and Python implementations.
        
        Args:
            bar_type: Type of bar to benchmark ('volume', 'tick', 'entropy', etc.)
            ratio: Parameter value for the calculation
            sample_size: Number of data points to use (will be random if more data is available)
            runs: Number of benchmark runs to average
            
        Returns:
            Dictionary with benchmark results including Python time, Cython time, and speedup factor
        """
        if not await self.has_cached_data():
            self.logger.warning("No data available for benchmarking")
            return {"error": "No data available"}
            
        try:
            # Create a sample of the data
            if len(self.raw_data) > sample_size:
                sample_data = self.raw_data.sample(sample_size)
            else:
                sample_data = self.raw_data
                
            # Import both implementations directly to ensure we can compare
            from backend.services.BarProcessingService.utils import bar_types
            
            # Function mappings for each implementation
            py_functions = {
                'volume': bar_types.calculate_volume_bars,
                'tick': bar_types.calculate_tick_bars,
                'time': bar_types.calculate_time_bars,
                'price': bar_types.calculate_price_bars,
                'dollar': bar_types.calculate_dollar_bars,
                'entropy': bar_types.calculate_entropy_bars,
                'information': bar_types.calculate_information_bars
            }
            
            # Check if Cython is available
            try:
                from backend.services.BarProcessingService.utils import bar_types_cy
                cy_available = True
                cy_functions = {
                    'volume': bar_types_cy.calculate_volume_bars,
                    'tick': bar_types_cy.calculate_tick_bars,
                    'time': bar_types_cy.calculate_time_bars,
                    'price': bar_types_cy.calculate_price_bars,
                    'dollar': bar_types_cy.calculate_dollar_bars,
                    'entropy': bar_types_cy.calculate_entropy_bars,
                    'information': bar_types_cy.calculate_information_bars
                }
            except ImportError:
                cy_available = False
                
            if bar_type not in py_functions:
                return {"error": f"Invalid bar type: {bar_type}"}
                
            # Setup kwargs based on bar type
            kwargs = {}
            if bar_type == 'entropy':
                kwargs = {
                    'window_size': self.window_size,
                    'method': self.calculation_method,
                    'avg_window': self.avg_window
                }
                
            # Benchmark Python implementation
            py_times = []
            for i in range(runs):
                start_time = asyncio.get_event_loop().time()
                if bar_type == 'volume':
                    py_result = py_functions[bar_type](sample_data, ratio, **kwargs)
                elif bar_type == 'tick':
                    py_result = py_functions[bar_type](sample_data, tick_size=ratio)
                elif bar_type == 'time':
                    py_result = py_functions[bar_type](sample_data, time_interval_minutes=ratio)
                elif bar_type == 'price':
                    py_result = py_functions[bar_type](sample_data, price_threshold=ratio)
                elif bar_type == 'dollar':
                    py_result = py_functions[bar_type](sample_data, dollar_threshold=ratio)
                elif bar_type == 'entropy':
                    py_result = py_functions[bar_type](sample_data, entropy_threshold=ratio, **kwargs)
                elif bar_type == 'information':
                    py_result = py_functions[bar_type](sample_data, info_threshold=ratio)
                end_time = asyncio.get_event_loop().time()
                py_times.append(end_time - start_time)
                
            python_avg_time = sum(py_times) / len(py_times)
            python_result_size = len(py_result) if not py_result.empty else 0
            
            results = {
                "bar_type": bar_type,
                "ratio": ratio,
                "data_points": len(sample_data),
                "python_time": python_avg_time,
                "python_result_size": python_result_size,
                "runs": runs
            }
            
            # Benchmark Cython implementation if available
            if cy_available:
                cy_times = []
                for i in range(runs):
                    start_time = asyncio.get_event_loop().time()
                    if bar_type == 'volume':
                        cy_result = cy_functions[bar_type](sample_data, ratio, **kwargs)
                    elif bar_type == 'tick':
                        cy_result = cy_functions[bar_type](sample_data, tick_size=ratio)
                    elif bar_type == 'time':
                        cy_result = cy_functions[bar_type](sample_data, time_interval_minutes=ratio)
                    elif bar_type == 'price':
                        cy_result = cy_functions[bar_type](sample_data, price_threshold=ratio)
                    elif bar_type == 'dollar':
                        cy_result = cy_functions[bar_type](sample_data, dollar_threshold=ratio)
                    elif bar_type == 'entropy':
                        cy_result = cy_functions[bar_type](sample_data, entropy_threshold=ratio, **kwargs)
                    elif bar_type == 'information':
                        cy_result = cy_functions[bar_type](sample_data, info_threshold=ratio)
                    end_time = asyncio.get_event_loop().time()
                    cy_times.append(end_time - start_time)
                    
                cython_avg_time = sum(cy_times) / len(cy_times)
                cython_result_size = len(cy_result) if not cy_result.empty else 0
                speedup = python_avg_time / cython_avg_time if cython_avg_time > 0 else 0
                
                results.update({
                    "cython_time": cython_avg_time,
                    "cython_result_size": cython_result_size,
                    "speedup": speedup
                })
                
                if python_result_size != cython_result_size:
                    self.logger.warning(
                        f"Result size mismatch: Python={python_result_size}, "
                        f"Cython={cython_result_size}"
                    )
                    results["warning"] = "Result size mismatch between implementations"
            else:
                results["cython_available"] = False
                results["note"] = "Cython implementation not available, compile it for performance gains"
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error in benchmark: {str(e)}")
            return {"error": str(e)}
