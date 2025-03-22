import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import tempfile
import shutil

# Import the service
from backend.services.BarProcessingService.BarProcessingService import BarProcessingService
from backend.db.models.RawData import RawData
from backend.db.models.ProcessedData import ProcessedData
from backend.db.models.Tickers import Tickers

class TestBarProcessingService(unittest.TestCase):
    """Test the BarProcessingService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test ticker
        self.ticker = "AAPL"
        
        # Create service instance with temp directory
        self.service = BarProcessingService(
            ticker=self.ticker, 
            output_dir=self.temp_dir,
            window_size=50,
            avg_window=100,
            calculation_method="shannon"
        )
        
        # Create sample raw data
        timestamps = pd.date_range(
            start=datetime(2023, 1, 1, 9, 30), 
            end=datetime(2023, 1, 1, 16, 0), 
            freq='1min'
        )
        n = len(timestamps)
        
        # Create sample data with some realistic patterns
        self.sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': 150.0 + np.cumsum(np.random.normal(0, 0.1, n)),
            'volume': np.random.randint(100, 5000, n),
            'open': 150.0 + np.cumsum(np.random.normal(0, 0.1, n)),
            'high': 150.0 + np.cumsum(np.random.normal(0, 0.1, n)) + np.random.uniform(0.1, 0.5, n),
            'low': 150.0 + np.cumsum(np.random.normal(0, 0.1, n)) - np.random.uniform(0.1, 0.5, n),
            'close': 150.0 + np.cumsum(np.random.normal(0, 0.1, n))
        })
        
        self.sample_data = self.sample_data.set_index('timestamp')
        
        # Set up the event loop for async tests
        self.loop = asyncio.get_event_loop()
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def run_async(self, coro):
        """Helper to run an async test method."""
        return self.loop.run_until_complete(coro)
    
    @patch('backend.db.models.RawData.RawData.get_price_data')
    def test_load_data(self, mock_get_price_data):
        """Test loading data."""
        # Set up the mock
        mock_get_price_data.return_value = self.sample_data
        
        # Run the async method
        result = self.run_async(self.service.load_data(
            start_time=datetime(2023, 1, 1, 9, 30),
            end_time=datetime(2023, 1, 1, 16, 0)
        ))
        
        # Verify the result
        self.assertFalse(result.empty)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Verify the mock was called with correct args
        mock_get_price_data.assert_called_once_with(
            ticker=self.ticker, 
            start_date=datetime(2023, 1, 1, 9, 30),
            end_date=datetime(2023, 1, 1, 16, 0)
        )
        
        # Verify the data was cached
        self.assertTrue(self.run_async(self.service.has_cached_data()))
    
    @patch('backend.db.models.ProcessedData.ProcessedData.add_dataframe_by_symbol')
    @patch('backend.db.models.RawData.RawData.get_price_data')
    def test_process_volume_bars(self, mock_get_price_data, mock_add_dataframe):
        """Test processing volume bars."""
        # Set up the mocks
        mock_get_price_data.return_value = self.sample_data
        mock_add_dataframe.return_value = None
        
        # Load data
        self.run_async(self.service.load_data(
            start_time=datetime(2023, 1, 1, 9, 30),
            end_time=datetime(2023, 1, 1, 16, 0)
        ))
        
        # Process volume bars
        result = self.run_async(self.service.process_bar(
            bar_type='volume',
            ratio=10
        ))
        
        # Verify the result
        self.assertFalse(result.empty)
        self.assertIn('bar_type', result.columns)
        self.assertEqual(result['bar_type'].iloc[0], 'volume')
        self.assertIn('ratio', result.columns)
        self.assertEqual(result['ratio'].iloc[0], 10)
        
        # Verify store_bars_in_db was called with correct args
        mock_add_dataframe.assert_called_once()
    
    @patch('backend.db.models.ProcessedData.ProcessedData.add_dataframe_by_symbol')
    @patch('backend.db.models.RawData.RawData.get_price_data')
    def test_process_tick_bars(self, mock_get_price_data, mock_add_dataframe):
        """Test processing tick bars."""
        # Set up the mocks
        mock_get_price_data.return_value = self.sample_data
        mock_add_dataframe.return_value = None
        
        # Load data
        self.run_async(self.service.load_data(
            start_time=datetime(2023, 1, 1, 9, 30),
            end_time=datetime(2023, 1, 1, 16, 0)
        ))
        
        # Process tick bars
        result = self.run_async(self.service.process_bar(
            bar_type='tick',
            ratio=20
        ))
        
        # Verify the result
        self.assertFalse(result.empty)
        self.assertIn('bar_type', result.columns)
        self.assertEqual(result['bar_type'].iloc[0], 'tick')
        self.assertIn('ratio', result.columns)
        self.assertEqual(result['ratio'].iloc[0], 20)
        
        # Verify store_bars_in_db was called with correct args
        mock_add_dataframe.assert_called_once()
    
    @patch('backend.db.models.ProcessedData.ProcessedData.add_dataframe_by_symbol')
    @patch('backend.db.models.RawData.RawData.get_price_data')
    def test_process_time_bars(self, mock_get_price_data, mock_add_dataframe):
        """Test processing time bars."""
        # Set up the mocks
        mock_get_price_data.return_value = self.sample_data
        mock_add_dataframe.return_value = None
        
        # Load data
        self.run_async(self.service.load_data(
            start_time=datetime(2023, 1, 1, 9, 30),
            end_time=datetime(2023, 1, 1, 16, 0)
        ))
        
        # Process time bars
        result = self.run_async(self.service.process_bar(
            bar_type='time',
            ratio=15
        ))
        
        # Verify the result
        self.assertFalse(result.empty)
        self.assertIn('bar_type', result.columns)
        self.assertEqual(result['bar_type'].iloc[0], 'time')
        self.assertIn('ratio', result.columns)
        self.assertEqual(result['ratio'].iloc[0], 15)
        
        # Verify store_bars_in_db was called with correct args
        mock_add_dataframe.assert_called_once()
    
    @patch('backend.db.models.ProcessedData.ProcessedData.add_dataframe_by_symbol')
    @patch('backend.db.models.RawData.RawData.get_price_data')
    def test_process_entropy_bars(self, mock_get_price_data, mock_add_dataframe):
        """Test processing entropy bars."""
        # Set up the mocks
        mock_get_price_data.return_value = self.sample_data
        mock_add_dataframe.return_value = None
        
        # Load data
        self.run_async(self.service.load_data(
            start_time=datetime(2023, 1, 1, 9, 30),
            end_time=datetime(2023, 1, 1, 16, 0)
        ))
        
        # Process entropy bars
        result = self.run_async(self.service.process_bar(
            bar_type='entropy',
            ratio=5,
            window_size=50,
            method='shannon'
        ))
        
        # Verify the result
        self.assertFalse(result.empty)
        self.assertIn('bar_type', result.columns)
        self.assertEqual(result['bar_type'].iloc[0], 'entropy')
        self.assertIn('ratio', result.columns)
        self.assertEqual(result['ratio'].iloc[0], 5)
        
        # Verify store_bars_in_db was called with correct args
        mock_add_dataframe.assert_called_once()
    
    @patch('backend.db.models.ProcessedData.ProcessedData.add_dataframe_by_symbol')
    @patch('backend.db.models.RawData.RawData.get_price_data')
    def test_process_multiple_bars(self, mock_get_price_data, mock_add_dataframe):
        """Test processing multiple bar types."""
        # Set up the mocks
        mock_get_price_data.return_value = self.sample_data
        mock_add_dataframe.return_value = None
        
        # Load data
        self.run_async(self.service.load_data(
            start_time=datetime(2023, 1, 1, 9, 30),
            end_time=datetime(2023, 1, 1, 16, 0)
        ))
        
        # Process multiple bar types
        bar_types_ratios = {
            'volume': [10, 20],
            'tick': [50],
            'time': [5, 15]
        }
        
        results = self.run_async(self.service.process_multiple_bars(bar_types_ratios))
        
        # Verify the results
        self.assertIn('volume', results)
        self.assertIn('tick', results)
        self.assertIn('time', results)
        
        self.assertIn(10, results['volume'])
        self.assertIn(20, results['volume'])
        self.assertIn(50, results['tick'])
        self.assertIn(5, results['time'])
        self.assertIn(15, results['time'])
        
        # Verify all bar types were processed and stored
        expected_calls = len(bar_types_ratios['volume']) + len(bar_types_ratios['tick']) + len(bar_types_ratios['time'])
        self.assertEqual(mock_add_dataframe.call_count, expected_calls)
    
    @patch('backend.db.models.RawData.RawData.get_price_data')
    def test_change_ticker(self, mock_get_price_data):
        """Test changing ticker."""
        # Set up the mocks
        mock_get_price_data.side_effect = [
            self.sample_data,  # First call for original ticker
            self.sample_data.copy()  # Second call for new ticker
        ]
        
        # Load data for original ticker
        self.run_async(self.service.load_data(
            start_time=datetime(2023, 1, 1, 9, 30),
            end_time=datetime(2023, 1, 1, 16, 0)
        ))
        
        # Verify original ticker
        self.assertEqual(self.service.ticker, self.ticker)
        
        # Change ticker
        new_ticker = "MSFT"
        change_result = self.run_async(self.service.change_ticker(
            new_ticker=new_ticker,
            load_data=True,
            start_time=datetime(2023, 1, 1, 9, 30),
            end_time=datetime(2023, 1, 1, 16, 0)
        ))
        
        # Verify the change
        self.assertTrue(change_result)
        self.assertEqual(self.service.ticker, new_ticker)
        
        # Verify data was loaded for new ticker
        self.assertTrue(self.run_async(self.service.has_cached_data()))
        
        # Verify get_price_data was called for the new ticker
        mock_get_price_data.assert_called_with(
            ticker=new_ticker, 
            start_date=datetime(2023, 1, 1, 9, 30),
            end_date=datetime(2023, 1, 1, 16, 0)
        )
    
    def test_is_cython_available(self):
        """Test Cython availability check."""
        # Just call the method to ensure it works
        result = self.service.is_cython_available()
        # Result could be True or False depending on environment
        self.assertIsInstance(result, bool)
    
    def test_update_params(self):
        """Test parameter updates."""
        # Original values
        original_window_size = self.service.window_size
        original_avg_window = self.service.avg_window
        original_calculation_method = self.service.calculation_method
        
        # New values
        new_window_size = 75
        new_avg_window = 150
        new_calculation_method = "tsallis"
        
        # Update params
        result = self.run_async(self.service.update_params(
            window_size=new_window_size,
            avg_window=new_avg_window,
            calculation_method=new_calculation_method
        ))
        
        # Verify the update
        self.assertTrue(result)
        self.assertEqual(self.service.window_size, new_window_size)
        self.assertEqual(self.service.avg_window, new_avg_window)
        self.assertEqual(self.service.calculation_method, new_calculation_method)
    
    @patch('backend.db.models.ProcessedData.ProcessedData.list_available_data')
    def test_get_available_processed_data(self, mock_list_available_data):
        """Test getting available processed data."""
        # Set up the mock
        mock_data = {
            self.ticker: {
                "volume": ["5", "10"],
                "tick": ["50", "100"],
                "time": ["15"]
            }
        }
        mock_list_available_data.return_value = mock_data
        
        # Call the method
        result = self.run_async(self.service.get_available_processed_data())
        
        # Verify the result
        self.assertEqual(result, mock_data[self.ticker])
        mock_list_available_data.assert_called_once_with(self.ticker)

if __name__ == '__main__':
    unittest.main() 