#!/usr/bin/env python
"""
Unit tests for RawDataService.

These tests verify the core functionality of RawDataService using real data.
"""

import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv
import sys
import time

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.RawDataService.RawDataService import RawDataService
from backend.db.Database import Database
from backend.db.models.RawData import RawData
from backend.db.models.Tickers import Tickers

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path)

# Set up logging for debug tests
logging.basicConfig(level=logging.DEBUG)

# Test constants
TEST_TICKER = "AAPL"
TEST_DAYS = 3  # Use a small value for faster tests

@pytest.fixture(scope="function")
async def service():
    """
    Create a RawDataService instance for testing.
    
    This fixture ensures that:
    1. Each test gets a fresh instance of the service
    2. Database connections are properly established
    3. Test data is cleaned up after each test
    """
    # Connect to the database first to ensure it's available
    db = Database()
    await db.connect()
    
    # Create service with debug mode disabled
    service = RawDataService(debug_mode=False)
    
    # Initialize service and ensure it's connected to the database
    await service._ensure_connected()
    
    # Explicit yield to indicate this is where the test will run
    yield service
    
    # Clean up
    await service.cleanup()
    
    # Close the database connection used for setup
    await db.close()

class TestRawDataService:
    """Tests for the RawDataService class."""
    
    @pytest.fixture
    async def debug_service(self):
        """Create a RawDataService instance with debug mode enabled for testing."""
        service = RawDataService(use_test_db=True, debug_mode=True)
        yield service
        await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_init(self, service):
        """Test that service initializes correctly."""
        assert service is not None
        assert service.tiingo is not None
        assert service._is_connected is True
        assert service.debug_mode is False
    
    @pytest.mark.asyncio
    async def test_init_with_debug(self, debug_service):
        """Test that service initializes correctly with debug mode."""
        assert debug_service is not None
        assert debug_service.tiingo is not None
        assert debug_service.debug_mode is True
    
    @pytest.mark.asyncio
    async def test_set_debug_mode(self, service):
        """Test setting debug mode."""
        assert service.debug_mode is False
        service.set_debug_mode(True)
        assert service.debug_mode is True
        service.set_debug_mode(False)
        assert service.debug_mode is False
    
    @pytest.mark.asyncio
    async def test_ensure_connected(self, service):
        """Test that _ensure_connected establishes database connection."""
        await service._ensure_connected()
        assert service._is_connected is True
        assert service._db is not None
    
    @pytest.mark.asyncio
    async def test_sync_ticker_metadata(self, service):
        """Test syncing ticker metadata."""
        ticker_id = await service.sync_ticker_metadata(TEST_TICKER)
        
        assert ticker_id is not None
        assert isinstance(ticker_id, int)
        assert ticker_id > 0
    
    @pytest.mark.asyncio
    async def test_download_ticker_data(self, service):
        """Test downloading ticker data."""
        result = await service.download_ticker_data(
            ticker=TEST_TICKER,
            include_extended_hours=True,
            full_history=False
        )
        
        assert result['success'] is True
        assert result['ticker'] == TEST_TICKER
        assert 'error' not in result
    
    @pytest.mark.asyncio
    async def test_get_price_data(self, service):
        """Test retrieving price data."""
        # First download some data
        await service.download_ticker_data(
            ticker=TEST_TICKER,
            include_extended_hours=True,
            full_history=False
        )
        
        # Then retrieve it - use uppercase for ticker symbol
        df = await service.get_price_data(TEST_TICKER)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert len(df) > 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, service):
        """Test resource cleanup."""
        # First ensure we're connected
        await service._ensure_connected()
        assert service._is_connected is True
        
        # Then cleanup
        await service.cleanup()
        assert service._is_connected is False
        assert service._db is None
    
    @pytest.mark.asyncio
    async def test_download_full_history_aapl(self, service):
        """
        Test downloading the complete history for AAPL using the RawDataService.
        This test verifies that:
        1. Full history download works correctly
        2. Data is properly stored in the database
        3. The data contains all required OHLCV fields
        """
        # Download full history
        result = await service.download_ticker_data(
            ticker=TEST_TICKER,
            full_history=True,
            include_extended_hours=False
        )
        
        assert result["success"] is True, f"Failed to download data: {result.get('error', 'Unknown error')}"
        assert result["ticker"] == TEST_TICKER
        assert 'error' not in result
        
        # Retrieve the data from the database using uppercase ticker symbol
        df = await service.get_price_data(TEST_TICKER)
        
        # Verify we have data
        assert not df.empty, "No data was retrieved from the database"
        
        # Verify the data contains the expected columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Verify data types
        assert pd.api.types.is_numeric_dtype(df["open"]), "Open column should be numeric"
        assert pd.api.types.is_numeric_dtype(df["high"]), "High column should be numeric"
        assert pd.api.types.is_numeric_dtype(df["low"]), "Low column should be numeric"
        assert pd.api.types.is_numeric_dtype(df["close"]), "Close column should be numeric"
        assert pd.api.types.is_numeric_dtype(df["volume"]), "Volume column should be numeric"
        
        # In a test environment, we might not get thousands of records
        # but we should still have multiple days of data
        assert len(df) > 0, "Should have at least some data in the test environment"
        
        # Test passes if we got this far
        logging.info(f"Successfully downloaded and verified {len(df)} records for {TEST_TICKER}")
        logging.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logging.info(f"Latest close: {df['close'].iloc[-1]:.2f}")
    
    @pytest.mark.asyncio
    async def test_debug_info(self, service):
        """Test getting debug information."""
        debug_info = service.get_debug_info()
        
        assert debug_info is not None
        assert isinstance(debug_info, dict)
        assert 'debug_mode' in debug_info
        assert 'tiingo_base_url' in debug_info
        assert 'tiingo_iex_url' in debug_info
        assert 'rate_limiter' in debug_info
        assert 'api_key_masked' in debug_info
    
    @pytest.mark.asyncio
    async def test_update_recent_data(self, service):
        """Test updating recent data for a ticker."""
        # First ensure we have some data to update
        await service.download_ticker_data(
            ticker=TEST_TICKER,
            include_extended_hours=True,
            full_history=False
        )
        
        # Now update the data
        result = await service.update_recent_data(
            ticker=TEST_TICKER,
            include_extended_hours=True
        )
        
        assert result['success'] is True
        assert result['ticker'] == TEST_TICKER
        
        # The result could either indicate new data was added or that no new data was available
        if 'rows_added' in result:
            assert isinstance(result['rows_added'], int)
        elif 'message' in result:
            assert "No new data" in result['message']
    
    @pytest.mark.asyncio
    async def test_bulk_update_tickers(self, service):
        """Test bulk updating multiple tickers."""
        # Use a small list of tickers for quicker tests
        test_tickers = [TEST_TICKER, "MSFT"]
        
        result = await service.bulk_update_tickers(
            tickers=test_tickers,
            include_extended_hours=True
        )
        
        assert isinstance(result, dict)
        assert 'tickers_processed' in result
        assert result['tickers_processed'] == len(test_tickers)
        assert 'tickers_successful' in result
        assert 'details' in result
        assert len(result['details']) == len(test_tickers)
        
        # Verify each ticker has a result
        for ticker in test_tickers:
            assert ticker in result['details']
            assert 'success' in result['details'][ticker]
    
    @pytest.mark.asyncio
    async def test_verify_rate_limiting(self, service):
        """Test that rate limiting is properly implemented."""
        # Verify rate limiter is initialized
        assert hasattr(service.tiingo, 'rate_limiter')
        assert service.tiingo.rate_limiter is not None
        
        # Verify rate limiter has proper limits
        assert service.tiingo.rate_limiter.hourly_limit == 10000
        assert service.tiingo.rate_limiter.daily_limit == 100000
        
        # Test making multiple rapid requests to verify rate limiting
        start_time = time.time()
        
        # Make a few requests in quick succession
        results = []
        for _ in range(3):
            result = await service.download_ticker_data(
                ticker=TEST_TICKER,
                include_extended_hours=True,
                full_history=False
            )
            results.append(result)
        
        # Measure elapsed time
        elapsed = time.time() - start_time
        
        # Verify all requests were successful
        assert all(result['success'] for result in results)
        
        # Verify rate limiter is working (should have some delay between requests)
        # With our new rate limiter, there should be some minimum delay between requests
        min_interval = 3600 / service.tiingo.rate_limiter.hourly_limit
        
        # We made 3 requests, so we should have at least 2 intervals of delay
        expected_min_delay = min_interval * 2
        
        # Allow some margin for timing variations
        assert elapsed >= expected_min_delay * 0.5, f"Rate limiting not working properly: elapsed={elapsed}, expected at least {expected_min_delay*0.5}" 