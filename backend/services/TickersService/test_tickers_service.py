import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import unittest
import asyncio
import pandas as pd
from datetime import datetime
import logging

from backend.services.TickersService import TickersService
from backend.db.models.Tickers import Tickers

# Configure logging
logging.basicConfig(level=logging.INFO)


class TestTickersService(unittest.TestCase):
    """Test suite for the TickersService class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.service = TickersService(api_key="demo")  # Using demo API key for testing
        
        # Ensure database connection and table
        self.run_async(self._setup_db())
        
    async def _setup_db(self):
        """Set up the database for tests."""
        await Tickers.create_tickers_table()
        
    def tearDown(self):
        """Clean up after each test."""
        self.run_async(self._cleanup_db())
        
    async def _cleanup_db(self):
        """Clean up database resources."""
        await Tickers.close_connection()

    def run_async(self, coro):
        """Helper method to run async methods in sync tests."""
        return asyncio.get_event_loop().run_until_complete(coro)
    
    def test_fetch_and_store_active_tickers(self):
        """Test fetching and storing active tickers using real AlphaVantage API."""
        # Call the method
        fetched, stored = self.run_async(self.service.fetch_and_store_active_tickers())
        
        # Verify types (don't assert exact counts as they may vary)
        self.assertIsInstance(fetched, int)
        self.assertIsInstance(stored, int)
        
        # Log the results
        print(f"Fetched {fetched} active tickers from AlphaVantage API")
        print(f"Stored {stored} active tickers in the database")
    
    def test_fetch_and_store_delisted_tickers(self):
        """Test fetching and storing delisted tickers using real AlphaVantage API."""
        # Call the method with a specific date
        test_date = "2020-01-01"
        fetched, stored = self.run_async(self.service.fetch_and_store_delisted_tickers(date=test_date))
        
        # Verify types (don't assert exact counts as they may vary)
        self.assertIsInstance(fetched, int)
        self.assertIsInstance(stored, int)
        
        # Log the results
        print(f"Fetched {fetched} delisted tickers from AlphaVantage API for date {test_date}")
        print(f"Stored {stored} delisted tickers in the database")
    
    def test_fetch_and_store_all_tickers(self):
        """Test fetching and storing all tickers using real AlphaVantage API."""
        # Call the method
        fetched, stored = self.run_async(self.service.fetch_and_store_all_tickers())
        
        # Verify types (don't assert exact counts as they may vary)
        self.assertIsInstance(fetched, int)
        self.assertIsInstance(stored, int)
        
        # Log the results
        print(f"Fetched {fetched} total tickers from AlphaVantage API")
        print(f"Stored {stored} total tickers in the database")
        
        # Verify we can get all tickers from database
        all_tickers = self.run_async(Tickers.get_all_tickers())
        self.assertIsInstance(all_tickers, list)
    
    def test_update_ticker_status(self):
        """Test updating ticker status using real Tickers class."""
        # First add a ticker to update
        ticker_id = self.run_async(Tickers.add_ticker(
            ticker="TEST",
            name="Test Company",
            exchange="NYSE",
            asset_type="Stock",
            ipo_date=datetime(2000, 1, 1),
            status="active"
        ))
        
        # Test updating to delisted with a date
        ticker = "TEST"
        status = "delisted"
        date = "2023-01-01"
        
        # Call the method
        success = self.run_async(self.service.update_ticker_status(ticker, status, date))
        
        # Verify results
        self.assertTrue(success)
        
        # Verify the update happened
        updated_ticker = self.run_async(Tickers.get_ticker(ticker))
        if updated_ticker:
            self.assertEqual(updated_ticker["status"], "delisted")
            self.assertIsNotNone(updated_ticker["delisting_date"])
    
    def test_get_earnings_calendar(self):
        """Test fetching earnings calendar data using real AlphaVantage API."""
        # Call the method with a specific symbol
        earnings = self.run_async(self.service.get_earnings_calendar(symbol="AAPL", horizon="6month"))
        
        # Verify we got some results, but don't assert specific content
        self.assertIsInstance(earnings, list)
        
        # Log the results
        print(f"Fetched {len(earnings)} earnings entries for AAPL")
        if earnings:
            print(f"First earnings entry keys: {list(earnings[0].keys())}")
        
        # Test with no symbol (all companies)
        earnings_all = self.run_async(self.service.get_earnings_calendar())
        self.assertIsInstance(earnings_all, list)
        print(f"Fetched {len(earnings_all)} earnings entries for all companies")
    
    def test_search_tickers(self):
        """Test searching for tickers using real Tickers class."""
        # First add some tickers to search for
        self.run_async(Tickers.add_ticker(
            ticker="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            asset_type="Stock",
            ipo_date=datetime(1980, 12, 12),
            status="active"
        ))
        
        self.run_async(Tickers.add_ticker(
            ticker="AMZN",
            name="Amazon.com Inc.",
            exchange="NASDAQ",
            asset_type="Stock",
            ipo_date=datetime(1997, 5, 15),
            status="active"
        ))
        
        # Call the method
        results = self.run_async(self.service.search_tickers(
            search_term="Apple", 
            exchange="NASDAQ", 
            asset_type="Stock", 
            status="active"
        ))
        
        # Verify types
        self.assertIsInstance(results, list)
        
        # Verify results (might be empty if filtering removed all)
        if results:
            self.assertIsInstance(results[0], dict)
            self.assertIn("ticker", results[0])
            self.assertIn("name", results[0])
    
    def test_prepare_tickers_dataframe(self):
        """Test the DataFrame preparation method."""
        # Create test data
        test_data = pd.DataFrame([
            {
                "symbol": "aapl",  # Lowercase to test uppercase conversion
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1980-12-12",
                "delistingDate": None
            }
        ])
        
        # Test active status
        result_active = self.service._prepare_tickers_dataframe(test_data, "active")
        
        # Verify column renaming
        self.assertIn("ticker", result_active.columns)
        self.assertIn("asset_type", result_active.columns)
        self.assertIn("ipo_date", result_active.columns)
        
        # Verify uppercase conversion
        self.assertEqual(result_active["ticker"].iloc[0], "AAPL")
        
        # Verify status setting
        self.assertEqual(result_active["status"].iloc[0], "active")
        
        # Verify date conversion
        self.assertIsInstance(result_active["ipo_date"].iloc[0], pd.Timestamp)
        
        # Test delisted status
        result_delisted = self.service._prepare_tickers_dataframe(test_data, "delisted")
        
        # Verify status setting
        self.assertEqual(result_delisted["status"].iloc[0], "delisted")
        
        # Verify delisting date is set for delisted status when missing
        self.assertIsNotNone(result_delisted["delisting_date"].iloc[0])
        self.assertIsInstance(result_delisted["delisting_date"].iloc[0], pd.Timestamp)

    def test_prepare_tickers_dataframe_with_missing_values(self):
        """Test the DataFrame preparation method with missing values."""
        # Create test data with some missing values
        test_data = pd.DataFrame([
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1980-12-12",
                "delistingDate": None,
                "status": "Active"
            },
            {
                "symbol": "MSFT",
                "name": None,  # Missing name
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1986-03-13",
                "delistingDate": None,
                "status": "Active"
            },
            {
                "symbol": None,  # Missing symbol
                "name": "Missing Symbol Corp",
                "exchange": "NYSE",
                "assetType": "Stock",
                "ipoDate": "2000-01-01",
                "delistingDate": None,
                "status": "Active"
            }
        ])
        
        # Process the data
        result_df = self.service._prepare_tickers_dataframe(test_data, "active")
        
        # Verify rows with missing required fields are dropped
        self.assertEqual(len(result_df), 1)  # Only the complete row should remain
        self.assertEqual(result_df["ticker"].iloc[0], "AAPL")
        
    def test_prepare_tickers_dataframe_with_different_status_formats(self):
        """Test the DataFrame preparation with different status value formats."""
        # Create test data with different status formats
        test_data = pd.DataFrame([
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1980-12-12",
                "delistingDate": None,
                "status": "Active"  # Title case
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corp",
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1986-03-13", 
                "delistingDate": None,
                "status": "ACTIVE"  # Uppercase
            },
            {
                "symbol": "YHOO",
                "name": "Yahoo Inc",
                "exchange": "NASDAQ",
                "assetType": "Stock",
                "ipoDate": "1996-04-12",
                "delistingDate": "2017-06-16",
                "status": "Delisted"  # Title case delisted
            }
        ])
        
        # Process the data
        result_df = self.service._prepare_tickers_dataframe(test_data)
        
        # Verify all status values are converted to lowercase
        self.assertEqual(result_df["status"].iloc[0], "active")
        self.assertEqual(result_df["status"].iloc[1], "active")
        self.assertEqual(result_df["status"].iloc[2], "delisted")
        
        # Verify delisted row has delisting date
        self.assertIsNotNone(result_df["delisting_date"].iloc[2])

    async def test_close(self):
        """Test closing resources."""
        await self.service.close()
        # No assertion needed, just checking it doesn't throw an exception


if __name__ == "__main__":
    unittest.main() 