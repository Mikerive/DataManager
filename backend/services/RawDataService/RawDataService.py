import os
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import time
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import pandas as pd

from backend.lib.Tiingo import Tiingo
from backend.db.models.Tickers import Tickers
from backend.db.models.RawData import RawData
from backend.db.Database import Database
from backend.db.utils.db_utils import log_db_error, log_db_success


class RawDataService:
    """
    Service for integrating Tiingo data with the local database.
    
    This class fetches data from Tiingo API and stores it in the local database
    using the Tickers and RawData classes.
    """
    
    def __init__(self, use_test_db: bool = None, debug_mode: bool = False):
        """
        Initialize the Tiingo data service.
        
        Args:
            use_test_db: If True, use test database. If False, use production.
                        If None, auto-detect based on environment.
            debug_mode: If True, enable detailed API request/response logging.
        """
        self.tiingo = Tiingo.get_instance()  # Get the singleton instance
        self.logger = logging.getLogger(__name__)
        self._is_connected = False
        self._download_progress = {}
        self._download_logs = {}
        self._db = None
        self.use_test_db = use_test_db
        self.debug_mode = debug_mode
        
        # Set debug mode on Tiingo instance
        if debug_mode:
            self.tiingo.set_debug_mode(enabled=True)
        
    async def _ensure_connected(self):
        """
        Ensure database connection is established.
        This is an internal method used by other service methods.
        """
        if not self._is_connected or self._db is None or not hasattr(self._db, 'pool') or self._db.pool is None:
            try:
                self._db = Database(
                    owner_name="RawDataService",
                    use_test_db=self.use_test_db,
                    debug_mode=self.debug_mode
                )
                
                await self._db.connect()
                self._is_connected = True
                
                db_info = self._db.get_connection_info()
                self.logger.info(f"Database connection established to {db_info['host']}:{db_info['port']}/{db_info['name']}")
            except Exception as e:
                self.logger.error(f"Error connecting to database: {str(e)}")
                raise
                
    async def get_price_data(self, ticker: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Get price data for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date for filtering data
            end_date: End date for filtering data
            
        Returns:
            DataFrame containing the price data
        """
        try:
            await self._ensure_connected()
            
            # Ensure consistent uppercase ticker format
            ticker = ticker.upper()
            
            # Use the RawData class to retrieve the data
            df = await RawData.get_price_data(ticker, start_date, end_date)
            return df
        except Exception as e:
            self.logger.error(f"Error getting price data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    async def cleanup(self):
        """Cleanup database connections."""
        if self._is_connected and self._db is not None:
            try:
                await self._db.close()
                self._is_connected = False
                self._db = None
                
                # Also close model class connections
                await RawData.close_connection()
                await Tickers.close_connection()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")
    
    # Context manager support
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def sync_ticker_metadata(self, ticker: str) -> int:
        """
        Sync ticker metadata from Tiingo to the local database.
        
        Args:
            ticker: Ticker symbol (will be converted to lowercase for API, uppercase for DB)
            
        Returns:
            int: The ticker_id in the database
        """
        try:
            await self._ensure_connected()
            
            # Convert ticker to lowercase for API calls, uppercase for DB
            api_ticker = ticker.lower()
            db_ticker = ticker.upper()
            
            self.logger.info(f"Syncing metadata for {ticker}")
            
            try:
                # Get metadata from Tiingo using lowercase ticker
                metadata = self.tiingo.get_ticker_metadata(api_ticker)
                
                # Extract relevant fields
                name = metadata.get('name', db_ticker)
                exchange = metadata.get('exchange', 'UNKNOWN')
                asset_type = 'Stock'  # Default to Stock
                
                # Parse dates
                try:
                    ipo_date = datetime.fromisoformat(metadata.get('startDate', '').replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    ipo_date = None
            except Exception as e:
                # For testing purposes, create minimal metadata if Tiingo API fails
                self.logger.warning(f"Error getting metadata from Tiingo, using defaults: {str(e)}")
                name = db_ticker
                exchange = "TEST"
                asset_type = "Stock"
                ipo_date = datetime.now() - timedelta(days=365*5)  # 5 years ago
            
            # Always None for delisting date in test
            delisting_date = None
            status = 'active'
            
            # Add to database 
            ticker_id = await Tickers.add_ticker(
                ticker=db_ticker,
                name=name,
                exchange=exchange,
                asset_type=asset_type,
                ipo_date=ipo_date,
                delisting_date=delisting_date,
                status=status
            )
            
            return ticker_id
        except Exception as e:
            self.logger.error(f"Error syncing ticker metadata for {ticker}: {str(e)}")
            return None
    
    async def download_ticker_data(self, ticker: str, include_extended_hours: bool = True, full_history: bool = False) -> Dict:
        """
        Download and store historical ticker data from Tiingo.
        
        This method automatically determines the appropriate date range and handles
        month-by-month downloads for historical intraday data, as required by Tiingo's API.
        Tiingo's API returns data for complete months regardless of the day component in dates.
        
        Args:
            ticker: The ticker symbol to download data for
            include_extended_hours: Whether to include extended hours data
            full_history: Whether to retrieve the complete available history from IPO date
        
        Returns:
            Dictionary with information about the download result
        """
        try:
            download_id = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.logger.info(f"Downloading data for {ticker} (full_history={full_history})")
            
            # Convert ticker to lowercase for API, uppercase for DB
            api_ticker = ticker.lower()
            db_ticker = ticker.upper()
            
            # Get the current time in UTC
            current_time = datetime.now(timezone.utc)
            
            # Ensure ticker exists in the database
            ticker_record = await Tickers.get_ticker(db_ticker)
            if not ticker_record:
                self.logger.info(f"Ticker {ticker} not found, syncing metadata")
                ticker_id = await self.sync_ticker_metadata(api_ticker)
                if not ticker_id:
                    return {"success": False, "ticker": ticker, "error": "Failed to sync ticker metadata"}
                ticker_record = await Tickers.get_ticker(db_ticker)
            
            # Determine start date
            if full_history:
                # Use the IPO date from the ticker record
                ipo_date = ticker_record.get('ipo_date')
                if not ipo_date:
                    # If no IPO date in database, try to get listing date from Tiingo
                    listing_date = self.tiingo.get_listing_date(api_ticker)
                    if listing_date:
                        ipo_date = listing_date
                        self.logger.info(f"Using listing date from Tiingo API: {ipo_date.isoformat()}")
                    else:
                        # Default to 5 years if no listing date available
                        ipo_date = current_time - timedelta(days=365*5)
                        self.logger.info(f"No IPO date available, using 5 years ago as fallback: {ipo_date.isoformat()}")
                
                self.logger.info(f"Fetching full history for {ticker} from {ipo_date.isoformat()}")
                start_date = ipo_date
            else:
                # Use a reasonable default period (30 days)
                start_date = current_time - timedelta(days=30)
                self.logger.info(f"Fetching recent history for {ticker} from {start_date.isoformat()}")
            
            # Determine the month boundaries
            start_year = start_date.year
            start_month = start_date.month
            end_year = current_time.year
            end_month = current_time.month
            
            # Calculate total months to process
            total_months = (end_year - start_year) * 12 + (end_month - start_month) + 1
            self.logger.info(f"Downloading {total_months} months of data for {ticker}")
            
            # Initialize an empty DataFrame to store all the data
            all_data = pd.DataFrame()
            total_rows_added = 0
            months_processed = 0
            
            # Process each month from start to end
            current_year = start_year
            current_month = start_month
            
            while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
                # Create a date in the current month
                if current_month == 12:
                    next_year = current_year + 1
                    next_month = 1
                else:
                    next_year = current_year
                    next_month = current_month + 1
                
                self.logger.info(f"Downloading {ticker} month {months_processed+1}/{total_months}: {current_year}-{current_month:02d}")
                
                # Create a date to represent this month (day doesn't matter, Tiingo uses only year/month)
                current_date = datetime(current_year, current_month, 1, tzinfo=timezone.utc)
                
                # Get data from Tiingo for this month (just need a date in the month)
                df_chunk = self.tiingo.get_historical_intraday(
                    ticker=api_ticker,
                    start_date=current_date,
                    frequency="1min",
                    include_extended_hours=include_extended_hours
                )
                
                if df_chunk is not None and not df_chunk.empty:
                    # Convert and prepare data for database
                    df_chunk_for_db = df_chunk.reset_index().rename(columns={
                        'date': 'timestamp',
                    })
                    df_chunk_for_db['ticker'] = db_ticker
                    
                    # Store this month's data in the database
                    chunk_result = await RawData.add_dataframe(df_chunk_for_db)
                    
                    if chunk_result["success"]:
                        rows_added = chunk_result.get('rows_added', 0)
                        total_rows_added += rows_added
                        self.logger.info(f"Successfully added {rows_added} rows for {ticker} ({current_year}-{current_month:02d})")
                        
                        # Append to our all_data DataFrame for result reporting
                        if all_data.empty:
                            all_data = df_chunk.copy()
                        else:
                            all_data = pd.concat([all_data, df_chunk])
                    else:
                        error_msg = chunk_result.get("error", "Unknown database error")
                        self.logger.error(f"Error storing {ticker} data for {current_year}-{current_month:02d}: {error_msg}")
                else:
                    self.logger.warning(f"No data returned for {ticker} for month {current_year}-{current_month:02d}")
                
                # Move to the next month
                months_processed += 1
                current_month = next_month
                current_year = next_year
            
            # Return final results
            if total_rows_added > 0:
                return {
                    "success": True,
                    "ticker": ticker,
                    "rows_added": total_rows_added,
                    "months_processed": months_processed,
                    "start_date": all_data.index.min().isoformat() if not all_data.empty else start_date.isoformat(),
                    "end_date": all_data.index.max().isoformat() if not all_data.empty else current_time.isoformat()
                }
            else:
                return {
                    "success": False, 
                    "ticker": ticker,
                    "error": "No data could be retrieved and stored",
                    "months_processed": months_processed
                }
                
        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return {"success": False, "ticker": ticker, "error": str(e)}
    
    async def update_recent_data(self, ticker: str, include_extended_hours: bool = True) -> Dict:
        """
        Update recent data for a ticker by comparing the most recent database entry with current time.
        
        This method will:
        1. Directly query the TimescaleDB for the most recent timestamp for the ticker
        2. Download data for the months between that timestamp and current time
        3. Store the new data in the database
        
        Note: Tiingo's API works on a month basis for intraday data, so we request whole months
        even if we only need data from partway through the month.
        
        Args:
            ticker: The ticker symbol to update
            include_extended_hours: Whether to include extended hours data
            
        Returns:
            Dictionary with information about the update operation
        """
        try:
            await self._ensure_connected()
            
            # Convert ticker format
            db_ticker = ticker.upper()
            api_ticker = ticker.lower()
            
            # Get the current time in UTC
            current_time = datetime.now(timezone.utc)
            current_year = current_time.year
            current_month = current_time.month
            
            # Directly query the database for the most recent timestamp
            query = f"""
            SELECT MAX(timestamp) as latest_time
            FROM raw_data_{api_ticker}
            """
            
            try:
                # Execute the query directly
                latest_record = await self._db.fetchval(query)
                
                if latest_record:
                    # We found a timestamp - use its month as our starting point
                    # If we're already in the current month, we only need to fetch the current month
                    start_date = latest_record
                    start_year = start_date.year
                    start_month = start_date.month
                    
                    # If the latest data is from the current month, we just need to update this month
                    if start_year == current_year and start_month == current_month:
                        self.logger.info(f"Latest data for {ticker} is from the current month. Updating current month.")
                        months_to_update = [(current_year, current_month)]
                    else:
                        # Calculate all months between the latest data and now
                        months_to_update = []
                        update_year = start_year
                        update_month = start_month
                        
                        # Always update at least the month of the latest record (to get any missing days)
                        # and continue through the current month
                        while (update_year < current_year) or (update_year == current_year and update_month <= current_month):
                            months_to_update.append((update_year, update_month))
                            
                            if update_month == 12:
                                update_year += 1
                                update_month = 1
                            else:
                                update_month += 1
                        
                        self.logger.info(f"Found latest data for {ticker} from {start_date.isoformat()}. Updating {len(months_to_update)} months.")
                else:
                    # No data found - check if we need to sync ticker metadata first
                    ticker_record = await Tickers.get_ticker(db_ticker)
                    if not ticker_record:
                        self.logger.info(f"Ticker {ticker} not found, syncing metadata")
                        await self.sync_ticker_metadata(api_ticker)
                    
                    # Use Tiingo's API to determine the appropriate start date
                    self.logger.info(f"No existing data found for {ticker}. Using Tiingo to determine start date.")
                    listing_date = self.tiingo.get_listing_date(api_ticker)
                    
                    if listing_date:
                        # Use listing date or 30 days ago, whichever is more recent
                        thirty_days_ago = current_time - timedelta(days=30)
                        start_date = max(listing_date, thirty_days_ago)
                        self.logger.info(f"Using start date from listing information: {start_date.isoformat()}")
                    else:
                        # Default to 30 days of data if we can't determine listing date
                        start_date = current_time - timedelta(days=30)
                        self.logger.info(f"Using default 30-day lookback period: {start_date.isoformat()}")
                    
                    # Set up months to update - just use the download_ticker_data method
                    # which handles month-by-month downloading
                    download_result = await self.download_ticker_data(
                        ticker=ticker,
                        include_extended_hours=include_extended_hours,
                        full_history=False  # Just get recent data
                    )
                    return download_result
            except Exception as db_error:
                # Table may not exist yet or other DB error
                self.logger.warning(f"Error querying latest timestamp for {ticker}: {str(db_error)}")
                
                # When table doesn't exist, check ticker metadata first
                ticker_record = await Tickers.get_ticker(db_ticker)
                if not ticker_record:
                    self.logger.info(f"Ticker {ticker} not found, syncing metadata")
                    await self.sync_ticker_metadata(api_ticker)
                
                # Determine a reasonable start date based on market data availability
                self.logger.info(f"Table for {ticker} may not exist yet. Using Tiingo to determine start date.")
                listing_date = self.tiingo.get_listing_date(api_ticker)
                
                if listing_date:
                    # Use listing date or 30 days ago, whichever is more recent
                    thirty_days_ago = current_time - timedelta(days=30)
                    start_date = max(listing_date, thirty_days_ago)
                    self.logger.info(f"Using start date from listing information: {start_date.isoformat()}")
                else:
                    # Default to 30 days of data if we can't determine listing date
                    start_date = current_time - timedelta(days=30)
                    self.logger.info(f"Using default 30-day lookback period: {start_date.isoformat()}")
                
                # Use the download_ticker_data method which handles month-by-month downloading
                download_result = await self.download_ticker_data(
                    ticker=ticker,
                    include_extended_hours=include_extended_hours,
                    full_history=False  # Just get recent data
                )
                return download_result
            
            # If we got this far, we have specific months to update
            # Process each month individually
            total_rows_added = 0
            all_data = pd.DataFrame()
            months_processed = 0
            
            for year, month in months_to_update:
                self.logger.info(f"Updating {ticker} for month {year}-{month:02d}")
                
                # Create a date object for this month (day doesn't matter for Tiingo)
                month_date = datetime(year, month, 1, tzinfo=timezone.utc)
                
                # Get data from Tiingo for this month
                df = self.tiingo.get_historical_intraday(
                    ticker=api_ticker,
                    start_date=month_date,
                    frequency="1min",
                    include_extended_hours=include_extended_hours
                )
                
                # Check if we received data
                if df is None or df.empty:
                    self.logger.warning(f"No data available for {ticker} for month {year}-{month:02d}")
                    months_processed += 1
                    continue
                
                # Process Tiingo data format
                df_for_db = df.reset_index().rename(columns={
                    'date': 'timestamp',
                })
                df_for_db['ticker'] = db_ticker
                
                # Store in the database
                result = await RawData.add_dataframe(df_for_db)
                months_processed += 1
                
                if result["success"]:
                    rows_added = result.get('rows_added', 0)
                    total_rows_added += rows_added
                    self.logger.info(f"Successfully added {rows_added} rows for {ticker} ({year}-{month:02d})")
                    
                    # Append to our all_data DataFrame for result reporting
                    if all_data.empty:
                        all_data = df.copy()
                    else:
                        all_data = pd.concat([all_data, df])
                else:
                    error_msg = result.get("error", "Unknown database error")
                    self.logger.error(f"Error storing {ticker} data for {year}-{month:02d}: {error_msg}")
            
            # Return results
            if total_rows_added > 0:
                return {
                    "success": True,
                    "ticker": ticker,
                    "rows_added": total_rows_added,
                    "months_processed": months_processed,
                    "start_date": all_data.index.min().isoformat() if not all_data.empty else start_date.isoformat(),
                    "end_date": all_data.index.max().isoformat() if not all_data.empty else current_time.isoformat()
                }
            else:
                return {
                    "success": True,  # Still a success case - maybe no new data for the period
                    "ticker": ticker,
                    "message": "No new data available or added", 
                    "rows_added": 0,
                    "months_processed": months_processed
                }
            
        except Exception as e:
            self.logger.error(f"Error updating recent data for {ticker}: {str(e)}")
            return {"success": False, "ticker": ticker, "error": str(e)}

    async def bulk_update_tickers(self, tickers: List[str], include_extended_hours: bool = True) -> Dict:
        """
        Update recent data for multiple tickers in sequence.
        
        Args:
            tickers: List of ticker symbols to update
            include_extended_hours: Whether to include extended hours data
            
        Returns:
            Dictionary with information about the update operation
        """
        results = {
            "tickers_processed": 0,
            "tickers_successful": 0,
            "tickers_failed": 0,
            "details": {}
        }
        
        for ticker in tickers:
            result = await self.update_recent_data(ticker, include_extended_hours)
            results["tickers_processed"] += 1
            
            if result["success"]:
                results["tickers_successful"] += 1
            else:
                results["tickers_failed"] += 1
                
            results["details"][ticker] = result
            
        return results

    def get_debug_info(self) -> Dict:
        """
        Get debug information about the service and Tiingo API client.
        
        Returns:
            Dictionary with debug information
        """
        # Mask API key for security
        masked_key = "********"
        if self.tiingo.api_key and len(self.tiingo.api_key) > 8:
            masked_key = self.tiingo.api_key[:4] + "****" + self.tiingo.api_key[-4:]
            
        return {
            "debug_mode": self.debug_mode,
            "is_connected": self._is_connected,
            "use_test_db": self.use_test_db,
            "tiingo_base_url": self.tiingo.base_url,
            "tiingo_iex_url": self.tiingo.iex_url,
            "rate_limiter": {
                "hourly_limit": self.tiingo.rate_limiter.hourly_limit,
                "daily_limit": self.tiingo.rate_limiter.daily_limit,
                "hourly_requests": len(self.tiingo.rate_limiter.hourly_timestamps),
                "daily_requests": len(self.tiingo.rate_limiter.daily_timestamps)
            },
            "api_key_masked": masked_key
        }
        
    def set_debug_mode(self, enabled=True):
        """
        Enable or disable debug mode.
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        self.tiingo.set_debug_mode(enabled)
        self.logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        
