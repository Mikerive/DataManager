import os
import asyncio
from datetime import datetime, timedelta
import logging
import time
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import pandas as pd

from ...lib.Tiingo import Tiingo
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
            self.set_debug_mode(True)
        
    def set_debug_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable debug mode for API request/response logging.
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        # Propagate to Tiingo instance
        self.tiingo.set_debug_mode(enabled)
        self.logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        
    async def _ensure_connected(self):
        """
        Ensure database connection is established.
        This is an internal method used by other service methods.
        """
        # Check both the flag and whether the pool is actually valid
        if not self._is_connected or self._db is None or not hasattr(self._db, 'pool') or self._db.pool is None:
            try:
                # Create a new Database instance with test db setting if specified
                self._db = Database(
                    owner_name="RawDataService",
                    use_test_db=self.use_test_db,
                    debug_mode=self.debug_mode  # Pass debug mode to Database
                )
                
                # Let the Database class handle the connection and retries
                await self._db.connect()
                self._is_connected = True
                
                # Log which database we're using
                db_info = self._db.get_connection_info()
                self.logger.info(f"Database connection established to {db_info['host']}:{db_info['port']}/{db_info['name']}")
                
                if self.debug_mode:
                    self.logger.debug("Database connection details:")
                    for key, value in db_info.items():
                        self.logger.debug(f"  {key}: {value}")
            except Exception as e:
                self.logger.error(f"Error connecting to database: {str(e)}")
                if self.debug_mode:
                    self.logger.debug(f"Database connection error details: {str(e)}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                raise
                
    async def get_database_info(self):
        """
        Get information about available tables in the database.
        Automatically handles database connection.
        
        Returns:
            Tuple containing (all_tables_info, raw_data_tables, processed_tables)
        """
        try:
            start_time = time.time()
            
            await self._ensure_connected()
            
            # Get all tables from the database using a direct query
            query = """
            SELECT tablename 
            FROM pg_catalog.pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename;
            """
            
            tables = await self._db.fetch(query)
                
            tables_info = {}
            for table in tables:
                table_name = table['tablename']
                # Get row count for each table
                try:
                    count_query = f"SELECT COUNT(*) FROM {table_name};"
                    count = await self._db.fetchval(count_query)
                    
                    has_price_data = False
                    # Check if it's a price data table (has OHLCV columns)
                    if table_name.startswith('raw_data_') or table_name.startswith('processed_'):
                        schema_query = f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = '{table_name}'
                        """
                        columns = await self._db.fetch(schema_query)
                        
                        column_names = [col['column_name'] for col in columns]
                        required_columns = ['open', 'high', 'low', 'close']
                        has_price_data = all(col in column_names for col in required_columns)
                    
                    tables_info[table_name] = {
                        'row_count': count,
                        'has_price_data': has_price_data
                    }
                except Exception as e:
                    self.logger.error(f"Error getting info for table {table_name}: {str(e)}")
                    tables_info[table_name] = {
                        'row_count': 0,
                        'has_price_data': False
                    }
            
            # Filter for raw data tables
            raw_data_tables = [t for t in tables_info.keys() if t.startswith('raw_data_')]
            
            # Filter for processed data tables
            processed_tables = [t for t in tables_info.keys() if t.startswith('processed_')]
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Get database info", duration_ms, self.logger)
            
            return tables_info, raw_data_tables, processed_tables
        except Exception as e:
            log_db_error("Get database info", e, self.logger)
            return {}, [], []
        
    async def get_price_data(self, table_name: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Get price data for a specific ticker or table.
        Automatically handles database connection.
        
        Args:
            table_name: Name of the table to query (raw_data_TICKER format)
            start_date: Start date for filtering data
            end_date: End date for filtering data
            
        Returns:
            DataFrame containing the price data
        """
        try:
            start_time = time.time()
            
            await self._ensure_connected()
            
            self.logger.info(f"Retrieving price data from {table_name}, {start_date} to {end_date}")
            
            # Extract ticker from table_name and ensure consistent case
            if table_name.startswith('raw_data_'):
                ticker = table_name[len('raw_data_'):].upper()
            else:
                ticker = table_name.upper()
                
            # Use the RawData class to retrieve the data
            df = await RawData.get_price_data(ticker, start_date, end_date)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Get price data for {table_name}", duration_ms, self.logger)
            
            return df
        except Exception as e:
            log_db_error(f"Get price data for {table_name}", e, self.logger)
            return pd.DataFrame()
    
    async def cleanup(self):
        """
        Cleanup resources, including database connections.
        Should be called when the service is no longer needed.
        """
        start_time = time.time()
        
        if self._is_connected and self._db is not None:
            try:
                await self._db.close()
                self._is_connected = False
                self._db = None
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error disconnecting from database: {str(e)}")
                # Don't raise here, as this is cleanup
        
        # Close RawData and Tickers class connections
        try:
            await RawData.close_connection()
            await Tickers.close_connection()
        except Exception as e:
            self.logger.error(f"Error closing class connections: {str(e)}")
            # Don't raise here, as this is cleanup
        
        # Log the cleanup time
        duration_ms = (time.time() - start_time) * 1000
        self.logger.debug(f"Resource cleanup completed in {duration_ms:.2f}ms")
    
    # Ensure proper cleanup with context manager support
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    # Regular methods that depend on database connection
    async def sync_ticker_metadata(self, ticker: str, debug_mode: bool = None) -> int:
        """
        Sync ticker metadata from Tiingo to the local database.
        Automatically handles database connection.
        
        Args:
            ticker: Ticker symbol (will be converted to lowercase for API, uppercase for DB)
            debug_mode: Override the service's debug mode setting for this call only
            
        Returns:
            int: The ticker_id in the database
        """
        # If debug_mode is provided, temporarily set it for this call
        previous_debug_mode = None
        if debug_mode is not None and debug_mode != self.debug_mode:
            previous_debug_mode = self.debug_mode
            self.set_debug_mode(debug_mode)
            
        try:
            start_time = time.time()
            
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
            
            # Add to database using class method with uppercase ticker
            ticker_id = await Tickers.add_ticker(
                ticker=db_ticker,
                name=name,
                exchange=exchange,
                asset_type=asset_type,
                ipo_date=ipo_date,
                delisting_date=delisting_date,
                status=status
            )
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Sync ticker metadata for {ticker}", duration_ms, self.logger)
            
            return ticker_id
        except Exception as e:
            log_db_error(f"Sync ticker metadata for {ticker}", e, self.logger)
            return None
        finally:
            # Restore previous debug mode if it was temporarily changed
            if previous_debug_mode is not None:
                self.set_debug_mode(previous_debug_mode)
    
    async def download_ticker_data(self, ticker, days_back=7, include_extended_hours=True, full_history=False, download_id=None, debug_mode=None):
        """
        Download and store historical ticker data from Tiingo.
        
        Args:
            ticker (str): The ticker symbol to download data for
            days_back (int, optional): Number of days to look back. Defaults to 7.
            include_extended_hours (bool, optional): Whether to include extended hours data. Defaults to True.
            full_history (bool, optional): Whether to retrieve the complete available history. Defaults to False.
            download_id (str, optional): A unique identifier for tracking this download. Defaults to None.
            debug_mode (bool, optional): Temporarily override the service's debug_mode setting for this call.
        
        Returns:
            dict: A dictionary with information about the download result
        """
        # Save original debug mode setting
        original_debug_mode = self.debug_mode
        
        # If debug_mode parameter is provided, temporarily override the service's setting
        if debug_mode is not None:
            self.debug_mode = debug_mode
            
        try:
            start_time = time.time()
            
            # Create a download ID if none was provided
            if download_id is None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                download_id = f"{ticker}_{timestamp}"
            
            # Start the actual download
            success, error_message = await self._download_ticker_data(ticker, days_back, include_extended_hours, full_history, download_id)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Download completed in {duration_ms:.2f}ms (success: {success})")
            
            # Build the result object
            result = {
                "download_id": download_id,
                "success": success,
                "ticker": ticker
            }
            
            # Include error message if there was a failure
            if not success and error_message:
                result["error"] = error_message
                
            # Include debug information if debug mode is enabled
            if self.debug_mode:
                # Construct a debug URL that can be used to manually verify the API call
                debug_url = self.construct_debug_url(ticker, days_back)
                
                # Get basic debug info about the service
                debug_info = self.get_debug_info()
                
                # Add call-specific details
                debug_info.update({
                    "api_url": debug_url,
                    "days_back": days_back,
                    "include_extended_hours": include_extended_hours,
                    "full_history": full_history
                })
                
                result["debug_info"] = debug_info
                
            return result
        except Exception as e:
            self.logger.error(f"Error in download_ticker_data for {ticker}: {str(e)}")
            return {
                "download_id": download_id,
                "success": False,
                "ticker": ticker,
                "error": str(e)
            }
        finally:
            # Restore original debug mode setting
            if debug_mode is not None:
                self.debug_mode = original_debug_mode
    
    async def _download_ticker_data(self, ticker, days_back, include_extended_hours, full_history, download_id=None):
        """
        Internal implementation of download_ticker_data.
        
        The debug mode is inherited from the parent method.
        """
        self.logger.info(f"Fetching historical data for {ticker}")
        await self._ensure_connected()
        
        # Use the provided download_id or create one based on the ticker
        download_key = download_id if download_id is not None else ticker
        
        # Clear any existing progress for this ticker
        self._init_download_progress(download_key)
        self._add_download_log(download_key, "Starting historical data download")
        
        # Update progress
        self._update_download_progress(download_key, 0.1, "Checking ticker metadata")
        
        # Convert ticker to lowercase for API, uppercase for DB
        api_ticker = ticker.lower()
        db_ticker = ticker.upper()
        
        # Ensure ticker exists in the database (maintains metadata)
        ticker_record = await Tickers.get_ticker(db_ticker)
        if not ticker_record:
            self._add_download_log(download_key, "Ticker not found, syncing metadata")
            # Use current debug mode setting when syncing metadata
            ticker_id = await self.sync_ticker_metadata(api_ticker)
            if not ticker_id:
                error_message = f"Failed to sync ticker metadata for {ticker}"
                self._add_download_log(download_key, error_message)
                self._update_download_progress(download_key, 1.0, "Failed to sync metadata", "failed")
                return False, error_message
        
        # Update progress    
        self._update_download_progress(download_key, 0.2, "Calculating date range")
        
        # Calculate the date range
        end_date = datetime.now()
        
        if full_history:
            # Use the get_full_history method for complete history
            self._update_download_progress(download_key, 0.3, "Fetching full history")
            self._add_download_log(download_key, "Downloading complete history from listing date")
            
            try:
                df = self.tiingo.get_full_history(
                    ticker=api_ticker,  # Use lowercase for API call
                    frequency="1min",
                    include_extended_hours=include_extended_hours
                )
            except Exception as e:
                error_message = str(e)
                self._add_download_log(download_key, f"Error fetching data: {error_message}")
                self._update_download_progress(download_key, 1.0, f"API error: {error_message}", "failed")
                self.logger.error(f"Error fetching data for {ticker}: {error_message}")
                return False, error_message
        else:
            # Use the specified days back
            start_date = end_date - timedelta(days=days_back)
            self._add_download_log(download_key, f"Fetching {days_back} days of price history")
            
            # Update progress
            self._update_download_progress(download_key, 0.3, "Fetching data from Tiingo")
            
            # Get the data from Tiingo
            try:
                self._add_download_log(download_key, f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                df = self.tiingo.get_historical_intraday(
                    ticker=api_ticker,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="1min",
                    include_extended_hours=include_extended_hours
                )
            except Exception as e:
                error_message = str(e)
                self._add_download_log(download_key, f"Error fetching data: {error_message}")
                self._update_download_progress(download_key, 1.0, f"API error: {error_message}", "failed")
                self.logger.error(f"Error fetching data for {ticker}: {error_message}")
                return False, error_message
        
        # Update progress    
        self._update_download_progress(download_key, 0.5, f"Processing {len(df) if df is not None else 0} records")
        
        if df is None or df.empty:
            error_message = "No data returned from Tiingo"
            self._add_download_log(download_key, error_message)
            self._update_download_progress(download_key, 1.0, "Completed - No data found", "failed")
            self.logger.warning(f"No data returned for {ticker}")
            return False, error_message
        
        # Update progress    
        self._update_download_progress(download_key, 0.7, "Processing data for storage")
        
        # Convert to format expected by RawData - explicitly map date to timestamp
        df_for_db = df.reset_index().rename(columns={
            'date': 'timestamp',
        })
        
        # Update progress
        self._update_download_progress(download_key, 0.8, "Storing data in database")
        
        # Store in the database
        try:
            self._add_download_log(download_key, f"Storing {len(df_for_db)} rows in database")
            
            # Ensure the ticker column is uppercase for consistency
            df_for_db['ticker'] = db_ticker
            
            result = await RawData.add_dataframe(df_for_db)
            
            if result["success"]:
                self._add_download_log(download_key, f"Successfully stored {result['rows_added']} rows in database")
                self._update_download_progress(download_key, 1.0, "Completed successfully", "success")
                return True, None
            else:
                error_message = result.get("error", "Unknown database error")
                self._add_download_log(download_key, f"Database error: {error_message}")
                self._update_download_progress(download_key, 1.0, f"Database error: {error_message}", "failed")
                return False, error_message
                
        except Exception as e:
            error_message = str(e)
            self._add_download_log(download_key, f"Database error: {error_message}")
            self._update_download_progress(download_key, 1.0, f"Database error: {error_message}", "failed")
            self.logger.error(f"Database error while storing data for {ticker}: {error_message}")
            return False, error_message
    
    def _init_download_progress(self, ticker: str):
        """Initialize download progress for a ticker."""
        self._download_progress[ticker] = {
            'progress': 0.0,
            'status': 'in_progress',
            'message': 'Initializing',
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        # Initialize log for this ticker
        if ticker not in self._download_logs:
            self._download_logs[ticker] = []
        self._add_download_log(ticker, "Download initialized")
    
    def _update_download_progress(self, ticker: str, progress: float, message: str, status: str = 'in_progress'):
        """Update download progress for a ticker."""
        if ticker not in self._download_progress:
            self._init_download_progress(ticker)
            
        self._download_progress[ticker].update({
            'progress': progress,
            'message': message,
            'status': status
        })
        
        # If status is completed or failed, set end time
        if status in ['success', 'completed', 'failed']:
            self._download_progress[ticker]['end_time'] = datetime.now().isoformat()
            
        # Add to log
        self._add_download_log(ticker, message)
        
    def _add_download_log(self, ticker: str, message: str):
        """Add a message to the download log for a ticker."""
        if ticker not in self._download_logs:
            self._download_logs[ticker] = []
            
        # Add the message directly to the logs list as a string
        self._download_logs[ticker].append(message)
        
        # Also log to application logger
        self.logger.info(f"[{ticker.upper()}] {message}")
    
    def get_download_progress(self, ticker: str = None) -> Dict:
        """Get download progress for a ticker or all tickers."""
        if ticker:
            return self._download_progress.get(ticker, {
                'progress': 0.0,
                'status': 'not_started',
                'message': 'Download not started',
                'start_time': None,
                'end_time': None
            })
        else:
            return self._download_progress
    
    def get_download_logs(self, ticker: str = None) -> List[str]:
        """Get download logs for a ticker or all tickers."""
        if ticker:
            # Return the logs for the specific ticker (empty list if none found)
            if ticker not in self._download_logs:
                self.logger.warning(f"No logs found for ticker {ticker}")
                return []
            return self._download_logs.get(ticker, [])
        else:
            return self._download_logs
    
    def get_download_status(self, ticker: str = None) -> Dict:
        """
        Get download status for a ticker or all tickers.
        This is an alias for get_download_progress to maintain backward compatibility.
        
        Args:
            ticker: Optional ticker symbol to get status for. If None, returns all statuses.
            
        Returns:
            Dictionary containing download status information
        """
        return self.get_download_progress(ticker)
    
    def clear_download_history(self, ticker: str = None):
        """Clear download history for a ticker or all tickers."""
        if ticker:
            if ticker in self._download_progress:
                del self._download_progress[ticker]
            if ticker in self._download_logs:
                del self._download_logs[ticker]
        else:
            self._download_progress = {}
            self._download_logs = {}
    
    def get_debug_info(self):
        """
        Get debug information about the service configuration.
        
        Returns:
            dict: A dictionary with debug information
        """
        return {
            "debug_mode": self.debug_mode,
            "tiingo_base_url": self.tiingo.base_url,
            "tiingo_iex_url": self.tiingo.iex_url,
            "rate_limit": self.tiingo.rate_limit,
            "api_key_masked": self._mask_api_key(self.tiingo.api_key) if self.tiingo.api_key else None
        }
    
    def construct_debug_url(self, ticker, days_back=7, frequency="1min", include_extended_hours=True, start_date=None, end_date=None):
        """
        Construct a debug URL that can be used to manually verify the API call in a browser.
        Masks the API key for security.
        
        Args:
            ticker (str): The ticker symbol to test
            days_back (int, optional): Number of days to look back. Defaults to 7.
            frequency (str, optional): Data frequency. Defaults to "1min".
            include_extended_hours (bool, optional): Whether to include extended hours data. Defaults to True.
            start_date (str or datetime, optional): Override start date. If None, it's calculated from days_back.
            end_date (str or datetime, optional): Override end date. If None, it's today.
            
        Returns:
            str: A URL that can be used to manually verify the API call
        """
        # Calculate dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None and days_back > 0:
            start_date = end_date - timedelta(days=days_back)
            
        # Convert datetime objects to strings if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")
            
        # Use lowercase ticker for API calls
        api_ticker = ticker.lower()
        
        # Get base URL from Tiingo client
        url = f"{self.tiingo.iex_url}/{api_ticker}/prices"
        
        # Construct query string
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": frequency,
            "columns": "date,open,high,low,close,volume",
            "format": "json",
            "token": self._mask_api_key(self.tiingo.api_key),
            "afterHours": "true" if include_extended_hours else "false"
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{url}?{query_string}"
    
    async def trace_api_call(self, ticker, days_back=7, include_extended_hours=True):
        """
        Perform a test API call with detailed logging and return debug information.
        This is useful for diagnosing issues with API calls.
        
        Args:
            ticker (str): The ticker symbol to test
            days_back (int, optional): Number of days to look back. Defaults to 7.
            include_extended_hours (bool, optional): Whether to include extended hours data. Defaults to True.
            
        Returns:
            dict: A dictionary with:
                - success: Whether the API call was successful
                - debug_info: Debug information including URLs, response details, etc.
                - error: Details about the error if it failed
        """
        # Temporarily enable debug mode for this call
        original_debug_mode = self.debug_mode
        self.debug_mode = True
        
        try:
            self.logger.info(f"Tracing API call for ticker: {ticker}")
            
            # Generate a debug URL for manual inspection
            debug_url = self.construct_debug_url(ticker, days_back)
            
            # Prepare dates for the API call
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Get the basic debug information
            debug_info = self.get_debug_info()
            
            # Add call-specific details
            debug_info.update({
                "ticker": ticker,
                "api_url": debug_url,
                "days_back": days_back,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "include_extended_hours": include_extended_hours
            })
            
            # Attempt the API call
            try:
                # Use lowercase ticker for API calls
                api_ticker = ticker.lower()
                
                # This will make the actual API call
                df = self.tiingo.get_historical_intraday(
                    ticker=api_ticker,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="1min",
                    include_extended_hours=include_extended_hours
                )
                
                # Record API call success details
                success = True if df is not None and not df.empty else False
                if success:
                    debug_info["data_points"] = len(df)
                    debug_info["first_timestamp"] = df.index[0].strftime("%Y-%m-%d %H:%M:%S") if len(df) > 0 else None
                    debug_info["last_timestamp"] = df.index[-1].strftime("%Y-%m-%d %H:%M:%S") if len(df) > 0 else None
                else:
                    error_message = "API call succeeded but returned empty data"
                    debug_info["error"] = error_message
                    return {
                        "success": False,
                        "debug_info": debug_info,
                        "error": error_message
                    }
                
                return {
                    "success": True,
                    "debug_info": debug_info
                }
                
            except Exception as e:
                # Record API call failure details
                error_message = str(e)
                error_type = type(e).__name__
                
                # Add error details to debug info
                debug_info["error_type"] = error_type
                debug_info["error_message"] = error_message
                
                # For HTTP errors, include status code
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    debug_info["status_code"] = e.response.status_code
                    # For 404 errors, note that the ticker may not exist
                    if e.response.status_code == 404:
                        debug_info["ticker_exists"] = False
                        debug_info["suggestion"] = "The ticker symbol may not exist or be supported by Tiingo"
                
                self.logger.error(f"API trace error: {error_type}: {error_message}")
                
                return {
                    "success": False,
                    "debug_info": debug_info,
                    "error": error_message
                }
                
        except Exception as e:
            # This catches any errors in our tracing code, not in the API call itself
            error_message = f"Error during trace: {str(e)}"
            self.logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
        finally:
            # Restore original debug mode
            self.debug_mode = original_debug_mode
    
    def _mask_api_key(self, api_key):
        """
        Mask an API key for secure display in debug information.
        Shows only the first and last 4 characters.
        
        Args:
            api_key (str): The API key to mask
            
        Returns:
            str: The masked API key
        """
        if not api_key or len(api_key) < 8:
            return "***API_KEY_MASKED***"
            
        visible_chars = 4  # Show first and last 4 characters
        masked_key = api_key[:visible_chars] + "*" * (len(api_key) - 2 * visible_chars) + api_key[-visible_chars:]
        return masked_key
    
    async def database_diagnostics(self, include_tables=True, test_query=True):
        """
        Run database diagnostics to troubleshoot connection issues.
        
        Args:
            include_tables (bool): Whether to check table information
            test_query (bool): Whether to run a test query
            
        Returns:
            dict: Diagnostic information about the database connection
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "service_info": {
                "debug_mode": self.debug_mode,
                "is_connected": self._is_connected,
                "use_test_db": self.use_test_db
            },
            "connection_status": "unknown",
            "connection_error": None,
        }
        
        # Temporarily enable debug mode for diagnostics if not already enabled
        original_debug_mode = self.debug_mode
        if not self.debug_mode:
            self.set_debug_mode(True)
            
        try:
            # Try to connect to the database
            self.logger.info("Running database diagnostics...")
            await self._ensure_connected()
            
            result["connection_status"] = "connected"
            
            # Get database connection information
            if self._db:
                result["database_info"] = self._db.get_connection_info()
                
                # Try to get table information if requested
                if include_tables:
                    try:
                        tables_info, raw_data_tables, processed_tables = await self.get_database_info()
                        result["tables"] = {
                            "count": len(tables_info),
                            "raw_data_tables": raw_data_tables,
                            "processed_tables": processed_tables
                        }
                    except Exception as e:
                        result["tables_error"] = str(e)
                
                # Try a simple test query if requested
                if test_query:
                    try:
                        # Run a simple query to verify database functionality
                        test_result = await self._db.fetchval("SELECT current_timestamp")
                        result["test_query"] = {
                            "success": test_result is not None,
                            "result": str(test_result)
                        }
                    except Exception as e:
                        result["test_query"] = {
                            "success": False,
                            "error": str(e)
                        }
                        
        except Exception as e:
            result["connection_status"] = "error"
            result["connection_error"] = str(e)
            import traceback
            result["error_traceback"] = traceback.format_exc()
            
        finally:
            # Restore original debug mode if we changed it
            if original_debug_mode != self.debug_mode:
                self.set_debug_mode(original_debug_mode)
                
        return result
    
    async def verify_tables(self, ticker=None):
        """
        Verify database tables for a specific ticker or all raw data tables.
        
        Args:
            ticker (str, optional): Specific ticker to check. If None, checks all raw data tables.
            
        Returns:
            dict: Information about table status and structure
        """
        await self._ensure_connected()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "tables_checked": 0,
            "tables_ok": 0,
            "tables_with_issues": 0,
            "table_details": {}
        }
        
        try:
            # Get all tables info
            tables_info, raw_data_tables, processed_tables = await self.get_database_info()
            
            # If a specific ticker is provided, check only that table
            if ticker:
                ticker = ticker.upper()  # Standardize to uppercase
                table_name = f"raw_data_{ticker.lower()}"  # Lowercase for table name
                tables_to_check = [table_name] if table_name in raw_data_tables else []
                if not tables_to_check:
                    result["error"] = f"Table for ticker {ticker} not found"
            else:
                # Check all raw data tables
                tables_to_check = raw_data_tables
            
            result["tables_checked"] = len(tables_to_check)
            
            # Check each table
            for table_name in tables_to_check:
                table_result = await self._check_table_structure(table_name)
                result["table_details"][table_name] = table_result
                
                if table_result["status"] == "ok":
                    result["tables_ok"] += 1
                else:
                    result["tables_with_issues"] += 1
            
        except Exception as e:
            result["error"] = str(e)
            if self.debug_mode:
                import traceback
                result["error_traceback"] = traceback.format_exc()
                
        return result
    
    async def _check_table_structure(self, table_name):
        """
        Check the structure of a specific table.
        
        Args:
            table_name (str): The name of the table to check
            
        Returns:
            dict: Information about the table structure
        """
        result = {
            "status": "unknown",
            "row_count": 0,
            "columns": [],
            "primary_key": None,
            "indexes": [],
            "issues": []
        }
        
        try:
            # Check if table exists
            exists_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = $1
            );
            """
            exists = await self._db.fetchval(exists_query, table_name)
            
            if not exists:
                result["status"] = "error"
                result["issues"].append("Table does not exist")
                return result
            
            # Get column information
            columns_query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position;
            """
            columns = await self._db.fetch(columns_query, table_name)
            
            if not columns:
                result["status"] = "error"
                result["issues"].append("Table exists but has no columns")
                return result
            
            result["columns"] = [
                {
                    "name": col["column_name"],
                    "type": col["data_type"],
                    "nullable": col["is_nullable"] == "YES"
                }
                for col in columns
            ]
            
            # Check for primary key
            pk_query = """
            SELECT a.attname as column_name
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = $1::regclass
            AND i.indisprimary;
            """
            pk_columns = await self._db.fetch(pk_query, table_name)
            if pk_columns:
                result["primary_key"] = [col["column_name"] for col in pk_columns]
            
            # Check for indexes
            index_query = """
            SELECT
                i.relname as index_name,
                a.attname as column_name,
                ix.indisunique as is_unique
            FROM
                pg_class t,
                pg_class i,
                pg_index ix,
                pg_attribute a
            WHERE
                t.oid = ix.indrelid
                and i.oid = ix.indexrelid
                and a.attrelid = t.oid
                and a.attnum = ANY(ix.indkey)
                and t.relkind = 'r'
                and t.relname = $1
            ORDER BY
                i.relname;
            """
            indexes = await self._db.fetch(index_query, table_name)
            
            # Group indexes by name
            index_dict = {}
            for idx in indexes:
                index_name = idx["index_name"]
                if index_name not in index_dict:
                    index_dict[index_name] = {
                        "name": index_name,
                        "columns": [],
                        "unique": idx["is_unique"]
                    }
                index_dict[index_name]["columns"].append(idx["column_name"])
            
            result["indexes"] = list(index_dict.values())
            
            # Get row count
            count_query = f"SELECT COUNT(*) FROM {table_name};"
            result["row_count"] = await self._db.fetchval(count_query)
            
            # Check for common issues
            if not result["primary_key"]:
                result["issues"].append("Table has no primary key")
            
            required_columns = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]
            missing_columns = [col for col in required_columns if col not in [c["name"] for c in result["columns"]]]
            if missing_columns:
                result["issues"].append(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Set final status
            if result["issues"]:
                result["status"] = "warning"
            else:
                result["status"] = "ok"
                
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(f"Error checking table: {str(e)}")
            if self.debug_mode:
                import traceback
                result["error_traceback"] = traceback.format_exc()
            
        return result
