import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import time
import collections
from typing import Dict, Union, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RateLimiter:
    """
    Rate limiter to enforce API request limits.
    
    Tracks requests over different time periods (hourly and daily)
    and enforces waiting when limits are approached.
    """
    
    def __init__(self, hourly_limit: int = 10000, daily_limit: int = 100000):
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.hourly_timestamps = collections.deque()
        self.daily_timestamps = collections.deque()
        self.logger = logging.getLogger(__name__)
    
    def wait_if_needed(self):
        """
        Check if we need to wait before making a request and wait if necessary.
        
        This method enforces both hourly and daily rate limits by tracking
        request timestamps and calculating appropriate wait times.
        """
        current_time = time.time()
        
        # Update our record of hourly requests (rolling 1-hour window)
        hour_ago = current_time - 3600
        while self.hourly_timestamps and self.hourly_timestamps[0] < hour_ago:
            self.hourly_timestamps.popleft()
            
        # Update our record of daily requests (rolling 24-hour window)
        day_ago = current_time - 86400
        while self.daily_timestamps and self.daily_timestamps[0] < day_ago:
            self.daily_timestamps.popleft()
        
        # Check if we're approaching hourly limits
        if len(self.hourly_timestamps) >= self.hourly_limit * 0.9:  # 90% of limit
            # Calculate time to wait until oldest request leaves the window
            wait_time = max(0, self.hourly_timestamps[0] - hour_ago)
            if wait_time > 0:
                self.logger.warning(f"Approaching hourly rate limit. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Check if we're approaching daily limits
        if len(self.daily_timestamps) >= self.daily_limit * 0.9:  # 90% of limit
            # Calculate time to wait until oldest request leaves the window
            wait_time = max(0, self.daily_timestamps[0] - day_ago)
            if wait_time > 0:
                self.logger.warning(f"Approaching daily rate limit. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Calculate minimum interval between requests to stay under hourly limit
        min_hourly_interval = 3600 / self.hourly_limit
        
        # Apply a small wait time even if we're not near limits to avoid bursts
        if self.hourly_timestamps:
            time_since_last = current_time - self.hourly_timestamps[-1]
            if time_since_last < min_hourly_interval:
                small_wait = min_hourly_interval - time_since_last
                time.sleep(small_wait)
        
        # Record this request
        request_time = time.time()  # Get current time again after any waits
        self.hourly_timestamps.append(request_time)
        self.daily_timestamps.append(request_time)
        
        return request_time

class Tiingo:
    """
    Simple Tiingo API client for fetching intraday price data.
    
    This class provides basic methods to make requests to the Tiingo API.
    It handles authentication and provides simple wrappers for the API endpoints.
    
    Attributes:
        api_key (str): Tiingo API key from environment variables
        base_url (str): Base URL for Tiingo API
        iex_url (str): URL for IEX endpoint (intraday data)
        headers (dict): HTTP headers including API key
        logger (logging.Logger): Logger for the class
        debug_mode (bool): Whether to log detailed request info
        rate_limiter (RateLimiter): Handles API rate limiting
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance of the Tiingo client."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self, debug_mode=False):
        """
        Initialize the Tiingo API client with API key and endpoints.
        
        Args:
            debug_mode (bool): Whether to log detailed request info
        """
        # If an instance already exists, use that one
        if Tiingo._instance is not None:
            raise RuntimeError("Tiingo instance already exists. Use Tiingo.get_instance() to get the singleton instance.")
            
        # Load API key from environment
        self.api_key = os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY environment variable not set")
            
        # Log the key being used (first few characters)
        logging.getLogger(__name__).info(f"Initialized Tiingo with API key: {self.api_key[:8]}...")
        
        # Set up URLs and headers
        self.base_url = "https://api.tiingo.com"
        self.iex_url = f"{self.base_url}/iex"
        self.headers = {'Content-Type': 'application/json'}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode
        
        # Set up rate limiting
        self.rate_limiter = RateLimiter(hourly_limit=10000, daily_limit=100000)
        
        # Register this as the singleton instance
        Tiingo._instance = self
    
    def set_debug_mode(self, enabled=True):
        """Enable or disable debug mode which logs detailed request information."""
        self.debug_mode = enabled
        self.logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to comply with API rate limits."""
        return self.rate_limiter.wait_if_needed()
    
    def _log_request(self, url, params):
        """Log request details if debug mode is enabled."""
        if not self.debug_mode:
            return
        
        # Create a safe copy of params without the API key
        safe_params = params.copy() if params else {}
        if 'token' in safe_params:
            safe_params['token'] = '********'
            
        # Log the request
        self.logger.debug(f"Request URL: {url}")
        self.logger.debug(f"Request params: {json.dumps(safe_params, indent=2)}")
    
    def get_ticker_metadata(self, ticker: str) -> Dict:
        """
        Get metadata for a ticker including listing date.
        
        Args:
            ticker (str): Ticker symbol (lowercase)
            
        Returns:
            Dict: Metadata for the ticker
        """
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/tiingo/daily/{ticker}"
        params = {'token': self.api_key}
        
        self._log_request(url, params)
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if self.debug_mode:
                self.logger.debug(f"Response status code: {response.status_code}")
                if response.status_code != 200:
                    self.logger.debug(f"Response body: {response.text}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching metadata for {ticker}: {e}")
            return {}
    
    def get_listing_date(self, ticker: str) -> Optional[datetime]:
        """
        Get the listing date for a ticker.
        
        Args:
            ticker (str): Ticker symbol (lowercase)
            
        Returns:
            Optional[datetime]: Listing date or None if not found
        """
        metadata = self.get_ticker_metadata(ticker)
        if metadata and 'startDate' in metadata:
            return datetime.fromisoformat(metadata['startDate'].replace('Z', '+00:00'))
        return None
    
    def get_historical_intraday(
        self, 
        ticker: str, 
        start_date: Union[str, datetime],
        frequency: str = "1min",
        include_extended_hours: bool = True
    ) -> pd.DataFrame:
        """
        Get historical intraday data for a ticker for a specific month.
        
        The Tiingo API returns data for the entire month containing the start_date.
        
        Args:
            ticker (str): Ticker symbol (lowercase)
            start_date (Union[str, datetime]): Date within the month to retrieve
            frequency (str): Data frequency ('1min', '5min', '30min', '1hour')
            include_extended_hours (bool): Include pre-market and after-hours data
            
        Returns:
            pd.DataFrame: Historical intraday OHLCV data with date as index
        """
        self._wait_for_rate_limit()
        
        # Convert to string format if datetime object
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
            
        # Prepare request parameters  
        params = {
            'startDate': start_date,  # Tiingo uses this to determine the month
            'resampleFreq': frequency,
            'columns': 'date,open,high,low,close,volume',
            'format': 'json',
            'token': self.api_key
        }
        
        # Add parameter for extended hours if requested
        if include_extended_hours:
            params['afterHours'] = 'true'
            
        url = f"{self.iex_url}/{ticker}/prices"
        
        # Log request details
        self.logger.info(f"Requesting intraday data for {ticker} for date: {start_date}")
        self._log_request(url, params)
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            # Log response details
            if self.debug_mode:
                self.logger.debug(f"Response status code: {response.status_code}")
                if response.status_code != 200:
                    self.logger.debug(f"Response body: {response.text[:500]}...")  # Truncate long responses
            else:
                self.logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.error(f"Error response from API: {response.text}")
                
            response.raise_for_status()
            data = response.json()
            
            if not data:
                self.logger.warning(f"No data returned for {ticker} for date {start_date}")
                return pd.DataFrame()
                
            # Create DataFrame from response data
            df = pd.DataFrame(data)
            
            # Convert date column to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df['ticker'] = ticker.upper()  # Add ticker column in uppercase
            df.set_index('date', inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching intraday data for {ticker}: {e}")
            return pd.DataFrame()