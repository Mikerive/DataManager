import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional, Union, Tuple
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Tiingo:
    """
    Tiingo API client for fetching historical intraday price data.
    
    This class provides methods to fetch intraday historical data going back to
    listing date for various tickers. It supports both synchronous and asynchronous
    requests, rate limiting, and error handling.
    
    Attributes:
        api_key (str): Tiingo API key from environment variables
        base_url (str): Base URL for Tiingo API
        iex_url (str): URL for IEX endpoint (intraday data)
        headers (dict): HTTP headers including API key
        logger (logging.Logger): Logger for the class
        rate_limit (int): Maximum requests per minute
        last_request_time (float): Timestamp of last API request
        debug_mode (bool): Whether to log detailed request info
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Get or create the singleton instance of the Tiingo client.
        
        Returns:
            Tiingo: The singleton instance of the Tiingo client
        """
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
            
        # Load environment variables with dotenv to ensure we get the updated values
        load_dotenv()
        
        self.api_key = os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY environment variable not set")
            
        # Log the key being used (first few characters)
        logging.getLogger(__name__).info(f"Initialized Tiingo with API key: {self.api_key[:8]}...")
        
        self.base_url = "https://api.tiingo.com"
        self.iex_url = f"{self.base_url}/iex"
        
        # Properly format the Authorization header with 'Token' prefix
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }
        
        self.logger = logging.getLogger(__name__)
        self.rate_limit = 100  # Tiingo's free tier limit (requests per minute)
        self.last_request_time = 0
        self.debug_mode = debug_mode
        
        # Register this as the singleton instance
        Tiingo._instance = self
    
    def set_debug_mode(self, enabled=True):
        """
        Enable or disable debug mode which logs detailed request information.
        
        Args:
            enabled (bool): Whether to enable debug mode
        """
        self.debug_mode = enabled
        self.logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
    def _log_request(self, method, url, headers=None, params=None, data=None):
        """
        Log detailed request information in debug mode.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            url (str): Request URL
            headers (dict): HTTP headers
            params (dict): Query parameters
            data (dict): Request body data
        """
        if not self.debug_mode:
            return
            
        # Clone and sanitize headers to avoid logging API keys
        safe_headers = {}
        if headers:
            safe_headers = headers.copy()
            if 'Authorization' in safe_headers:
                auth_parts = safe_headers['Authorization'].split(' ')
                if len(auth_parts) > 1:
                    safe_headers['Authorization'] = f"{auth_parts[0]} {'*' * 8}"
        
        # Format the full request details
        request_info = {
            'method': method,
            'url': url,
            'headers': safe_headers,
            'params': params,
            'data': data
        }
        
        # Create a formatted string representation
        formatted_request = json.dumps(request_info, indent=2)
        self.logger.debug(f"Tiingo API Request:\n{formatted_request}")
    
    def _rate_limit_wait(self):
        """Implement rate limiting to avoid API throttling."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # Ensure we don't exceed rate limit
        min_interval = 60.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def get_ticker_metadata(self, ticker: str) -> Dict:
        """
        Get metadata for a ticker including listing date.
        
        Args:
            ticker (str): Ticker symbol
            
        Returns:
            Dict: Metadata for the ticker
        """
        self._rate_limit_wait()
        # Convert ticker to lowercase only for API call
        api_ticker = ticker.lower()
        url = f"{self.base_url}/tiingo/daily/{api_ticker}"
        
        # Use token as query parameter
        params = {'token': self.api_key}
        headers = {'Content-Type': 'application/json'}
        
        # Log the request details in debug mode
        self._log_request("GET", url, headers, params)
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if self.debug_mode:
                self.logger.debug(f"Response status code: {response.status_code}")
                self.logger.debug(f"Response headers: {dict(response.headers)}")
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
            ticker (str): Ticker symbol
            
        Returns:
            Optional[datetime]: Listing date or None if not found
        """
        # Convert ticker to lowercase only for API call
        api_ticker = ticker.lower()
        metadata = self.get_ticker_metadata(api_ticker)
        if metadata and 'startDate' in metadata:
            return datetime.fromisoformat(metadata['startDate'].replace('Z', '+00:00'))
        return None
    
    def get_historical_intraday(
        self, 
        ticker: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        frequency: str = "1min",
        include_extended_hours: bool = True
    ) -> pd.DataFrame:
        """
        Get historical intraday data for a ticker.
        
        Args:
            ticker (str): Ticker symbol
            start_date (Union[str, datetime]): Start date
            end_date (Union[str, datetime], optional): End date, defaults to today
            frequency (str, optional): Data frequency ('1min', '5min', '30min', '1hour')
            include_extended_hours (bool, optional): Include pre-market and after-hours data
            
        Returns:
            pd.DataFrame: Historical intraday OHLCV data with date as index and ticker column
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Convert to string format if datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Only request OHLCV fields
        columns = [
            'date',
            'open',
            'high',
            'low',
            'close',
            'volume'
        ]
            
        self._rate_limit_wait()
        
        # Convert ticker to lowercase only for API call
        api_ticker = ticker.lower()
        
        # Add token as a query parameter instead of in the Authorization header
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'resampleFreq': frequency,
            'columns': ','.join(columns),
            'format': 'json',
            'token': self.api_key  # Add token as a query parameter
        }
        
        # Add parameter for extended hours if requested
        if include_extended_hours:
            params['afterHours'] = 'true'
        
        url = f"{self.iex_url}/{api_ticker}/prices"
        
        # Log the complete request URL and parameters for debugging
        self.logger.info(f"Requesting intraday data for {ticker} from {start_date} to {end_date}")
        if self.debug_mode:
            self.logger.debug(f"Request URL: {url}")
            self.logger.debug(f"Request parameters: {params}")
        
        # Use Content-Type header only, not Authorization
        headers = {'Content-Type': 'application/json'}
        
        # Log full request details in debug mode
        self._log_request("GET", url, headers, params)
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            # Log response details
            if self.debug_mode:
                self.logger.debug(f"Response status code: {response.status_code}")
                self.logger.debug(f"Response headers: {dict(response.headers)}")
                if response.status_code != 200:
                    self.logger.debug(f"Response body: {response.text[:500]}...")  # Truncate long responses
            else:
                self.logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.error(f"Error response from API: {response.text}")
            
            response.raise_for_status()
            data = response.json()
            
            if not data:
                self.logger.warning(f"No data returned for {ticker} from {start_date} to {end_date}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add ticker column in UPPERCASE - needed for database storage
            df['ticker'] = ticker.upper()
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching intraday data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_full_history(
        self, 
        ticker: str, 
        frequency: str = "1min",
        batch_size: int = 30,  # Days per request
        max_retries: int = 3,
        include_extended_hours: bool = True
    ) -> pd.DataFrame:
        """
        Get full historical intraday data from listing date to present.
        
        This method chunks requests to handle Tiingo's limitations on
        historical data range per request.
        
        Args:
            ticker (str): Ticker symbol
            frequency (str, optional): Data frequency
            batch_size (int, optional): Days per API request
            max_retries (int, optional): Maximum retry attempts per chunk
            include_extended_hours (bool, optional): Include pre-market and after-hours data
            
        Returns:
            pd.DataFrame: Complete historical intraday data
        """
        # Convert ticker to lowercase only for API call
        api_ticker = ticker.lower()
        listing_date = self.get_listing_date(api_ticker)
        if not listing_date:
            self.logger.error(f"Could not determine listing date for {ticker}")
            return pd.DataFrame()
            
        # Ensure we don't go back further than Tiingo's data availability
        # (typically ~5 years for intraday data)
        five_years_ago = datetime.now() - timedelta(days=365*5)
        start_date = max(listing_date, five_years_ago)
        end_date = datetime.now()
        
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=batch_size), end_date)
            
            if self.debug_mode:
                self.logger.debug(f"Fetching chunk from {current_start} to {current_end}")
            
            # Try with retries
            for attempt in range(max_retries):
                chunk_data = self.get_historical_intraday(
                    ticker=api_ticker,
                    start_date=current_start,
                    end_date=current_end,
                    frequency=frequency,
                    include_extended_hours=include_extended_hours
                )
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    break
                    
                if attempt < max_retries - 1:
                    self.logger.warning(f"Retrying chunk for {ticker} ({current_start} to {current_end})")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            current_start = current_end + timedelta(days=1)
        
        if not all_data:
            return pd.DataFrame()
            
        # Combine all chunks and sort by date
        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep='first')]
        result.sort_index(inplace=True)
        
        return result
    
    async def get_historical_intraday_async(
        self, 
        ticker: str, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        frequency: str = "1min",
        include_extended_hours: bool = True
    ) -> pd.DataFrame:
        """
        Get historical intraday data for a ticker asynchronously.
        
        Args:
            ticker (str): Ticker symbol
            start_date (Union[str, datetime]): Start date
            end_date (Union[str, datetime], optional): End date, defaults to today
            frequency (str, optional): Data frequency ('1min', '5min', '30min', '1hour')
            include_extended_hours (bool, optional): Include pre-market and after-hours data
            
        Returns:
            pd.DataFrame: Historical intraday OHLCV data with date as index
        """
        if end_date is None:
            end_date = datetime.now()
            
        # Convert to string format if datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Only request OHLCV fields
        columns = [
            'date',
            'open',
            'high',
            'low',
            'close',
            'volume'
        ]
            
        self._rate_limit_wait()
        
        # Convert ticker to lowercase only for API call
        api_ticker = ticker.lower()
        
        # Add token as a query parameter instead of in the Authorization header
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'resampleFreq': frequency,
            'columns': ','.join(columns),
            'format': 'json',
            'token': self.api_key  # Add token as a query parameter
        }
        
        # Add parameter for extended hours if requested
        if include_extended_hours:
            params['afterHours'] = 'true'
        
        url = f"{self.iex_url}/{api_ticker}/prices"
        
        # Log the complete request URL and parameters for debugging
        self.logger.info(f"Requesting intraday data for {ticker} from {start_date} to {end_date}")
        self.logger.info(f"Request URL: {url}")
        self.logger.info(f"Request parameters: {params}")
        
        # Use Content-Type header only, not Authorization
        headers = {'Content-Type': 'application/json'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    # Log response details for debugging
                    self.logger.info(f"Response status code: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Error response from API: {error_text}")
                        response.raise_for_status()
                    
                    data = await response.json()
                    
                    if not data:
                        self.logger.warning(f"No data returned for {ticker} from {start_date} to {end_date}")
                        return pd.DataFrame()
                    
                    df = pd.DataFrame(data)
                    
                    # Convert date column to datetime
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Add ticker column in UPPERCASE
                    df['ticker'] = ticker.upper()
                    
                    # Set date as index
                    df.set_index('date', inplace=True)
                    
                    return df
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching intraday data for {ticker}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_tickers(
        self, 
        tickers: List[str], 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        frequency: str = "1min",
        include_extended_hours: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical intraday data for multiple tickers asynchronously.
        
        Args:
            tickers (List[str]): List of ticker symbols
            start_date (Union[str, datetime]): Start date
            end_date (Union[str, datetime], optional): End date, defaults to today
            frequency (str, optional): Data frequency ('1min', '5min', '30min', '1hour')
            include_extended_hours (bool, optional): Include pre-market and after-hours data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping tickers to their historical data
        """
        # Create tasks for each ticker
        tasks = [
            self.get_historical_intraday_async(
                ticker=ticker.lower(),  # Ensure lowercase for API
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                include_extended_hours=include_extended_hours
            )
            for ticker in tickers
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Map results to tickers
        return {ticker: df for ticker, df in zip(tickers, results)}