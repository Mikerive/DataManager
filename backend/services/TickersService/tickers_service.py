import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import asyncio

from backend.lib.AlphaVantage import AlphaVantage
from backend.db.models.Tickers import Tickers

class TickersService:
    """
    Service for fetching, storing, and updating ticker data using AlphaVantage API.
    Integrates with the Tickers database model to persist data.
    """
    
    logger = logging.getLogger(__name__)
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the TickersService.
        
        Args:
            api_key: Optional API key for AlphaVantage. If not provided, will use environment variable.
        """
        self.alpha_vantage = AlphaVantage(api_key)
    
    async def fetch_and_store_active_tickers(self) -> Tuple[int, int]:
        """
        Fetch active tickers from AlphaVantage and store them in the database.
        
        Returns:
            Tuple containing (number of tickers fetched, number of tickers stored)
        """
        self.logger.info("Fetching active tickers from AlphaVantage")
        
        try:
            # Fetch active listings
            listings = self.alpha_vantage.get_listing_status(state="active")
            
            if not listings:
                self.logger.warning("No active tickers found")
                return 0, 0
                
            # Convert to DataFrame for processing
            df = pd.DataFrame(listings)
            self.logger.info(f"Fetched {len(df)} active tickers from AlphaVantage")
            
            # Prepare data for database
            df_processed = self._prepare_tickers_dataframe(df, "active")
            
            # Store in database
            ticker_ids = await Tickers.add_tickers_from_dataframe(df_processed)
            
            self.logger.info(f"Successfully stored {len(ticker_ids)} active tickers in database")
            return len(df), len(ticker_ids)
            
        except Exception as e:
            self.logger.error(f"Error fetching and storing active tickers: {str(e)}")
            return 0, 0
    
    async def fetch_and_store_delisted_tickers(self, date: Optional[str] = None) -> Tuple[int, int]:
        """
        Fetch delisted tickers from AlphaVantage and store them in the database.
        
        Args:
            date: Optional date string in YYYY-MM-DD format. If not provided, uses the latest data.
            
        Returns:
            Tuple containing (number of tickers fetched, number of tickers stored)
        """
        self.logger.info(f"Fetching delisted tickers from AlphaVantage{' for ' + date if date else ''}")
        
        try:
            # Fetch delisted listings
            listings = self.alpha_vantage.get_listing_status(date=date, state="delisted")
            
            if not listings:
                self.logger.warning("No delisted tickers found")
                return 0, 0
                
            # Convert to DataFrame for processing
            df = pd.DataFrame(listings)
            self.logger.info(f"Fetched {len(df)} delisted tickers from AlphaVantage")
            
            # Prepare data for database
            df_processed = self._prepare_tickers_dataframe(df, "delisted")
            
            # Store in database
            ticker_ids = await Tickers.add_tickers_from_dataframe(df_processed)
            
            self.logger.info(f"Successfully stored {len(ticker_ids)} delisted tickers in database")
            return len(df), len(ticker_ids)
            
        except Exception as e:
            self.logger.error(f"Error fetching and storing delisted tickers: {str(e)}")
            return 0, 0
    
    async def fetch_and_store_all_tickers(self) -> Tuple[int, int]:
        """
        Fetch both active and delisted tickers and store them in the database.
        
        Returns:
            Tuple containing (total number of tickers fetched, total number of tickers stored)
        """
        self.logger.info("Fetching all tickers (active and delisted) from AlphaVantage")
        
        try:
            # Fetch all tickers (active and delisted) in a single API call
            listings = self.alpha_vantage.get_all_listings()
            
            if not listings:
                self.logger.warning("No tickers found")
                return 0, 0
                
            # Convert to DataFrame for processing
            df = pd.DataFrame(listings)
            self.logger.info(f"Fetched {len(df)} tickers from AlphaVantage")
            
            # Prepare data for database - we don't specify a status here since
            # the AlphaVantage API should provide the status in the data
            df_processed = self._prepare_tickers_dataframe(df, status=None)
            
            # Store in database
            ticker_ids = await Tickers.add_tickers_from_dataframe(df_processed)
            
            self.logger.info(f"Successfully stored {len(ticker_ids)} tickers in database")
            return len(df), len(ticker_ids)
            
        except Exception as e:
            self.logger.error(f"Error fetching and storing all tickers: {str(e)}")
            return 0, 0
    
    async def update_ticker_status(self, ticker: str, new_status: str, delisting_date: Optional[str] = None) -> bool:
        """
        Update a ticker's status in the database.
        
        Args:
            ticker: The ticker symbol to update
            new_status: New status ('active' or 'delisted')
            delisting_date: Date string (YYYY-MM-DD) when ticker was delisted (required if status is 'delisted')
            
        Returns:
            True if the update was successful, False otherwise
        """
        self.logger.info(f"Updating ticker {ticker} status to {new_status}")
        
        try:
            # Convert string date to datetime if provided
            delisting_datetime = None
            if delisting_date:
                delisting_datetime = datetime.fromisoformat(delisting_date.replace('Z', '+00:00'))
            
            # Update ticker in database
            success = await Tickers.update_ticker_status(ticker, new_status, delisting_datetime)
            
            if success:
                self.logger.info(f"Successfully updated ticker {ticker} status to {new_status}")
            else:
                self.logger.warning(f"Failed to update ticker {ticker} status")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating ticker {ticker} status: {str(e)}")
            return False
    
    async def get_earnings_calendar(self, symbol: Optional[str] = None, horizon: str = "3month") -> List[Dict[str, Any]]:
        """
        Fetch earnings calendar data from AlphaVantage.
        
        Args:
            symbol: Optional ticker symbol to filter data for
            horizon: Time period for earnings data ('3month', '6month', or '12month')
            
        Returns:
            List of earnings calendar entries
        """
        self.logger.info(f"Fetching earnings calendar for {'all companies' if not symbol else symbol}")
        
        try:
            # Fetch earnings calendar data
            earnings = self.alpha_vantage.get_earnings_calendar(symbol=symbol, horizon=horizon)
            
            self.logger.info(f"Fetched {len(earnings)} earnings calendar entries")
            return earnings
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings calendar: {str(e)}")
            return []
    
    async def search_tickers(self, search_term: Optional[str] = None,
                          exchange: Optional[str] = None,
                          asset_type: Optional[str] = None,
                          status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for tickers in the database.
        
        Args:
            search_term: Text to search in ticker symbol or name
            exchange: Filter by exchange
            asset_type: Filter by asset type ('Stock' or 'ETF')
            status: Filter by status ('active' or 'delisted')
            
        Returns:
            List of matching ticker entries
        """
        self.logger.info(f"Searching for tickers with term: {search_term}")
        
        try:
            # Search for tickers in database
            results = await Tickers.search_tickers(
                search_term=search_term,
                exchange=exchange,
                asset_type=asset_type,
                status=status,
                as_dataframe=False
            )
            
            self.logger.info(f"Found {len(results)} tickers matching search criteria")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching for tickers: {str(e)}")
            return []
    
    def _prepare_tickers_dataframe(self, df: pd.DataFrame, status: Optional[str] = None) -> pd.DataFrame:
        """
        Process raw AlphaVantage DataFrame into the format expected by the Tickers database model.
        
        Args:
            df: Raw DataFrame from AlphaVantage
            status: Status to set for the tickers ('active' or 'delisted').
                    If None, will use status from data if available.
            
        Returns:
            Processed DataFrame ready for database insertion
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Based on analysis of listing_status.csv, we know the exact column names to expect
        # Column names from AlphaVantage API are: symbol, name, exchange, assetType, ipoDate, delistingDate, status
        # Map to our database column names
        column_mapping = {
            'symbol': 'ticker',
            'assetType': 'asset_type',
            'ipoDate': 'ipo_date',
            'delistingDate': 'delisting_date'
        }
        
        # Rename columns that need to be renamed
        for old_col, new_col in column_mapping.items():
            if old_col in result_df.columns:
                result_df.rename(columns={old_col: new_col}, inplace=True)
        
        # Make sure ticker is uppercase - analysis shows it's already uppercase, but just to be sure
        if 'ticker' in result_df.columns:
            result_df['ticker'] = result_df['ticker'].str.upper()
        
        # Handle missing values in required fields
        required_fields = ['ticker', 'name', 'exchange', 'asset_type']
        for field in required_fields:
            if field in result_df.columns and result_df[field].isnull().any():
                self.logger.warning(f"Found null values in required field '{field}'. These rows will be dropped.")
                result_df = result_df.dropna(subset=[field])
        
        # Handle status field - analysis shows AlphaVantage provides 'Active' 
        if 'status' in result_df.columns:
            # Map the status values to our expected format (lowercase)
            result_df['status'] = result_df['status'].str.lower()
        else:
            # Set status if not provided by AlphaVantage
            result_df['status'] = status or 'active'
        
        # Convert date strings to datetime objects
        # Analysis shows dates are consistently in YYYY-MM-DD format
        if 'ipo_date' in result_df.columns:
            result_df['ipo_date'] = pd.to_datetime(result_df['ipo_date'], errors='coerce')
            
        if 'delisting_date' in result_df.columns:
            result_df['delisting_date'] = pd.to_datetime(result_df['delisting_date'], errors='coerce')
        
        # Handle missing delisting dates for securities with 'delisted' status
        # For both explicitly provided status and status from the data
        if 'status' in result_df.columns:
            is_delisted = result_df['status'].str.lower() == 'delisted'
            
            # Check if we have any delisted securities without a delisting date
            has_missing_date = False
            if 'delisting_date' in result_df.columns:
                has_missing_date = result_df.loc[is_delisted, 'delisting_date'].isna().any()
            else:
                # If no delisting_date column exists at all
                has_missing_date = is_delisted.any()
                # Create the column if it doesn't exist
                if has_missing_date and 'delisting_date' not in result_df.columns:
                    result_df['delisting_date'] = None
            
            # Set current date for delisted securities without a date
            if has_missing_date:
                current_date = pd.Timestamp(datetime.now())
                # Use .loc to avoid SettingWithCopyWarning
                result_df.loc[is_delisted & result_df['delisting_date'].isna(), 'delisting_date'] = current_date
        
        return result_df
    
    async def close(self):
        """Close any open resources."""
        await Tickers.close_connection() 