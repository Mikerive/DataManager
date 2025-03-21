import csv
import requests
import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()

class AlphaVantage:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AlphaVantage API client.
        
        Args:
            api_key: API key for AlphaVantage. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_listing_status(self, date: Optional[str] = None, state: str = "active") -> List[Dict[str, str]]:
        """Get listing status of stocks and ETFs.
        
        Args:
            date: Date in YYYY-MM-DD format. If not provided, returns latest trading day.
                Any date later than 2010-01-01 is supported.
            state: 'active' or 'delisted'. Default is 'active'.
        
        Returns:
            List of dictionaries containing listing data.
        """
        params = {
            "function": "LISTING_STATUS",
            "apikey": self.api_key,
        }
        
        if date:
            params["date"] = date
        
        if state:
            params["state"] = state
        
        return self._get_csv_data(params)
    
    def get_all_listings(self, date: Optional[str] = None) -> List[Dict[str, str]]:
        """Get both active and delisted listings in a single call.
        
        Args:
            date: Date in YYYY-MM-DD format. If not provided, returns latest trading day.
                Any date later than 2010-01-01 is supported.
        
        Returns:
            Combined list of dictionaries containing both active and delisted listings.
        """
        # Get active listings
        active_listings = self.get_listing_status(date=date, state="active")
        
        # Get delisted listings
        delisted_listings = self.get_listing_status(date=date, state="delisted")
        
        # Combine both lists
        return active_listings + delisted_listings
    
    def get_earnings_calendar(self, symbol: Optional[str] = None, horizon: str = "3month") -> List[Dict[str, str]]:
        """Get earnings calendar for companies.
        
        Args:
            symbol: Stock symbol. If not provided, returns data for all companies.
            horizon: Time period - '3month', '6month', or '12month'. Default is '3month'.
        
        Returns:
            List of dictionaries containing earnings data.
        """
        params = {
            "function": "EARNINGS_CALENDAR",
            "apikey": self.api_key,
            "horizon": horizon
        }
        
        if symbol:
            params["symbol"] = symbol
        
        return self._get_csv_data(params)
    
    def _get_csv_data(self, params: Dict[str, str]) -> List[Dict[str, Any]]:
        """Helper method to get and process CSV data from AlphaVantage.
        
        Args:
            params: Dictionary of query parameters.
            
        Returns:
            List of dictionaries containing the data.
        """
        with requests.Session() as session:
            response = session.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            decoded_content = response.content.decode('utf-8')
            csv_reader = csv.reader(decoded_content.splitlines(), delimiter=',')
            
            data = list(csv_reader)
            if not data:
                return []
            
            headers = data[0]
            result = []
            
            for row in data[1:]:
                if len(row) == len(headers):
                    result.append(dict(zip(headers, row)))
            
            return result


# Example usage:
if __name__ == "__main__":
    alpha = AlphaVantage()
    
    # Example 1: Get all active listings
    active_listings = alpha.get_listing_status()
    print(f"Active listings count: {len(active_listings)}")
    if active_listings:
        print(f"First active listing: {active_listings[0]}")
    
    # Example 2: Get delisted stocks from a specific date
    delisted = alpha.get_listing_status(date="2014-07-10", state="delisted")
    print(f"Delisted count: {len(delisted)}")
    if delisted:
        print(f"First delisted: {delisted[0]}")
    
    # Example 3: Get all listings (both active and delisted) in one call
    all_listings = alpha.get_all_listings()
    print(f"All listings count: {len(all_listings)}")
    if all_listings:
        print(f"First listing: {all_listings[0]}")
    
    # Example 4: Get earnings calendar for next 3 months
    earnings = alpha.get_earnings_calendar()
    print(f"Earnings events count: {len(earnings)}")
    if earnings:
        print(f"First earnings event: {earnings[0]}")
    
    # Example 5: Get IBM earnings for next 12 months
    ibm_earnings = alpha.get_earnings_calendar(symbol="IBM", horizon="12month")
    print(f"IBM earnings events count: {len(ibm_earnings)}")
    if ibm_earnings:
        print(f"IBM earnings: {ibm_earnings[0]}")
