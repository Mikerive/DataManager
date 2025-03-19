from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Optional
from services.tickerService import TickerService
import os

router = APIRouter()

# Initialize service with API key from environment variable
def get_ticker_service():
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    return TickerService(api_key)

# Pydantic models for request validation
class TickerCreate(BaseModel):
    symbol: str
    name: str
    exchange: str
    asset_type: str
    ipo_date: str = None
    delisting_date: str = None
    status: str = "active"

class TimeSeriesRequest(BaseModel):
    symbols: List[str]
    interval: str = "daily"  # daily, weekly, monthly

class TickersEndpoint:
    @router.post('/tickers/initialize')
    async def initialize_tickers(self, background_tasks: BackgroundTasks, service: TickerService = Depends(get_ticker_service)):
        """
        Initialize the ticker database by downloading all tickers from AlphaVantage.
        This is a long-running operation, so it runs in the background.
        """
        # Run in background to avoid blocking the API
        background_tasks.add_task(service.initialize_ticker_database)
        return {"message": "Ticker database initialization started in background"}

    @router.post('/tickers')
    async def add_ticker(self, ticker_data: TickerCreate, service: TickerService = Depends(get_ticker_service)):
        """Add or update a single ticker with provided data"""
        # Convert pydantic model to dict
        ticker_dict = ticker_data.dict()
        
        result = await service.update_ticker(ticker_dict)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @router.get('/tickers/{symbol}')
    async def get_ticker(self, symbol: str, service: TickerService = Depends(get_ticker_service)):
        """Get information for a single ticker"""
        ticker = await service.get_ticker(symbol)
        if not ticker:
            raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")
        return ticker

    @router.delete('/tickers/{symbol}')
    async def delete_ticker(self, symbol: str, service: TickerService = Depends(get_ticker_service)):
        """Delete a ticker from the database"""
        result = await service.delete_ticker(symbol)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
        
    @router.get('/tickers/{symbol}/timeseries')
    async def get_ticker_timeseries(
        self, 
        symbol: str, 
        interval: str = Query("daily", description="Data interval: daily, weekly, or monthly"),
        service: TickerService = Depends(get_ticker_service)
    ):
        """
        Get time series data with technical indicators for a ticker symbol.
        
        - **symbol**: Ticker symbol (e.g., AAPL, MSFT)
        - **interval**: Data interval (daily, weekly, monthly)
        """
        # First check if ticker exists in our database
        ticker = await service.get_ticker(symbol)
        if not ticker:
            raise HTTPException(status_code=404, detail=f"Ticker {symbol} not found")
            
        # Get time series data
        result = await service.get_time_series(symbol, interval)
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"Time series data not available for {symbol}"
            )
            
        return result
        
    @router.post('/tickers/timeseries/batch')
    async def batch_get_timeseries(
        self, 
        request: TimeSeriesRequest,
        service: TickerService = Depends(get_ticker_service)
    ):
        """
        Get time series data for multiple ticker symbols in a single request.
        
        Request body:
        - **symbols**: List of ticker symbols
        - **interval**: Data interval (daily, weekly, monthly)
        """
        if not request.symbols:
            raise HTTPException(
                status_code=400,
                detail="At least one symbol must be provided"
            )
            
        if len(request.symbols) > 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 symbols allowed per batch request"
            )
            
        result = await service.batch_get_time_series(request.symbols, request.interval)
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch time series data: {result.get('error', 'Unknown error')}"
            )
            
        return result
