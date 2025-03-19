import streamlit as st
import asyncio
import pandas as pd
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union
import nest_asyncio

# For handling async functions in Streamlit
try:
    nest_asyncio.apply()
except RuntimeError:
    # Nest-asyncio already applied
    pass

# Cache for services
_raw_data_service = None
_bar_processing_service = None
_data_integrity_service = None

def load_css():
    """Load CSS for styling the dashboard."""
    return """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #1E3A8A;
        }
        
        .sub-header {
            font-size: 1.8rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            color: #1E3A8A;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid #E5E7EB;
        }
        
        .metric-container {
            padding: 0.5rem;
            border-radius: 0.5rem;
            background-color: #F3F4F6;
            margin-bottom: 1rem;
        }
        
        .ticker-card {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #F3F4F6;
            margin-bottom: 1rem;
            border-left: 4px solid #1E3A8A;
        }
        
        .download-status {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #F0F9FF;
            margin: 1rem 0;
            border-left: 4px solid #0EA5E9;
        }
        
        .download-status p {
            margin: 0.5rem 0;
        }
        
        .status-completed {
            color: #047857;
            font-weight: bold;
        }
        
        .status-failed {
            color: #DC2626;
            font-weight: bold;
        }
        
        .status-in-progress {
            color: #0EA5E9;
            font-weight: bold;
        }
        
        .data-quality-high {
            color: #047857;
            font-weight: bold;
        }
        
        .data-quality-medium {
            color: #D97706;
            font-weight: bold;
        }
        
        .data-quality-low {
            color: #DC2626;
            font-weight: bold;
        }
    </style>
    """

def run_async(func: Callable, *args, **kwargs) -> Any:
    """
    Run an async function from a sync context.
    This is necessary when calling async functions from Streamlit.
    
    Args:
        func: The async function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the async function
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(func(*args, **kwargs))

def calculate_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily statistics from a DataFrame of OHLCV data.
    
    Args:
        df: DataFrame with timestamp, open, high, low, close, volume columns
        
    Returns:
        DataFrame with daily statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Ensure timestamp is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create a day column
    df['day'] = df['timestamp'].dt.date
    
    # Group by day and calculate daily statistics
    daily_stats = df.groupby('day').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'timestamp': 'count'
    }).rename(columns={'timestamp': 'bar_count'})
    
    # Calculate daily return
    daily_stats['daily_return'] = (daily_stats['close'] / daily_stats['close'].shift(1) - 1) * 100
    
    # Calculate daily range (high-low as percentage of open)
    daily_stats['daily_range'] = (daily_stats['high'] - daily_stats['low']) / daily_stats['open'] * 100
    
    return daily_stats 

async def ensure_service_connected(service):
    """
    Ensure a service is connected to the database.
    This should be called before using any service method that requires database access.
    
    Args:
        service: The service instance to check
        
    Returns:
        The service instance, guaranteed to be connected
    """
    if service is None:
        return None
    
    try:
        # Call the internal _ensure_connected method
        await service._ensure_connected()
        return service
    except Exception as e:
        # Log the error
        st.error(f"Error connecting service to database: {str(e)}")
        st.code(traceback.format_exc())
        raise 