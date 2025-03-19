import asyncio
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import database classes using absolute imports
from backend.db.Database import Database
from backend.db.models.RawData import RawData
from backend.db.models.Tickers import Tickers
from backend.db.models.ProcessedData import ProcessedData

# Import RawDataService for real market data
from backend.services.RawDataService.RawDataService import RawDataService
from backend.lib.Tiingo import Tiingo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample tickers to include in the database
SAMPLE_TICKERS = [
    ("AAPL", "Apple Inc.", "NASDAQ", "Stock", datetime(1980, 12, 12), None, "active"),
    ("MSFT", "Microsoft Corporation", "NASDAQ", "Stock", datetime(1986, 3, 13), None, "active"),
    ("GOOG", "Alphabet Inc.", "NASDAQ", "Stock", datetime(2004, 8, 19), None, "active"),
    ("AMZN", "Amazon.com Inc.", "NASDAQ", "Stock", datetime(1997, 5, 15), None, "active"),
    ("META", "Meta Platforms Inc.", "NASDAQ", "Stock", datetime(2012, 5, 18), None, "active")
]


async def init_database():
    """Initialize the database with tables and fetch real market data using RawDataService."""
    logger.info("Starting database initialization")
    
    # Force DB_HOST to localhost when running locally
    os.environ["DB_HOST"] = "localhost"
    
    # Initialize database connection (singleton)
    db = Database()
    
    try:
        # Connect to database
        logger.info("Connecting to database...")
        await db.connect()
        print("✅ Connected to the database!")
        
        # Initialize database wrapper classes
        tickers_db = Tickers()
        raw_data_db = RawData()
        processed_data_db = ProcessedData()
        
        # Create tickers table
        logger.info("Creating tickers table...")
        await tickers_db.create_tickers_table()
        
        # Create processed data tables
        logger.info("Creating processed data template table...")
        await processed_data_db.create_template_table()
        
        # Reset raw data structure to ensure proper partitioning
        logger.info("Resetting raw data structure...")
        await raw_data_db.reset_raw_data_structure()
        
        # Add sample tickers if they don't exist
        for ticker_data in SAMPLE_TICKERS:
            ticker = ticker_data[0]
            
            # Check if ticker already exists
            existing = await tickers_db.get_ticker(ticker)
            if not existing:
                # Add ticker to the database
                logger.info(f"Adding ticker: {ticker}")
                await tickers_db.add_ticker(*ticker_data)
            else:
                logger.info(f"Ticker already exists: {ticker}")
        
        # Get all tickers to setup partitioning
        all_tickers = await tickers_db.get_all_tickers()
        logger.info(f"Found {len(all_tickers)} tickers in the database")
        
        # Initialize raw data tables for each ticker
        for ticker_record in all_tickers:
            ticker = ticker_record['ticker']
            ticker_id = ticker_record['id']
            
            # Create partition for this ticker
            logger.info(f"Creating raw data partition for {ticker}")
            await raw_data_db.create_ticker_partition(ticker)
            
            # Create processed data partitions for different bar types for this ticker
            for bar_type in ['time', 'tick', 'volume', 'dollar', 'entropy', 'information']:
                logger.info(f"Creating processed data partition for {ticker} / {bar_type}")
                await processed_data_db.create_ticker_partition(ticker_id, bar_type)
        
        # Ensure Tiingo API key is set
        if not os.getenv("TIINGO_API_KEY"):
            logger.error("TIINGO_API_KEY environment variable is not set.")
            logger.info("Database tables have been initialized, but no market data was fetched.")
            return
        
        # Check if extended hours should be included (default to True)
        include_extended_hours = os.getenv("INCLUDE_EXTENDED_HOURS", "True").lower() in ("true", "1", "yes")
        
        # Flag to determine if we should import full history (set to True for full history)
        import_full_history = True
        
        logger.info(f"Data fetching settings: full_history={import_full_history}, include_extended_hours={include_extended_hours}")
        
        # Initialize RawDataService for fetching real market data
        try:
            # Get the Tiingo singleton instance
            _ = Tiingo.get_instance()  # This will initialize the singleton
            raw_data_service = RawDataService(use_test_db=False)  # Use production DB for init
            
            # Configure which tickers to process
            tickers_to_process = [ticker_data[0] for ticker_data in SAMPLE_TICKERS]
            
            # Option to process specific tickers only
            if os.getenv("SPECIFIC_TICKER"):
                specific_ticker = os.getenv("SPECIFIC_TICKER").upper()
                if specific_ticker in tickers_to_process:
                    tickers_to_process = [specific_ticker]
                    logger.info(f"Processing only ticker: {specific_ticker}")
                else:
                    logger.warning(f"Specified ticker {specific_ticker} not in sample list, processing all")
            
            if import_full_history:
                # Process each ticker individually for full history to be more robust
                for ticker in tickers_to_process:
                    logger.info(f"Fetching full history for {ticker}")
                    
                    # Fetch the full history for this ticker
                    result = await raw_data_service.download_ticker_data(
                        ticker=ticker,
                        include_extended_hours=include_extended_hours,
                        full_history=True
                    )
                    
                    if result['success']:
                        logger.info(f"✅ Successfully fetched full history for {ticker}")
                    else:
                        logger.error(f"❌ Failed to fetch full history for {ticker}: {result.get('error', 'Unknown error')}")
                    
                    # Sleep between API calls to be nice to the API
                    logger.info("Waiting 5 seconds before processing the next ticker...")
                    await asyncio.sleep(5)
            else:
                # For partial history, we can use the batch import with days_back
                days_back = 90
                logger.info(f"Fetching {days_back} days of data for sample tickers")
                
                # Override the method with a custom one that passes days_back and include_extended_hours
                original_fetch_method = raw_data_service.fetch_and_store_historical_data
                
                async def custom_fetch(ticker, **kwargs):
                    # Explicitly pass only the parameters we want (days_back and include_extended_hours)
                    return await original_fetch_method(ticker, days_back, include_extended_hours)
                
                # Replace the original method with our custom one
                raw_data_service.fetch_and_store_historical_data = custom_fetch
                
                # Process with the custom method in batch mode
                results = await raw_data_service.sync_multiple_tickers(
                    tickers=tickers_to_process,
                    full_history=False,
                    include_extended_hours=include_extended_hours
                )
                
                # Log results
                for ticker, success in results.items():
                    if success:
                        logger.info(f"Successfully fetched data for {ticker}")
                    else:
                        logger.warning(f"Failed to fetch data for {ticker}")
                
            logger.info("Data import completed successfully")
                    
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            logger.info("Database tables have been initialized, but market data fetch failed.")
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Close database connection
        if hasattr(db, 'pool') and db.pool is not None:
            logger.info("Closing database connection...")
            await db.close()
            print("🔌 Database connection closed.")
        logger.info("Database connections closed")


def main():
    """Run the database initialization."""
    asyncio.run(init_database())


if __name__ == "__main__":
    main()
