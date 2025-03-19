import pandas as pd
import logging
from datetime import datetime
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from backend.db.Database import Database
from backend.db.utils.db_utils import log_db_error, log_db_success
from backend.db.utils.db_utils import execute_values


class RawData:
    """
    Model for raw price data from Tiingo.
    
    This class handles the storage and retrieval of raw price data in the database.
    The data is partitioned by ticker_id for better performance.
    """
    
    # Required columns in the DataFrame
    REQUIRED_COLUMNS = [
        'ticker_id',  # For partitioning
        'open',
        'high',
        'low',
        'close',
        'volume'
    ]
    
    TABLE_TEMPLATE = "raw_data_template"
    logger = logging.getLogger(__name__)
    _db = None
    
    @classmethod
    async def _get_db(cls):
        """Get or create a Database instance for this class."""
        if cls._db is None or cls._db.pool is None or cls._db.pool._closed:
            cls._db = Database(owner_name="RawData")
            await cls._db.connect()
        return cls._db
    
    @classmethod
    async def close_connection(cls):
        """Close the database connection if it exists."""
        if cls._db is not None:
            await cls._db.close()
            cls._db = None
    
    @classmethod
    async def create_template_table(cls) -> None:
        """Create the template table for raw data with partitioning by ticker_id."""
        try:
            start_time = datetime.now()
            db = await cls._get_db()
            
            # Create template table with ticker_id partitioning
            query = """
            CREATE TABLE IF NOT EXISTS raw_data_template (
                id SERIAL PRIMARY KEY,
                ticker_id INT NOT NULL REFERENCES tickers(id) ON DELETE CASCADE,
                timestamp TIMESTAMPTZ NOT NULL,  -- Matches Tiingo's timestamp field
                quote_timestamp TIMESTAMPTZ,     -- Matches Tiingo's quoteTimestamp field
                last_sale_timestamp TIMESTAMPTZ, -- Matches Tiingo's lastSaleTimeStamp field
                last DECIMAL(10, 4),            -- Matches Tiingo's last field
                last_size INT,                  -- Matches Tiingo's lastSize field
                tngo_last DECIMAL(10, 4),       -- Matches Tiingo's tngoLast field
                prev_close DECIMAL(10, 4),      -- Matches Tiingo's prevClose field
                open DECIMAL(10, 4) NOT NULL,   -- Matches Tiingo's open field
                high DECIMAL(10, 4) NOT NULL,   -- Matches Tiingo's high field
                low DECIMAL(10, 4) NOT NULL,    -- Matches Tiingo's low field
                mid DECIMAL(10, 4),             -- Matches Tiingo's mid field
                volume BIGINT NOT NULL,         -- Matches Tiingo's volume field
                bid_size INT,                   -- Matches Tiingo's bidSize field
                bid_price DECIMAL(10, 4),       -- Matches Tiingo's bidPrice field
                ask_size INT,                   -- Matches Tiingo's askSize field
                ask_price DECIMAL(10, 4),       -- Matches Tiingo's askPrice field
                UNIQUE (ticker_id, timestamp) ON CONFLICT (ticker_id, timestamp) DO UPDATE SET 
                    quote_timestamp = EXCLUDED.quote_timestamp,
                    last_sale_timestamp = EXCLUDED.last_sale_timestamp,
                    last = EXCLUDED.last,
                    last_size = EXCLUDED.last_size,
                    tngo_last = EXCLUDED.tngo_last,
                    prev_close = EXCLUDED.prev_close,
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    mid = EXCLUDED.mid,
                    volume = EXCLUDED.volume,
                    bid_size = EXCLUDED.bid_size,
                    bid_price = EXCLUDED.bid_price,
                    ask_size = EXCLUDED.ask_size,
                    ask_price = EXCLUDED.ask_price
            ) PARTITION BY LIST (ticker_id);
            """
            
            await db.execute_query(query)
            
            # Convert to hypertable for efficient time-series querying
            await db.execute_query("SELECT create_hypertable('raw_data_template', 'timestamp');")
            
            # Create index for fast lookups
            await db.execute_query("CREATE INDEX IF NOT EXISTS idx_raw_data_ticker_time ON raw_data_template (ticker_id, timestamp DESC);")
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_db_success("Create raw data template table", duration_ms, cls.logger)
        except Exception as e:
            log_db_error("Create raw data template table", e, cls.logger)
    
    @classmethod
    async def drop_template_table(cls):
        """Drop the raw_data_template table and all partitions if they exist."""
        try:
            start_time = datetime.now()
            db = await cls._get_db()
            
            query = """
            DROP TABLE IF EXISTS raw_data_template CASCADE;
            """
            
            await db.execute_query(query)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_db_success("Drop template table", duration_ms, cls.logger)
            
            return True
        except Exception as e:
            log_db_error("Drop template table", e, cls.logger)
            return False
                
    @classmethod
    async def reset_raw_data_structure(cls):
        """Reset the entire raw data structure, dropping all tables and recreating the template."""
        try:
            start_time = datetime.now()
            
            # Drop template first (cascades to all partitions)
            await cls.drop_template_table()
            
            # Create new template
            await cls.create_template_table()
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_db_success("Reset raw data structure", duration_ms, cls.logger)
            
            return True
        except Exception as e:
            log_db_error("Reset raw data structure", e, cls.logger)
            return False

    @classmethod
    def _sanitize_ticker(cls, ticker: str) -> str:
        """
        Convert ticker symbol to a safe table name format.
        Always uses lowercase for PostgreSQL compatibility.
        
        Args:
            ticker: The ticker symbol (e.g., 'AAPL', 'BRK.A', 'BF-B')
            
        Returns:
            Sanitized ticker suitable for table name
        """
        # Replace special characters with underscores and convert to lowercase for DB
        # (PostgreSQL treats identifiers as case-insensitive by default)
        safe_ticker = re.sub(r'[^a-zA-Z0-9]', '_', ticker.lower())
        return safe_ticker
        
    @classmethod
    async def create_ticker_partition(cls, ticker: str) -> bool:
        """
        Create a table for a specific ticker.
        
        Args:
            ticker: The ticker symbol
            
        Returns:
            True if the table was created successfully, False otherwise
        """
        try:
            db = await cls._get_db()
            
            # Convert ticker to uppercase for display/reference
            display_ticker = ticker.upper()
            
            # But use lowercase for table name (PostgreSQL compatibility)
            safe_ticker = cls._sanitize_ticker(ticker)
            table_name = f"raw_data_{safe_ticker}"
            
            # Check if the table already exists
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = '{table_name}'
            );
            """
            
            exists = await db.fetchval(check_query)
            if exists:
                # If the table exists, let's drop and recreate it 
                # to ensure it matches our current schema
                drop_query = f"DROP TABLE IF EXISTS {table_name};"
                await db.execute_query(drop_query)
                cls.logger.info(f"Dropped existing table {table_name} to recreate with correct schema")
                
            # Create the table with the correct schema
            # Use timestamp as primary key, no ticker column needed since it's in the table name
            query = f"""
            CREATE TABLE {table_name} (
                timestamp TIMESTAMP NOT NULL,
                open FLOAT NOT NULL,
                high FLOAT NOT NULL,
                low FLOAT NOT NULL,
                close FLOAT NOT NULL,
                volume FLOAT NOT NULL,
                PRIMARY KEY (timestamp)
            );
            """
            await db.execute_query(query)
            
            # Create an index on timestamp for faster queries
            index_query = f"""
            CREATE INDEX idx_{table_name}_timestamp 
            ON {table_name} (timestamp);
            """
            await db.execute_query(index_query)
            
            cls.logger.info(f"Created table {table_name} for ticker {display_ticker}")
            return True
        except Exception as e:
            log_db_error(f"Create table for ticker {ticker}", e, cls.logger)
            return False
    
    @classmethod
    async def delete_ticker_partition(cls, ticker: str) -> bool:
        """
        Delete a raw data partition for a specific ticker.
        
        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            
        Returns:
            True if deleted, False if doesn't exist
        """
        try:
            start_time = datetime.now()
            db = await cls._get_db()
            
            safe_ticker = cls._sanitize_ticker(ticker)
            table_name = f"raw_data_{safe_ticker}"
            
            # First check if the table exists
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = '{table_name}'
            );
            """
            
            exists = await db.fetchval(check_query)
            if not exists:
                return False
            
            # If it exists, drop it
            query = f"""
            DROP TABLE IF EXISTS {table_name} CASCADE;
            """
            
            await db.execute_query(query)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_db_success(f"Delete ticker partition for {ticker}", duration_ms, cls.logger)
            
            return True
        except Exception as e:
            log_db_error(f"Delete ticker partition for {ticker}", e, cls.logger)
            return False
    
    @classmethod
    async def add_price_datapoint(
        cls,
        ticker: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> bool:
        """
        Add a single raw data entry for a ticker.
        
        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            timestamp: Timestamp of the data point
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            volume: Trading volume
            
        Returns:
            True if successfully added
        """
        try:
            start_time = datetime.now()
            db = await cls._get_db()
            
            # Make sure partition exists
            await cls.create_ticker_partition(ticker)
            
            query = f"""
            INSERT INTO raw_data_template (ticker, timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (ticker, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume;
            """
            
            await db.execute_query(query, ticker, timestamp, open_price, high, low, close, volume)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_db_success(f"Add price datapoint for {ticker}", duration_ms, cls.logger)
            
            return True
        except Exception as e:
            log_db_error(f"Add price datapoint for {ticker}", e, cls.logger)
            return False
    
    @classmethod
    async def get_ticker_data(cls, ticker: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Get raw data for a specific ticker within a time range.
        
        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with raw price data
        """
        try:
            operation_start = datetime.now()
            db = await cls._get_db()
            
            # Convert datetime objects to string if needed
            if isinstance(start_time, datetime):
                start_time = start_time.isoformat()
            if isinstance(end_time, datetime):
                end_time = end_time.isoformat()
            
            # Check if the partition exists
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = 'raw_data_{cls._sanitize_ticker(ticker)}'
            );
            """
            
            exists = await db.fetchval(check_query)
                
            if not exists:
                # Return empty DataFrame if partition doesn't exist
                return pd.DataFrame()
            
            # Fetch data if partition exists
            query = f"""
            SELECT * FROM raw_data_template 
            WHERE ticker = $1 AND timestamp BETWEEN $2 AND $3
            ORDER BY timestamp DESC;
            """
            
            rows = await db.fetch(query, ticker, start_time, end_time)
            
            # Convert to DataFrame
            if not rows:
                return pd.DataFrame()
                
            result = pd.DataFrame(
                [(r['ticker'], r['timestamp'], r['open'], r['high'], r['low'], r['close'], r['volume']) 
                    for r in rows],
                columns=['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            duration_ms = (datetime.now() - operation_start).total_seconds() * 1000
            log_db_success(f"Get ticker data for {ticker}", duration_ms, cls.logger)
            
            return result
        except Exception as e:
            log_db_error(f"Get ticker data for {ticker}", e, cls.logger)
            return pd.DataFrame()
    
    @classmethod
    def _ensure_timestamp_is_datetime(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure timestamp column is in datetime format."""
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    @classmethod
    async def add_dataframe(cls, df: pd.DataFrame) -> Dict[str, any]:
        """
        Add a dataframe of price data to the database
        
        Required columns: timestamp, open, high, low, close, volume, ticker
        
        Args:
            df: The dataframe to add
            
        Returns:
            Dict with success status and rows_added count
        """
        
        # Check for required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]
        for col in required_columns:
            if col not in df.columns:
                cls.logger.error(f"Missing required column: {col}")
                return {"success": False, "rows_added": 0}
                
        # Handle empty dataframe
        if df.empty:
            cls.logger.warning("Empty dataframe, nothing to add")
            return {"success": True, "rows_added": 0}
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure timestamp is datetime and normalize to UTC without timezone info
        if not pd.api.types.is_datetime64_ns_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Handle timezone-aware datetimes by converting to UTC and removing timezone info
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Group by ticker for batch processing
        total_rows_added = 0
        success = True
        
        # Group the dataframe by ticker
        for ticker, ticker_df in df.groupby("ticker"):
            try:
                # Display ticker in uppercase for logging, but use lowercase for table name
                display_ticker = ticker.upper()
                safe_ticker = cls._sanitize_ticker(ticker)
                table_name = f"raw_data_{safe_ticker}"
                
                # Ensure the table exists
                table_exists = await cls.create_ticker_partition(ticker)
                if not table_exists:
                    cls.logger.error(f"Failed to create table for ticker {display_ticker}")
                    success = False
                    continue
                
                # Remove the ticker column as it's not in the table schema and not needed
                # (ticker is encoded in the table name)
                cols_to_insert = ["timestamp", "open", "high", "low", "close", "volume"]
                data = ticker_df[cols_to_insert]
                
                # Start a transaction and insert the data
                db = await cls._get_db()
                
                async with db.pool.acquire() as conn:
                    # Build values for batch insert
                    values = []
                    for _, row in data.iterrows():
                        values.append((
                            row["timestamp"],
                            float(row["open"]),
                            float(row["high"]),
                            float(row["low"]),
                            float(row["close"]),
                            float(row["volume"])
                        ))
                    
                    # Create SQL query for batch insert with ON CONFLICT
                    sql = f"""
                    INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                    """
                    
                    # Execute batch insert
                    await conn.executemany(sql, values)
                    
                # Log success
                rows_added = len(data)
                total_rows_added += rows_added
                cls.logger.info(f"Added {rows_added} rows for ticker {display_ticker}")
                
            except Exception as e:
                cls.logger.error(f"Error inserting data for {ticker.upper()}: {str(e)}")
                success = False
        
        return {"success": success, "rows_added": total_rows_added}
    
    @classmethod
    async def get_price_data(cls, 
                          ticker_or_table: str, 
                          start_date: datetime = None, 
                          end_date: datetime = None,
                          limit: int = 10000) -> pd.DataFrame:
        """
        Get price data for a ticker with optional date filtering.
        
        Args:
            ticker_or_table: Ticker symbol (e.g., 'AAPL') or table name ('raw_data_aapl')
            start_date: Optional start datetime to filter data
            end_date: Optional end datetime to filter data
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with price data
        """
        try:
            start_time = datetime.now()
            db = await cls._get_db()
            
            # Handle both ticker symbols and table names
            if ticker_or_table.startswith('raw_data_'):
                # Extract ticker from table name
                table_name = ticker_or_table.lower()  # Ensure lowercase for DB
            else:
                # Convert ticker to appropriate format
                table_name = f"raw_data_{cls._sanitize_ticker(ticker_or_table)}"
            
            # Check if the ticker table exists
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = '{table_name}'
            );
            """
            
            exists = await db.fetchval(check_query)
            if not exists:
                cls.logger.warning(f"Table {table_name} does not exist")
                return pd.DataFrame()
            
            # Base query
            if start_date and end_date:
                query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table_name}
                WHERE timestamp BETWEEN $1 AND $2
                ORDER BY timestamp
                LIMIT {limit};
                """
                params = [start_date, end_date]
            else:
                query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table_name}
                ORDER BY timestamp
                LIMIT {limit};
                """
                params = []
            
            rows = await db.fetch(query, *params)
                
            if not rows:
                cls.logger.warning(f"No data found for {table_name}")
                return pd.DataFrame()
                
            df = pd.DataFrame(
                [(r['timestamp'], r['open'], r['high'], r['low'], r['close'], r['volume']) 
                 for r in rows],
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df = df.sort_values('timestamp')
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_db_success(f"Get price data for {table_name}", duration_ms, cls.logger)
            
            return df
        except Exception as e:
            log_db_error(f"Get price data for {ticker_or_table}", e, cls.logger)
            return pd.DataFrame()
    
    @classmethod
    async def list_ticker_partitions(cls) -> List[str]:
        """
        List all ticker partitions that have been created.
        
        Returns:
            List of ticker symbols with data partitions
        """
        try:
            start_time = datetime.now()
            db = await cls._get_db()
            
            query = """
            SELECT tablename 
            FROM pg_catalog.pg_tables 
            WHERE schemaname = 'public' AND tablename LIKE 'raw_data_%'
            AND tablename != 'raw_data_template';
            """
            
            rows = await db.fetch(query)
            
            # Extract ticker symbols from table names
            tickers = []
            prefix = "raw_data_"
            for row in rows:
                table_name = row['tablename']
                if table_name.startswith(prefix):
                    # This is a simplification - we'd need to map from sanitized back to original
                    # for tickers with special characters
                    ticker = table_name[len(prefix):]
                    tickers.append(ticker)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_db_success("List ticker partitions", duration_ms, cls.logger)
                    
            return tickers
        except Exception as e:
            log_db_error("List ticker partitions", e, cls.logger)
            return []
        
    @classmethod
    async def get_ticker_statistics(cls, ticker: str) -> Dict[str, Any]:
        """
        Get statistics for a ticker's data, such as row count, date range, etc.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with statistics
        """
        try:
            start_time = datetime.now()
            db = await cls._get_db()
            
            # Check if the ticker partition exists
            safe_ticker = cls._sanitize_ticker(ticker)
            table_name = f"raw_data_{safe_ticker}"
            
            # Debug logging
            cls.logger.debug(f"Getting statistics for ticker {ticker}, table {table_name}")
            
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = '{table_name}'
            );
            """
            
            exists = await db.fetchval(check_query)
            if not exists:
                cls.logger.debug(f"Table {table_name} does not exist")
                return {
                    "ticker": ticker,
                    "exists": False,
                    "count": 0,
                    "min_date": None,
                    "max_date": None,
                    "avg_volume": None
                }
            
            # Table exists, get statistics
            query = """
            SELECT 
                COUNT(*) as row_count,
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                AVG(volume) as avg_volume
            FROM raw_data_template
            WHERE ticker = $1;
            """
            
            row = await db.fetchrow(query, ticker)
            
            # Debug output to check what's returned
            cls.logger.debug(f"Row returned from database: {row}")
            
            if not row:
                return {
                    "ticker": ticker,
                    "exists": True,
                    "count": 0,
                    "min_date": None,
                    "max_date": None,
                    "avg_volume": None
                }
            
            # Safely extract values from the row
            count = row['row_count'] if 'row_count' in row else 0
            min_date = row['min_date'] if 'min_date' in row else None
            max_date = row['max_date'] if 'max_date' in row else None
            avg_volume = row['avg_volume'] if 'avg_volume' in row else None
            
            result = {
                "ticker": ticker,
                "exists": True,
                "count": count,
                "min_date": min_date,
                "max_date": max_date,
                "avg_volume": avg_volume
            }
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            log_db_success(f"Get ticker statistics for {ticker}", duration_ms, cls.logger)
            
            return result
        except Exception as e:
            # If we get here, there was a problem accessing the row
            log_db_error(f"Get ticker statistics for {ticker}", e, cls.logger)
            
            # Return a safe default
            return {
                "ticker": ticker,
                "exists": True,
                "count": 0,
                "min_date": None,
                "max_date": None,
                "avg_volume": None
            } 