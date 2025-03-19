"""ProcessedData model for storing processed bar data."""
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from backend.db.models.Tickers import Tickers
from backend.db.Database import Database
import numpy as np
import logging
import re


class ProcessedData:
    """
    Class for managing processed price data organized by ticker and bar type.
    Provides direct CRUD operations and batch functionality with DataFrames.
    """

    TABLE_TEMPLATE = "processed_data_template"
    logger = logging.getLogger(__name__)
    _db = None
    
    @classmethod
    async def _get_db(cls):
        """Get or create a Database instance for this class."""
        if cls._db is None or cls._db.pool is None or cls._db.pool._closed:
            cls._db = Database()
            await cls._db.connect()
        return cls._db
    
    @classmethod
    async def create_template_table(cls):
        """Create the processed_data_template table if it doesn't exist."""
        db = await cls._get_db()
        
        query = f"""
        CREATE TABLE IF NOT EXISTS {cls.TABLE_TEMPLATE} (
            id SERIAL,
            ticker_id INTEGER NOT NULL,
            bar_type VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            start_time TIMESTAMP WITH TIME ZONE NOT NULL,
            end_time TIMESTAMP WITH TIME ZONE NOT NULL,
            open DECIMAL(16, 6),
            high DECIMAL(16, 6),
            close DECIMAL(16, 6),
            low DECIMAL(16, 6),
            volume BIGINT,
            num_trades INTEGER,
            ratio DECIMAL(16, 6),
            PRIMARY KEY (id, ticker_id, bar_type)
        );
        """
        
        try:
            await db.execute_query(query)
        except Exception as e:
            cls.logger.error(f"Error creating template table: {str(e)}")
            raise

    @classmethod
    async def create_ticker_partition(cls, ticker_id: int, bar_type: str) -> bool:
        """
        Create a partition table for a specific ticker_id and bar_type combination.
        
        Args:
            ticker_id: The ticker ID 
            bar_type: Type of bar ('volume', 'tick', 'time', etc.)
            
        Returns:
            True if created, False if already exists
        """
        # Ensure template table exists
        await cls.create_template_table()
        
        db = await cls._get_db()
        table_name = f"processed_data_{ticker_id}_{bar_type}"

        # First check if the table exists
        check_query = f"""
        SELECT EXISTS (
            SELECT FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relname = '{table_name}'
        );
        """
        
        exists = await db.fetchval(check_query)
        if exists:
            return True

        # Create the table if it doesn't exist
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            CHECK (ticker_id = {ticker_id} AND bar_type = '{bar_type}')
        ) INHERITS ({cls.TABLE_TEMPLATE});

        CREATE INDEX idx_{table_name}_time ON {table_name} (timestamp DESC);
        """

        try:
            await db.execute_query(query)
            return True
        except Exception as e:
            cls.logger.error(f"Error creating partition {table_name}: {str(e)}")
            raise

    @classmethod
    async def delete_ticker_partition(cls, ticker_id: int, bar_type: str) -> bool:
        """
        Delete a processed data partition for a specific ticker and bar type.
        
        Args:
            ticker_id: The ticker ID
            bar_type: Type of bar 
            
        Returns:
            True if deleted, False if doesn't exist
        """
        db = await cls._get_db()
        table_name = f"processed_data_{ticker_id}_{bar_type}"

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

        query = f"""
        DROP TABLE IF EXISTS {table_name} CASCADE;
        """

        try:
            await db.execute_query(query)
            return True
        except Exception as e:
            cls.logger.error(f"Error deleting partition {table_name}: {str(e)}")
            raise

    @classmethod
    async def add_bar_datapoint(
        cls,
        ticker_id: int,
        timestamp: datetime,
        bar_type: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        start_time: datetime = None,
        end_time: datetime = None,
        num_trades: int = 0,
        ratio: float = None
    ):
        """
        Add a single processed data entry for a ticker.
        
        Args:
            ticker_id: The ticker ID
            timestamp: Timestamp of the processed bar
            bar_type: Type of bar ('volume', 'tick', 'time', etc.)
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            volume: Trading volume
            start_time: Start timestamp of the bar period (defaults to timestamp)
            end_time: End timestamp of the bar period (defaults to timestamp)
            num_trades: Number of trades in this bar
            ratio: Ratio/threshold value for this bar type
            
        Returns:
            True if successfully added
        """
        db = await cls._get_db()
        await cls.create_template_table()
        await cls.create_ticker_partition(ticker_id, bar_type)

        # Default start_time and end_time to timestamp if not provided
        if start_time is None:
            start_time = timestamp
        if end_time is None:
            end_time = timestamp

        table_name = f"processed_data_{ticker_id}_{bar_type}"

        query = f"""
        INSERT INTO {table_name} (ticker_id, timestamp, bar_type, start_time, end_time, open, high, low, close, volume, num_trades, ratio)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (id, ticker_id, bar_type) DO UPDATE SET
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            num_trades = EXCLUDED.num_trades,
            ratio = EXCLUDED.ratio;
        """

        try:
            await db.execute_query(
                query,
                ticker_id,
                timestamp,
                bar_type,
                start_time,
                end_time,
                open_price,
                high,
                low,
                close,
                volume,
                num_trades,
                ratio
            )
            return True
        except Exception as e:
            cls.logger.error(f"Error adding processed data for ticker_id {ticker_id}, bar_type {bar_type}: {str(e)}")
            raise

    @classmethod
    async def get_processed_data(cls, 
                              ticker_id: int, 
                              bar_type: str,
                              start_date: Optional[datetime] = None, 
                              end_date: Optional[datetime] = None,
                              limit: int = 10000) -> pd.DataFrame:
        """
        Get processed data for a ticker and bar type with optional date filtering.
        
        Args:
            ticker_id: The ticker ID
            bar_type: Type of bar
            start_date: Optional start datetime to filter data
            end_date: Optional end datetime to filter data
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with processed bar data
        """
        db = await cls._get_db()
        
        # Check if the partition exists
        table_name = f"processed_data_{ticker_id}_{bar_type}"
        
        check_query = f"""
        SELECT EXISTS (
            SELECT FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relname = '{table_name}'
        );
        """
        
        exists = await db.fetchval(check_query)
        if not exists:
            return pd.DataFrame()
        
        # First get the schema to determine columns
        schema = await cls._get_table_schema(table_name)
        column_names = [col[0] for col in schema]
        
        # Build column list for query
        columns_str = ", ".join(column_names)
        
        # Base query
        if start_date and end_date:
            query = f"""
            SELECT {columns_str}
            FROM {table_name}
            WHERE timestamp BETWEEN $1 AND $2
            ORDER BY timestamp
            LIMIT {limit};
            """
            params = [start_date, end_date]
        else:
            query = f"""
            SELECT {columns_str}
            FROM {table_name}
            ORDER BY timestamp
            LIMIT {limit};
            """
            params = []
        
        try:
            if params:
                rows = await db.fetch(query, *params)
            else:
                rows = await db.fetch(query)
                
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(r) for r in rows])
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            return df
        except Exception as e:
            cls.logger.error(f"Error getting processed data for {table_name}: {str(e)}")
            return pd.DataFrame()
    
    @classmethod
    async def _get_table_schema(cls, table_name: str) -> List[Tuple[str, str]]:
        """Get the schema of a table."""
        db = await cls._get_db()
        
        query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position;
        """
        
        try:
            columns = await db.fetch(query, table_name)
            return [(col['column_name'], col['data_type']) for col in columns]
        except Exception as e:
            cls.logger.error(f"Error getting schema for {table_name}: {str(e)}")
            return []
                
    @classmethod
    async def create_processed_data_table(cls, ticker: str, bar_type: str):
        """
        Create a processed data table for a specific ticker and bar type.
        
        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            bar_type: Type of bar (e.g., 'volume', 'tick', 'time')
            
        Returns:
            True if created, False if already exists
        """
        db = await cls._get_db()
        
        # Sanitize ticker for table name
        safe_ticker = re.sub(r'[^a-zA-Z0-9]', '_', ticker.upper())
        table_name = f"processed_data_{safe_ticker}_{bar_type}"
        
        # Using DO $$ BEGIN/END $$ to make this idempotent
        query = f"""
        DO $$ 
        DECLARE
            new_table_created BOOLEAN;
        BEGIN
            new_table_created := FALSE;
            -- Check if the table already exists
            IF NOT EXISTS (
                SELECT FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = '{table_name}'
            ) THEN
                -- Create the table
                CREATE TABLE {table_name} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    open FLOAT NOT NULL,
                    high FLOAT NOT NULL,
                    low FLOAT NOT NULL,
                    close FLOAT NOT NULL,
                    volume FLOAT NOT NULL,
                    PRIMARY KEY (timestamp)
                );
                
                -- Create an index for better query performance
                CREATE INDEX idx_{table_name}_time ON {table_name} (timestamp DESC);
                
                new_table_created := TRUE;
                RAISE NOTICE 'Created processed data table {table_name}';
            ELSE
                RAISE NOTICE 'Processed data table {table_name} already exists';
            END IF;
            
            RETURN new_table_created;
        EXCEPTION
            WHEN duplicate_table THEN
                RAISE NOTICE 'Processed data table {table_name} already exists';
                RETURN FALSE;
        END $$;
        """

        try:
            await db.execute_query(query)
            
            # Check if the table exists now
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = '{table_name}'
            );
            """
            
            exists = await db.fetchval(check_query)
            return exists
        except Exception as e:
            cls.logger.error(f"Error creating processed data table {table_name}: {str(e)}")
            return False
                
    @classmethod
    async def store_dataframe(cls, ticker: str, bar_type: str, df: pd.DataFrame) -> int:
        """
        Store processed bar data for a ticker from a DataFrame.
        
        Args:
            ticker: Ticker symbol
            bar_type: Type of bar (e.g., 'volume', 'tick', 'time')
            df: DataFrame with processed data (must have timestamp, open, high, low, close, volume)
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            cls.logger.warning(f"Empty DataFrame provided for {ticker} {bar_type}, nothing to insert")
            return 0
        
        db = await cls._get_db()
        
        # Create the table if it doesn't exist
        created = await cls.create_processed_data_table(ticker, bar_type)
        if not created:
            cls.logger.info(f"Using existing table for {ticker} {bar_type}")
        
        # Prepare the table name
        safe_ticker = re.sub(r'[^a-zA-Z0-9]', '_', ticker.upper())
        table_name = f"processed_data_{safe_ticker}_{bar_type}"
        
        # Ensure DataFrame has required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")
        
        # Reset index if timestamp is the index
        if df.index.name == 'timestamp':
            df = df.reset_index()
        
        # Process data in batches
        BATCH_SIZE = 1000
        total_rows = len(df)
        rows_processed = 0
        
        for i in range(0, total_rows, BATCH_SIZE):
            batch = df.iloc[i:i+BATCH_SIZE]
            
            # Prepare batch insert
            values_placeholders = []
            params = []
            
            for _, row in batch.iterrows():
                values_placeholders.append("($" + str(len(params) + 1) + ", $" + 
                                         str(len(params) + 2) + ", $" + 
                                         str(len(params) + 3) + ", $" + 
                                         str(len(params) + 4) + ", $" + 
                                         str(len(params) + 5) + ", $" + 
                                         str(len(params) + 6) + ")")
                params.extend([
                    row['timestamp'],
                    float(row['open']), 
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                ])
            
            # If we have no values to insert, skip this batch
            if not values_placeholders:
                continue
            
            # Build and execute the query
            values_part = ", ".join(values_placeholders)
            query = f"""
            INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
            VALUES {values_part}
            ON CONFLICT (timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume;
            """
            
            try:
                await db.execute_query(query, *params)
                rows_processed += len(batch)
            except Exception as e:
                cls.logger.error(f"Error inserting batch for {ticker} ({bar_type}): {str(e)}")
                # Continue with next batch instead of failing completely
                
        cls.logger.info(f"Successfully stored {rows_processed} processed data points for {ticker} ({bar_type})")
        return rows_processed
        
    @classmethod
    async def delete_ticker_data(cls, ticker: str, bar_type: str = None) -> int:
        """
        Delete processed data for a ticker, optionally for a specific bar type.
        
        Args:
            ticker: Ticker symbol
            bar_type: Optional bar type to delete. If None, deletes all bar types for the ticker.
            
        Returns:
            Number of tables deleted
        """
        db = await cls._get_db()
        
        # Sanitize ticker for table name
        safe_ticker = re.sub(r'[^a-zA-Z0-9]', '_', ticker.upper())
        tables_deleted = 0
        
        if bar_type:
            # Delete specific bar type
            table_name = f"processed_data_{safe_ticker}_{bar_type}"
            
            # Check if table exists
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relname = '{table_name}'
            );
            """
            
            exists = await db.fetchval(check_query)
            if not exists:
                return 0
                
            query = f"""
            DROP TABLE IF EXISTS {table_name};
            """
            
            await db.execute_query(query)
            cls.logger.info(f"Deleted processed data table {table_name}")
            tables_deleted = 1
        else:
            # Delete all bar types for the ticker
            # First get a list of tables matching the pattern
            query = """
            SELECT tablename FROM pg_catalog.pg_tables 
            WHERE schemaname = 'public' AND tablename LIKE $1;
            """
            
            pattern = f"processed_data_{safe_ticker}_%"
            
            tables = await db.fetch(query, pattern)
            
            for table in tables:
                table_name = table['tablename']
                await db.execute_query(f"DROP TABLE IF EXISTS {table_name};")
                cls.logger.info(f"Deleted processed data table {table_name}")
                tables_deleted += 1
                
        return tables_deleted
                    
    @classmethod
    async def list_available_data(cls, ticker: str = None) -> Dict[str, List[str]]:
        """
        List processed data tables, optionally filtered by ticker.
        
        Args:
            ticker: Optional ticker symbol to filter by
            
        Returns:
            Dictionary mapping tickers to available bar types
        """
        db = await cls._get_db()
        
        if ticker:
            # Filter by ticker
            safe_ticker = re.sub(r'[^a-zA-Z0-9]', '_', ticker.upper())
            pattern = f"processed_data_{safe_ticker}_%"
        else:
            # All processed data tables
            pattern = "processed_data_%"
            
        query = """
        SELECT tablename FROM pg_catalog.pg_tables 
        WHERE schemaname = 'public' AND tablename LIKE $1
        AND tablename != 'processed_data_template'
        ORDER BY tablename;
        """
        
        tables = await db.fetch(query, pattern)
        
        # Organize by ticker
        result = {}
        for table in tables:
            table_name = table['tablename']
            # Skip the template
            if table_name == 'processed_data_template':
                continue
                
            # Extract ticker and bar_type from table name
            parts = table_name.split('_')
            if len(parts) >= 3:
                # Handle tables named by ticker ID
                if parts[2].isdigit():
                    # This is a table using ticker_id format
                    ticker_id = int(parts[2])
                    bar_type = '_'.join(parts[3:])  # Join remaining parts for bar_type
                    
                    # Try to get ticker symbol if possible
                    try:
                        # This would require implementing a reverse lookup
                        # For now, just use the ID
                        ticker_key = f"ticker_id_{ticker_id}"
                    except:
                        ticker_key = f"ticker_id_{ticker_id}"
                else:
                    # This is a table using ticker symbol format
                    ticker_key = parts[2]
                    bar_type = '_'.join(parts[3:])  # Join remaining parts for bar_type
                
                if ticker_key not in result:
                    result[ticker_key] = []
                
                result[ticker_key].append(bar_type)
                
        return result

    @classmethod
    async def add_dataframe_by_id(cls, ticker_id: int, bar_type: str, df: pd.DataFrame, ratio: float = None) -> int:
        """
        Add a batch of processed data from a pandas DataFrame by ticker ID.
        
        Args:
            ticker_id (int): The ticker ID
            bar_type (str): Type of bar ('volume', 'tick', 'entropy', etc.)
            df (pd.DataFrame): DataFrame with columns: timestamp, open, high, low, close, volume
                               (and optionally start_time, end_time, num_trades)
            ratio (float, optional): The ratio/threshold value for this bar type
            
        Returns:
            Number of rows processed
        """
        if df.empty:
            return 0  # No data to insert
            
        db = await cls._get_db()
        await cls.create_template_table()
        await cls.create_ticker_partition(ticker_id, bar_type)

        table_name = f"processed_data_{ticker_id}_{bar_type}"
        
        # Handle different DataFrame structures using vectorized operations
        start_times = df.get('start_time', df['timestamp'])
        end_times = df.get('end_time', df['timestamp'])
        num_trades = df.get('num_trades', 0)
        bar_ratios = df.get('ratio', ratio)

        rows_processed = 0
        
        try:
            # Process data in batches to avoid memory issues
            BATCH_SIZE = 1000
            total_rows = len(df)
            cls.logger.info(f"Processing {total_rows} rows for ticker_id {ticker_id}, bar_type {bar_type}")
            
            for i in range(0, total_rows, BATCH_SIZE):
                batch = df.iloc[i:i+BATCH_SIZE]
                cls.logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(total_rows+BATCH_SIZE-1)//BATCH_SIZE}")

                # Create placeholder strings for each row
                batch_size = len(batch)
                values_placeholders = []
                params = []
                
                for j in range(batch_size):
                    values_placeholders.append(f"(${j*12+1}, ${j*12+2}, ${j*12+3}, ${j*12+4}, ${j*12+5}, ${j*12+6}, ${j*12+7}, ${j*12+8}, ${j*12+9}, ${j*12+10}, ${j*12+11}, ${j*12+12})")

                # Add parameters in the same order as the placeholders
                for j, (idx, row) in enumerate(batch.iterrows()):
                    params.extend([
                        ticker_id,
                        bar_type,
                        row['timestamp'],
                        start_times.iloc[j],
                        end_times.iloc[j],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume']),
                        int(num_trades.iloc[j]),
                        float(bar_ratios.iloc[j]) if pd.notna(bar_ratios.iloc[j]) else None
                    ])
                
                # If we have no values to insert, skip this batch
                if not values_placeholders:
                    continue
                    
                query = f"""
                INSERT INTO {table_name} (ticker_id, bar_type, timestamp, start_time, end_time, open, high, low, close, volume, num_trades, ratio)
                VALUES {", ".join(values_placeholders)}
                ON CONFLICT (id, ticker_id, bar_type) DO UPDATE SET
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    num_trades = EXCLUDED.num_trades,
                    ratio = EXCLUDED.ratio;
                """
                
                await db.execute_query(query, *params)
                rows_processed += batch_size
                
            return rows_processed
        except Exception as e:
            cls.logger.error(f"Error batch inserting processed data for ticker_id {ticker_id}, bar_type {bar_type}: {str(e)}")
            raise

    @classmethod
    async def add_dataframe_by_symbol(cls, ticker: str, bar_type: str, df: pd.DataFrame, ratio: float = None) -> int:
        """
        Add a batch of processed data from a pandas DataFrame by ticker symbol.
        First looks up the ticker ID, then calls add_dataframe_by_id.
        
        Args:
            ticker (str): The ticker symbol
            bar_type (str): Type of bar ('volume', 'tick', 'entropy', etc.)
            df (pd.DataFrame): DataFrame with columns: timestamp, open, high, low, close, volume
                               (and optionally start_time, end_time, num_trades)
            ratio (float, optional): The ratio/threshold value for this bar type
            
        Returns:
            Number of rows processed, or -1 if ticker not found
        """
        # First get the ticker ID
        ticker_id = await Tickers.get_ticker_id(ticker)
        if ticker_id is None:
            cls.logger.error(f"Ticker {ticker} not found in the database")
            return -1
            
        # Call the ID-based method
        return await cls.add_dataframe_by_id(ticker_id, bar_type, df, ratio)
    
    @classmethod
    async def get_data_by_symbol(cls, 
                              ticker: str, 
                              bar_type: str,
                              start_date: Optional[datetime] = None, 
                              end_date: Optional[datetime] = None,
                              limit: int = 10000) -> pd.DataFrame:
        """
        Get processed data for a ticker and bar type with optional date filtering.
        First looks up the ticker ID, then calls get_processed_data.
        
        Args:
            ticker (str): The ticker symbol
            bar_type (str): Type of bar
            start_date: Optional start datetime to filter data
            end_date: Optional end datetime to filter data
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with processed bar data
        """
        # First get the ticker ID
        ticker_id = await Tickers.get_ticker_id(ticker)
        if ticker_id is None:
            cls.logger.error(f"Ticker {ticker} not found in the database")
            return pd.DataFrame()
            
        # Call the ID-based method
        return await cls.get_processed_data(ticker_id, bar_type, start_date, end_date, limit)
        
    @classmethod
    async def close_connection(cls):
        """Close the database connection if it exists."""
        if cls._db is not None:
            await cls._db.close()
            cls._db = None