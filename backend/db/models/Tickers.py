from datetime import datetime
import time
from typing import List, Tuple, Dict, Optional, Any
import logging
import pandas as pd
from backend.db.Database import Database
from backend.db.utils.db_utils import log_db_error, log_db_success


class Tickers:
    """
    Class for managing ticker symbols and their metadata.
    Provides direct CRUD operations on ticker data.
    """
    
    _db = None
    logger = logging.getLogger(__name__)
    
    @classmethod
    async def _get_db(cls):
        """Get or create a Database instance for this class."""
        if cls._db is None or cls._db.pool is None or cls._db.pool._closed:
            cls._db = Database(owner_name="Tickers")
            await cls._db.connect()
        return cls._db
    
    @classmethod
    async def create_tickers_table(cls):
        """Create the tickers table if it doesn't exist."""
        try:
            start_time = time.time()
            db = await cls._get_db()
            query = """
            CREATE TABLE IF NOT EXISTS tickers (
                id SERIAL PRIMARY KEY,
                ticker TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                exchange TEXT NOT NULL,
                asset_type TEXT NOT NULL CHECK (asset_type IN ('Stock', 'ETF')),
                ipo_date TIMESTAMPTZ,
                delisting_date TIMESTAMPTZ,
                status TEXT NOT NULL CHECK (status IN ('active', 'delisted'))
            );
            """
            await db.execute_query(query)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Create tickers table", duration_ms, cls.logger)
            
            return True
        except Exception as e:
            log_db_error("Create tickers table", e, cls.logger)
            return False

    @classmethod
    async def add_ticker(cls,
        ticker: str,
        name: str,
        exchange: str,
        asset_type: str,
        ipo_date: datetime,
        delisting_date: datetime = None,
        status: str = "active"
    ) -> int:
        """
        Add or update a ticker in the database.
        
        Args:
            ticker: Symbol for the ticker (e.g., 'AAPL') - will be stored as UPPERCASE
            name: Full name of the company or instrument
            exchange: Exchange the ticker is listed on
            asset_type: Type of asset ('Stock' or 'ETF')
            ipo_date: Initial public offering date
            delisting_date: Date when the ticker was delisted (if applicable)
            status: Current status ('active' or 'delisted')
            
        Returns:
            The id of the inserted or updated ticker
        """
        try:
            start_time = time.time()
            db = await cls._get_db()
            await cls.create_tickers_table()  # Ensure table exists before inserting

            # Convert ticker to uppercase for consistency
            ticker = ticker.upper()

            query = """
            INSERT INTO tickers (ticker, name, exchange, asset_type, ipo_date, delisting_date, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (ticker) DO UPDATE SET 
                name = EXCLUDED.name,
                exchange = EXCLUDED.exchange,
                asset_type = EXCLUDED.asset_type,
                ipo_date = EXCLUDED.ipo_date,
                delisting_date = EXCLUDED.delisting_date,
                status = EXCLUDED.status
            RETURNING id;
            """

            ticker_id = await db.fetchval(
                query,
                ticker,
                name,
                exchange,
                asset_type,
                ipo_date,
                delisting_date,
                status,
            )
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Add ticker {ticker}", duration_ms, cls.logger)
            
            return ticker_id
        except Exception as e:
            log_db_error(f"Add ticker {ticker}", e, cls.logger)
            return None

    @classmethod
    async def add_tickers_from_dataframe(cls, df: pd.DataFrame) -> List[int]:
        """
        Add multiple tickers from a pandas DataFrame.
        
        Args:
            df: DataFrame with columns: ticker, name, exchange, asset_type, 
                ipo_date, delisting_date (optional), status (optional)
        
        Returns:
            List of inserted/updated ticker IDs
        """
        try:
            start_time = time.time()
            
            if df.empty:
                return []  # No tickers to insert
                
            # Ensure required columns exist
            required_columns = ['ticker', 'name', 'exchange', 'asset_type', 'ipo_date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
                
            # Set default values for optional columns if missing
            if 'delisting_date' not in df.columns:
                df['delisting_date'] = None
            if 'status' not in df.columns:
                df['status'] = 'active'
                
            # Convert to list of tuples for batch insertion
            tickers_data = [
                (
                    row['ticker'],
                    row['name'],
                    row['exchange'],
                    row['asset_type'],
                    row['ipo_date'],
                    row['delisting_date'],
                    row['status']
                )
                for _, row in df.iterrows()
            ]
            
            result = await cls.add_tickers_batch(tickers_data)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Add tickers from dataframe ({len(result)} tickers)", duration_ms, cls.logger)
            
            return result
        except Exception as e:
            log_db_error("Add tickers from dataframe", e, cls.logger)
            return []

    @classmethod
    async def add_tickers_batch(cls, 
        tickers: List[Tuple[str, str, str, str, datetime, Optional[datetime], str]]
    ) -> List[int]:
        """
        Add multiple tickers to the database in a single batch operation.
        
        Args:
            tickers: List of ticker tuples, each containing:
                     (ticker, name, exchange, asset_type, ipo_date, delisting_date, status)
        
        Returns:
            The list of inserted/updated ticker IDs
        """
        try:
            start_time = time.time()
            
            if not tickers:
                return []  # No tickers to insert

            db = await cls._get_db()
            await cls.create_tickers_table()  # Ensure table exists before inserting

            query = """
            INSERT INTO tickers (ticker, name, exchange, asset_type, ipo_date, delisting_date, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (ticker) DO UPDATE SET 
                name = EXCLUDED.name,
                exchange = EXCLUDED.exchange,
                asset_type = EXCLUDED.asset_type,
                ipo_date = EXCLUDED.ipo_date,
                delisting_date = EXCLUDED.delisting_date,
                status = EXCLUDED.status
            RETURNING id;
            """

            # Execute batch insert and collect returned IDs
            results = await db.executemany(query, tickers)
            ids = [result[0] for result in results] if results else []
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Add tickers batch ({len(ids)} tickers)", duration_ms, cls.logger)
            
            return ids
        except Exception as e:
            log_db_error("Add tickers batch", e, cls.logger)
            return []

    @classmethod
    async def get_ticker(cls, ticker: str) -> Dict[str, Any]:
        """
        Get details for a specific ticker.
        
        Args:
            ticker: The ticker symbol to look up (will be converted to uppercase)
            
        Returns:
            A dictionary with ticker details, or None if not found
        """
        try:
            start_time = time.time()
            db = await cls._get_db()
            
            # Ensure uppercase for consistent lookup
            ticker = ticker.upper()
            
            query = "SELECT * FROM tickers WHERE ticker = $1;"
            
            result = await db.fetchrow(query, ticker)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Get ticker {ticker}", duration_ms, cls.logger)
            
            return dict(result) if result else None
        except Exception as e:
            log_db_error(f"Get ticker {ticker}", e, cls.logger)
            return None

    @classmethod
    async def delete_ticker(cls, ticker: str) -> bool:
        """
        Delete a ticker from the database.
        
        Args:
            ticker: The ticker symbol to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            start_time = time.time()
            db = await cls._get_db()
            query = "DELETE FROM tickers WHERE ticker = $1 RETURNING id;"
            
            result = await db.fetchval(query, ticker)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Delete ticker {ticker}", duration_ms, cls.logger)
            
            return result is not None
        except Exception as e:
            log_db_error(f"Delete ticker {ticker}", e, cls.logger)
            return False

    @classmethod
    async def get_all_tickers(cls, as_dataframe: bool = False) -> pd.DataFrame:
        """
        Get all tickers from the database.
        
        Args:
            as_dataframe: If True, returns results as a pandas DataFrame
            
        Returns:
            List of ticker dictionaries or DataFrame with ticker data
        """
        try:
            start_time = time.time()
            db = await cls._get_db()
            query = "SELECT * FROM tickers ORDER BY ticker;"
            
            results = await db.fetch(query)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Get all tickers", duration_ms, cls.logger)
            
            if not results:
                return pd.DataFrame() if as_dataframe else []
                
            if as_dataframe:
                return pd.DataFrame([dict(r) for r in results])
            else:
                return [dict(r) for r in results]
        except Exception as e:
            log_db_error("Get all tickers", e, cls.logger)
            return pd.DataFrame() if as_dataframe else []
    
    @classmethod
    async def get_ticker_id(cls, ticker: str) -> Optional[int]:
        """
        Get the ID for a specific ticker symbol.
        
        Args:
            ticker: The ticker symbol to look up
            
        Returns:
            The ticker ID, or None if not found
        """
        try:
            start_time = time.time()
            db = await cls._get_db()
            query = "SELECT id FROM tickers WHERE ticker = $1;"
            
            result = await db.fetchval(query, ticker)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Get ticker ID for {ticker}", duration_ms, cls.logger)
            
            return result
        except Exception as e:
            log_db_error(f"Get ticker ID for {ticker}", e, cls.logger)
            return None
    
    @classmethod
    async def update_ticker_status(cls, ticker: str, status: str, delisting_date: Optional[datetime] = None) -> bool:
        """
        Update a ticker's status and optionally its delisting date.
        
        Args:
            ticker: The ticker symbol to update
            status: New status ('active' or 'delisted')
            delisting_date: Date when the ticker was delisted (required if status is 'delisted')
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            start_time = time.time()
            
            if status not in ['active', 'delisted']:
                raise ValueError("Status must be either 'active' or 'delisted'")
                
            if status == 'delisted' and delisting_date is None:
                raise ValueError("Delisting date is required when setting status to 'delisted'")
                
            db = await cls._get_db()
            
            query = """
            UPDATE tickers 
            SET status = $2, delisting_date = $3
            WHERE ticker = $1
            RETURNING id;
            """
            
            result = await db.fetchval(query, ticker, status, delisting_date)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Update ticker status for {ticker}", duration_ms, cls.logger)
            
            return result is not None
        except Exception as e:
            log_db_error(f"Update ticker status for {ticker}", e, cls.logger)
            return False
    
    @classmethod
    async def search_tickers(cls, 
                          search_term: str = None, 
                          exchange: str = None, 
                          asset_type: str = None,
                          status: str = None,
                          as_dataframe: bool = False) -> pd.DataFrame:
        """
        Search for tickers based on various criteria.
        
        Args:
            search_term: Text to search in ticker symbol or name (case insensitive)
            exchange: Filter by specific exchange
            asset_type: Filter by asset type ('Stock' or 'ETF')
            status: Filter by status ('active' or 'delisted')
            as_dataframe: If True, returns results as a pandas DataFrame
            
        Returns:
            List of ticker dictionaries or DataFrame with ticker data
        """
        try:
            start_time = time.time()
            db = await cls._get_db()
            
            conditions = []
            params = []
            
            if search_term:
                conditions.append("(LOWER(ticker) LIKE $1 OR LOWER(name) LIKE $1)")
                params.append(f'%{search_term.lower()}%')
                
            if exchange:
                conditions.append(f"exchange = ${len(params) + 1}")
                params.append(exchange)
                
            if asset_type:
                if asset_type not in ['Stock', 'ETF']:
                    raise ValueError("Asset type must be either 'Stock' or 'ETF'")
                conditions.append(f"asset_type = ${len(params) + 1}")
                params.append(asset_type)
                
            if status:
                if status not in ['active', 'delisted']:
                    raise ValueError("Status must be either 'active' or 'delisted'")
                conditions.append(f"status = ${len(params) + 1}")
                params.append(status)
                
            query = "SELECT * FROM tickers"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY ticker;"
            
            results = await db.fetch(query, *params)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Search tickers", duration_ms, cls.logger)
            
            if not results:
                return pd.DataFrame() if as_dataframe else []
                
            if as_dataframe:
                return pd.DataFrame([dict(r) for r in results])
            else:
                return [dict(r) for r in results]
        except Exception as e:
            log_db_error("Search tickers", e, cls.logger)
            return pd.DataFrame() if as_dataframe else []
            
    @classmethod
    async def close_connection(cls):
        """Close the database connection if it exists."""
        if cls._db is not None:
            await cls._db.close()
            cls._db = None 