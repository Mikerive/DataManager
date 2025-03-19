import logging
import traceback
import asyncio
from typing import Optional, Callable, Any, TypeVar, Dict, List
from functools import wraps
import time
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values as pg_execute_values

logger = logging.getLogger(__name__)

T = TypeVar('T')

async def safe_pool_execute(pool, query_or_callable, *args, **kwargs):
    """
    Execute a query safely, handling connection errors.
    
    Args:
        pool: Database connection pool
        query_or_callable: Either an SQL query string or a callable that takes a connection
        *args: Arguments for the query
        **kwargs: Keyword arguments for the query
        
    Returns:
        Result of the query execution
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if pool is None or pool._closed:
                raise ValueError("Database pool is closed or None")
                
            async with pool.acquire() as conn:
                if callable(query_or_callable):
                    return await query_or_callable(conn)
                else:
                    return await conn.execute(query_or_callable, *args, **kwargs)
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 0.5 * (2 ** retry_count)  # Exponential backoff
                logger.warning(f"Database operation failed, retrying in {wait_time:.1f}s: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                # Final attempt failed
                logger.error(f"Database operation failed after {max_retries} attempts: {str(e)}")
                raise
                
def format_error_details(e: Exception) -> Dict[str, Any]:
    """
    Format exception details for logging and monitoring.
    
    Args:
        e: The exception to format
        
    Returns:
        Dictionary with formatted error details
    """
    details = {
        'error_type': type(e).__name__,
        'message': str(e),
        'timestamp': datetime.now(),
        'traceback': traceback.format_exc()
    }
    
    # Add additional details for specific error types
    if hasattr(e, 'pgcode'):
        details['pg_code'] = e.pgcode
        
    if hasattr(e, 'pgerror'):
        details['pg_error'] = e.pgerror
        
    return details

def log_db_error(operation_name: str, e: Exception, logger=None):
    """
    Log database error with appropriate details.
    
    Args:
        operation_name: Name of the operation that failed
        e: The exception that occurred
        logger: Optional logger instance (uses module logger if None)
    """
    log = logger or logging.getLogger(__name__)
    error_details = format_error_details(e)
    
    log.error(f"{operation_name} failed: {error_details['message']}")
    log.debug(error_details['traceback'])
    
    return error_details

def log_db_success(operation_name: str, duration_ms: float, logger=None):
    """
    Log successful database operation.
    
    Args:
        operation_name: Name of the operation
        duration_ms: Duration in milliseconds
        logger: Optional logger instance (uses module logger if None)
    """
    log = logger or logging.getLogger(__name__)
    
    # Only log operations that take significant time
    if duration_ms > 100:
        log.debug(f"{operation_name} completed in {duration_ms:.2f}ms")

async def get_tables_info(db, include_statistics: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about database tables.
    
    Args:
        db: Database instance to use for queries
        include_statistics: Whether to include detailed statistics for data tables
        
    Returns:
        Dictionary of table information indexed by table name
    """
    logger.info("Getting information about database tables")
    
    try:
        # Get all tables from the database
        query = """
        SELECT tablename 
        FROM pg_catalog.pg_tables 
        WHERE schemaname = 'public'
        ORDER BY tablename;
        """
        
        tables = await db.fetch(query)
            
        tables_info = {}
        for table in tables:
            table_name = table['tablename']
            # Get row count for each table
            try:
                count_query = f"SELECT COUNT(*) FROM {table_name};"
                count = await db.fetchval(count_query)
                
                has_price_data = False
                # Check if it's a price data table (has OHLCV columns)
                if table_name.startswith('raw_data_') or table_name.startswith('processed_'):
                    schema_query = f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                    """
                    columns = await db.fetch(schema_query)
                    
                    column_names = [col['column_name'] for col in columns]
                    required_columns = ['open', 'high', 'low', 'close']
                    has_price_data = all(col in column_names for col in required_columns)
                
                tables_info[table_name] = {
                    'row_count': count,
                    'has_price_data': has_price_data
                }
            except Exception as e:
                logger.error(f"Error getting info for table {table_name}: {str(e)}")
                tables_info[table_name] = {
                    'row_count': 0,
                    'has_price_data': False
                }
        
        # Get raw data tables
        raw_data_tables = [t for t in tables_info.keys() if t.startswith('raw_data_') and t != 'raw_data_template']
        
        # If statistics are requested and we have a RawData module available, get detailed stats
        if include_statistics and raw_data_tables:
            try:
                # Only import RawData if we need it
                from backend.db.RawData import RawData
                
                # For each table, get more detailed statistics
                for table in raw_data_tables:
                    try:
                        ticker = table.replace('raw_data_', '')
                        logger.debug(f"Getting statistics for {ticker} from table {table}")
                        stats = await RawData.get_ticker_statistics(ticker)
                        
                        if stats is not None:
                            tables_info[table]['statistics'] = stats
                        else:
                            logger.warning(f"No statistics returned for {ticker}")
                            tables_info[table]['statistics'] = {
                                "ticker": ticker,
                                "exists": True,
                                "count": tables_info[table].get('row_count', 0),
                                "min_date": None,
                                "max_date": None,
                                "avg_volume": None
                            }
                    except Exception as e:
                        logger.error(f"Error getting statistics for {table}: {str(e)}")
                        tables_info[table]['statistics'] = {
                            "ticker": table.replace('raw_data_', ''),
                            "exists": True,
                            "count": tables_info[table].get('row_count', 0),
                            "min_date": None,
                            "max_date": None,
                            "avg_volume": None
                        }
            except ImportError:
                logger.warning("RawData module not available, skipping detailed statistics")
        
        return tables_info
    except Exception as e:
        log_db_error("Getting tables info", e, logger)
        return {}

def execute_values(conn, query: str, values: List[tuple], page_size: int = 1000):
    """
    Execute a batch insert query using psycopg2's execute_values.
    
    Args:
        conn: Database connection
        query: SQL query with %s placeholder for values
        values: List of value tuples to insert
        page_size: Number of rows to insert per batch
        
    Returns:
        Number of rows inserted
    """
    try:
        cursor = conn.cursor()
        pg_execute_values(cursor, query, values, page_size=page_size)
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        conn.rollback()
        log_db_error(f"Execute query:\n{query}", e, logger)
        raise 