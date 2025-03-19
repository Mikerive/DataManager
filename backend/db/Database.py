import asyncpg
import os
import asyncio
import time
import logging
import socket
import traceback
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional, Callable, TypeVar, Union

from .utils.db_monitor import get_connection_monitor
from .utils.db_utils import safe_pool_execute, log_db_error, log_db_success

load_dotenv()

# Type for generic return value
T = TypeVar('T')

class Database:
    """
    Manages database connections and provides query execution functions.
    Creates and manages a database connection pool for database operations.
    """
    
    # Class variable to track if we're in test mode
    _is_test_environment = False
    
    @classmethod
    def set_test_environment(cls, is_test: bool = True):
        """Set whether we're operating in a test environment."""
        cls._is_test_environment = is_test
        
    @classmethod
    def is_test_environment(cls) -> bool:
        """Check if we're running in a test environment."""
        return cls._is_test_environment or os.getenv('TEST_DB_ENABLED', 'False').lower() == 'true'
    
    def __init__(self, owner_name: str = None, connection_expiration: int = 60, pool_min_size: int = 1, pool_max_size: int = 10,
                 db_host: str = None, db_port: str = None, db_name: str = None, db_user: str = None, db_password: str = None,
                 use_test_db: bool = None, debug_mode: bool = False):
        """Initialize the database connection parameters.
        
        Args:
            owner_name: Name identifying this database connection owner (default: None)
            connection_expiration: Number of seconds after which an idle connection is closed (default: 60)
            pool_min_size: Minimum number of connections in the pool (default: 1)
            pool_max_size: Maximum number of connections in the pool (default: 10)
            db_host: Database host (default: None, will use environment variable)
            db_port: Database port (default: None, will use environment variable)
            db_name: Database name (default: None, will use environment variable)
            db_user: Database user (default: None, will use environment variable)
            db_password: Database password (default: None, will use environment variable)
            use_test_db: Force using test database if True, production if False, or auto-detect if None
            debug_mode: Enable detailed database operation logging (default: False)
        """
        self.pool = None
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode
        
        # Adjust logging level based on debug mode
        if debug_mode:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Database debug mode enabled")
        
        # Determine if we should use test database
        self.use_test_db = use_test_db if use_test_db is not None else self.is_test_environment()
        
        # Store any directly provided connection parameters
        self.db_host_override = db_host
        self.db_port_override = db_port
        self.db_name_override = db_name
        self.db_user_override = db_user
        self.db_password_override = db_password
        
        # Load environment variables with defaults
        self._load_connection_settings()
        
        # Connection management settings
        self.connection_expiration = connection_expiration
        self.last_connection_time = None
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        
        # Owner name for monitoring
        self.owner_name = owner_name or f"db-{id(self)}"
        
        # Setup semaphores for concurrent operations
        self.semaphore = asyncio.Semaphore(1)  # For connection management
        self._operation_semaphore = asyncio.Semaphore(10)  # For DB operations, allowing 10 concurrent
        
        # Setup cleanup task for this instance
        self._cleanup_task = None
        
        # Connection retry settings
        self.max_retries = 5
        self.retry_delay = 5
        self.connection_timeout = 30  # seconds
    
    def _load_connection_settings(self):
        """
        Load database connection settings from environment variables,
        with special handling for local development environment.
        
        If connection parameters were provided directly in the constructor,
        they will override the environment variables.
        """
        # Get regular or test environment variables based on use_test_db
        if self.use_test_db:
            # Load test database settings
            self.logger.info("Using test database connection settings")
            self.db_host = os.getenv("TEST_DB_HOST", "localhost")
            self.db_port = os.getenv("TEST_DB_PORT", "5432")
            self.db_user = os.getenv("TEST_DB_USER", "algotrader")
            self.db_password = os.getenv("TEST_DB_PASSWORD", "algotrader")
            self.db_name = os.getenv("TEST_DB_NAME", "algotrader_test")
        else:
            # Load production database settings
            self.db_host = os.getenv("DB_HOST", "localhost")
            self.db_port = os.getenv("DB_PORT", "5432")
            self.db_user = os.getenv("DB_USER", "algotrader")
            self.db_password = os.getenv("DB_PASSWORD", "algotrader")
            self.db_name = os.getenv("DB_NAME", "algotrader")
        
        # Override with directly provided parameters if any
        if self.db_host_override is not None:
            self.db_host = self.db_host_override
        if self.db_port_override is not None:
            self.db_port = self.db_port_override
        if self.db_name_override is not None:
            self.db_name = self.db_name_override
        if self.db_user_override is not None:
            self.db_user = self.db_user_override
        if self.db_password_override is not None:
            self.db_password = self.db_password_override
        
        # Validate required settings
        if not self.db_password:
            raise ValueError("Database password is required")
        if not self.db_user:
            raise ValueError("Database user is required")
        if not self.db_name:
            raise ValueError("Database name is required")
        
        # Check if we're running in a local development environment
        is_local = self._is_local_environment()
        
        # In local environment, override host to localhost if necessary
        if is_local:
            # Check if we're trying to connect to a Docker service by name
            docker_service_names = ["db", "postgres", "database", "mysql", "mongo", "redis", "db_test"]
            if self.db_host in docker_service_names and not self.db_host_override:
                self.logger.info(f"Local environment detected, converting Docker service name '{self.db_host}' to localhost")
                self.db_host = "localhost"
                
                # If connecting to a service with a mapped port, adjust the port
                if self.db_host == "db_test" and self.db_port == "5432":
                    self.logger.info("Adjusting port for db_test service from 5432 to 5433 (likely mapped port)")
                    self.db_port = "5433"
            elif self.db_host != "localhost" and not self.db_host_override:
                self.logger.info("Local environment detected, using localhost for database connection")
                self.db_host = "localhost"
    
    def _is_local_environment(self):
        """
        Determine if we're running in a local development environment.
        Returns True if local, False otherwise.
        """
        # Method 1: Check for environment variable flag
        if os.getenv("ENVIRONMENT") == "development" or os.getenv("DEBUG") == "True":
            return True
            
        # Method 2: Check for common development hostnames
        hostname = socket.gethostname().lower()
        dev_indicators = ["local", "dev", "development", "laptop", "desktop", "home"]
        for indicator in dev_indicators:
            if indicator in hostname:
                return True
        
        # Method 3: Check if we're running in a common development location
        current_dir = os.getcwd()
        dev_paths = ["projects", "repos", "development", "src", "algotrader"]
        for path in dev_paths:
            if path.lower() in current_dir.lower():
                return True
                
        # Method 4: Check if we're running outside Docker
        # Docker containers usually have hostnames that are their container IDs
        if not any(c for c in hostname if not (c.isalnum() or c == '-')):
            # If the host is a Docker service name but we're not in Docker
            docker_services = ["db", "postgres", "database", "mysql", "mongo", "redis"]
            if self.db_host in docker_services:
                return True
        
        # Default to treating as non-local environment
        return False

    def get_connection_info(self) -> Dict[str, str]:
        """
        Return information about the current database connection.
        
        Returns:
            Dict with host, port, name, user information
        """
        info = {
            'host': self.db_host,
            'port': self.db_port,
            'name': self.db_name,
            'user': self.db_user,
            'is_test': self.use_test_db
        }
        
        # Add additional debug info if debug mode is enabled
        if self.debug_mode:
            info.update({
                'debug_mode': self.debug_mode,
                'pool_status': 'active' if self.pool and not getattr(self.pool, '_closed', True) else 'closed',
                'min_pool_size': self.pool_min_size,
                'max_pool_size': self.pool_max_size,
                'connection_timeout': self.connection_timeout,
                'last_connection_time': self.last_connection_time,
                'is_local_environment': self._is_local_environment()
            })
            
        return info

    async def _start_cleanup_task(self):
        """Starts the background task that cleans up idle connections."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background task that checks and closes idle connections."""
        try:
            while True:
                # Check if connection has expired
                if (self.pool is not None and self.last_connection_time is not None and 
                    time.time() - self.last_connection_time > self.connection_expiration):
                    self.logger.info(f"Connection expired after {self.connection_expiration} seconds of inactivity. Closing pool.")
                    await self.close()
                
                # Check every 5 seconds
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            self.logger.debug("Cleanup task cancelled")
        except Exception as e:
            self.logger.error(f"Error in cleanup task: {str(e)}")
            if get_connection_monitor():
                get_connection_monitor().log_error(self.owner_name, e, "Cleanup task")

    async def connect(self, retries=None, delay=None):
        """
        Establishes a connection pool with retries if the database is not ready.
        """
        start_time = time.time()
        
        async with self.semaphore:
            # If pool already exists and is not closed, return immediately
            if self.pool is not None and not self.pool._closed:
                self.update_connection_time()
                if self.debug_mode:
                    self.logger.debug(f"Reusing existing connection pool: {self.db_host}:{self.db_port}/{self.db_name}")
                return
            
            # Refresh connection settings to ensure they're current
            self._load_connection_settings()
    
            # Get local environment status for better error messages
            is_local = self._is_local_environment()
            
            # Use provided retries/delay or defaults
            retries = retries if retries is not None else self.max_retries
            delay = delay if delay is not None else self.retry_delay
            
            dsn = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            self.logger.info(f"Connecting to database at {self.db_host}:{self.db_port}/{self.db_name}")
            
            if self.debug_mode:
                self.logger.debug(f"DSN: {dsn.replace(self.db_password, '********')}")
                self.logger.debug(f"Pool settings: min_size={self.pool_min_size}, max_size={self.pool_max_size}, timeout={self.connection_timeout}s")
                self.logger.debug(f"Retry settings: attempts={retries}, delay={delay}s")
    
            last_error = None
            for attempt in range(retries):
                try:
                    if self.debug_mode:
                        self.logger.debug(f"Connection attempt {attempt + 1}/{retries}")
                        
                    # Create connection pool with timeout
                    self.pool = await asyncio.wait_for(
                        asyncpg.create_pool(
                            dsn, 
                            min_size=self.pool_min_size, 
                            max_size=self.pool_max_size,
                            command_timeout=self.connection_timeout
                        ),
                        timeout=self.connection_timeout
                    )
                    
                    # Test the connection with a simple query
                    async with self.pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    
                    self.logger.info("✅ Connected to the database!")
                    if self.debug_mode:
                        self.logger.debug(f"Successfully established connection pool after {attempt + 1} attempts")
                        
                    self.update_connection_time()
                    
                    # Register with connection monitor
                    if get_connection_monitor():
                        get_connection_monitor().register_connection(self.owner_name, self)
                    
                    # Start the cleanup task
                    await self._start_cleanup_task()
                    
                    # Log the operation
                    duration_ms = (time.time() - start_time) * 1000
                    if get_connection_monitor():
                        get_connection_monitor().log_operation(
                            self.owner_name, 
                            "Connect to database", 
                            duration_ms, 
                            True
                        )
                    return
                    
                except asyncio.TimeoutError:
                    last_error = "Connection timed out"
                    self.logger.error(f"❌ Database connection timed out after {self.connection_timeout} seconds")
                    if self.debug_mode:
                        self.logger.debug(f"Connection timeout on attempt {attempt + 1}/{retries}")
                except Exception as e:
                    last_error = str(e)
                    self.logger.error(f"❌ Database connection failed: {e}")
                    
                    if self.debug_mode:
                        # More detailed error logging in debug mode
                        self.logger.debug(f"Connection error details (attempt {attempt + 1}/{retries}):")
                        self.logger.debug(f"  Error type: {type(e).__name__}")
                        self.logger.debug(f"  Error message: {str(e)}")
                        self.logger.debug(f"  Connection params: {self.db_host}:{self.db_port}/{self.db_name} (user: {self.db_user})")
                        self.logger.debug(f"  Traceback: {traceback.format_exc()}")
                    
                    # Provide more specific error messages based on environment
                    if is_local:
                        if "connection refused" in str(e).lower():
                            self.logger.error("It appears PostgreSQL is not running locally. Please start your database server.")
                            if self.debug_mode:
                                self.logger.debug("Suggestion: Check if PostgreSQL service is running with 'services.msc' or 'systemctl status postgresql'")
                        elif "password authentication failed" in str(e).lower():
                            self.logger.error("Database credentials are incorrect. Check your .env file for DB_USER and DB_PASSWORD.")
                            if self.debug_mode:
                                self.logger.debug("Suggestion: Verify credentials in .env file and PostgreSQL user permissions")
                        elif "does not exist" in str(e).lower() and self.db_name in str(e).lower():
                            self.logger.error(f"Database '{self.db_name}' does not exist. Please create it or update DB_NAME in your .env file.")
                            if self.debug_mode:
                                self.logger.debug(f"Suggestion: Create database with 'CREATE DATABASE {self.db_name};'")
                    
                    if attempt < retries - 1:
                        self.logger.info(f"⏳ Retrying in {delay} seconds... ({attempt + 1}/{retries})")
                        await asyncio.sleep(delay)
    
            # Build a more helpful error message
            error_msg = f"❌ Failed to connect to the database after {retries} attempts. Last error: {last_error}"
            if is_local:
                error_msg += "\nFor local development, ensure PostgreSQL is running and check your database credentials."
                error_msg += f"\nAttempted connection to: {self.db_host}:{self.db_port}/{self.db_name}"
            
            # Add debug info if enabled
            if self.debug_mode:
                error_msg += "\n\n--- Debug Information ---"
                error_msg += f"\nEnvironment: {'Local' if is_local else 'Remote'}"
                error_msg += f"\nTest DB: {self.use_test_db}"
                error_msg += f"\nConnection params: {self.db_host}:{self.db_port}/{self.db_name} (user: {self.db_user})"
                error_msg += f"\nPool settings: min={self.pool_min_size}, max={self.pool_max_size}, timeout={self.connection_timeout}s"
                error_msg += "\n------------------------"
            
            # Log the error in the monitor
            duration_ms = (time.time() - start_time) * 1000
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name,
                    "Connect to database",
                    duration_ms,
                    False,
                    error_msg
                )
            
            raise ConnectionError(error_msg)

    async def close(self):
        """Closes the connection pool."""
        async with self.semaphore:
            if self.pool and not self.pool._closed:
                # Log with monitor before closing
                if get_connection_monitor():
                    get_connection_monitor().log_connection_closed(self.owner_name)
                
                await self.pool.close()
                self.logger.info("🔌 Database connection closed.")
                self.pool = None
                self.last_connection_time = None
                
                # Cancel the cleanup task if it's running
                if self._cleanup_task and not self._cleanup_task.done():
                    self._cleanup_task.cancel()
                    try:
                        await self._cleanup_task
                    except asyncio.CancelledError:
                        pass
                self._cleanup_task = None

    async def check_connection(self):
        """
        Check if the connection is still valid and reconnect if necessary.
        Also updates the connection time when the connection is valid.
        Uses a semaphore to ensure only one check happens at a time.
        """
        async with self.semaphore:
            try:
                # Check if pool exists and is not closed
                if self.pool is None or self.pool._closed:
                    if self.debug_mode:
                        self.logger.debug("Connection check: Pool is closed or None, reconnecting...")
                    self.logger.info("Pool is closed or None, reconnecting...")
                    await self.connect()
                    return True
                    
                # Check if connection time has expired
                if self.last_connection_time and (time.time() - self.last_connection_time) < self.connection_expiration:
                    if self.debug_mode:
                        self.logger.debug(f"Connection check: Recent connection ({time.time() - self.last_connection_time:.1f}s ago), skipping test query")
                    self.update_connection_time()
                    return True
                
                # Test the connection with a simple query
                if self.debug_mode:
                    self.logger.debug("Connection check: Testing with SELECT 1 query")
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                    self.update_connection_time()
                    if self.debug_mode:
                        self.logger.debug("Connection check: Connection is valid")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Error checking connection: {str(e)}")
                if self.debug_mode:
                    self.logger.debug(f"Connection check error details:")
                    self.logger.debug(f"  Error type: {type(e).__name__}")
                    self.logger.debug(f"  Error message: {str(e)}")
                    self.logger.debug(f"  Traceback: {traceback.format_exc()}")
                
                # Log the error with the monitor
                if get_connection_monitor():
                    get_connection_monitor().log_error(
                        self.owner_name, 
                        e, 
                        "Check connection"
                    )
                
                # Connection is invalid, close and reconnect
                await self.close()
                await self.connect()
                
            return True
                
    def update_connection_time(self):
        """Update the last connection time."""
        self.last_connection_time = time.time()
        if get_connection_monitor():
            get_connection_monitor().update_connection_usage(self.owner_name)
    
    async def execute_query(self, query: str, *args, timeout: float = None) -> None:
        """Execute a query without returning results."""
        start_time = time.time()
        
        try:
            if self.debug_mode:
                truncated_query = query[:100] + "..." if len(query) > 100 else query
                params_str = ", ".join([str(arg)[:20] for arg in args])
                self.logger.debug(f"Executing query: {truncated_query}")
                if args:
                    self.logger.debug(f"With parameters: {params_str}")
                
            await self.check_connection()
            async with self._operation_semaphore:
                await safe_pool_execute(
                    self.pool,
                    lambda conn: conn.execute(query, *args, timeout=timeout)
                )
            
            # Log the successful operation
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Execute query: {query[:50]}...", duration_ms, self.logger)
            
            if self.debug_mode:
                self.logger.debug(f"Query executed successfully in {duration_ms:.2f}ms")
                
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name,
                    f"Execute query: {query[:50]}...",
                    duration_ms,
                    True
                )
        except Exception as e:
            # Log the failed operation
            duration_ms = (time.time() - start_time) * 1000
            log_db_error(f"Execute query: {query[:50]}...", e, self.logger)
            
            if self.debug_mode:
                self.logger.debug(f"Query execution failed after {duration_ms:.2f}ms")
                self.logger.debug(f"Error type: {type(e).__name__}")
                self.logger.debug(f"Error message: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name,
                    f"Execute query: {query[:50]}...",
                    duration_ms,
                    False,
                    str(e)
                )
            raise
    
    async def fetch(self, query: str, *args, timeout: float = None) -> List[asyncpg.Record]:
        """Execute a query and return all results."""
        start_time = time.time()
        
        try:
            if self.debug_mode:
                truncated_query = query[:100] + "..." if len(query) > 100 else query
                params_str = ", ".join([str(arg)[:20] for arg in args])
                self.logger.debug(f"Fetching query: {truncated_query}")
                if args:
                    self.logger.debug(f"With parameters: {params_str}")
                    
            await self.check_connection()
            async with self._operation_semaphore:
                result = await safe_pool_execute(
                    self.pool,
                    lambda conn: conn.fetch(query, *args, timeout=timeout)
                )
            
            # Log success
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Fetch query", duration_ms, self.logger)
            
            if self.debug_mode:
                row_count = len(result) if result else 0
                self.logger.debug(f"Fetch query returned {row_count} rows in {duration_ms:.2f}ms")
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name, 
                    "Fetch query", 
                    duration_ms, 
                    True
                )
                
            return result
        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            log_db_error("Fetch query", e, self.logger)
            
            if self.debug_mode:
                self.logger.debug(f"Fetch query failed after {duration_ms:.2f}ms")
                self.logger.debug(f"Error type: {type(e).__name__}")
                self.logger.debug(f"Error message: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name,
                    "Fetch query",
                    duration_ms,
                    False,
                    str(e)
                )
            
            # Return empty list instead of None to avoid null reference errors
            return []
    
    async def fetchval(self, query: str, *args, timeout: float = None) -> Any:
        """Execute a query and return a single value."""
        start_time = time.time()
        
        try:
            await self.check_connection()
            async with self._operation_semaphore:
                result = await safe_pool_execute(
                    self.pool,
                    lambda conn: conn.fetchval(query, *args, timeout=timeout)
                )
            
            # Log success
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Fetchval query", duration_ms, self.logger)
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name, 
                    "Fetchval query", 
                    duration_ms, 
                    True
                )
                
            return result
        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            log_db_error("Fetchval query", e, self.logger)
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name,
                    "Fetchval query",
                    duration_ms,
                    False,
                    str(e)
                )
            
            return None
    
    async def fetchrow(self, query: str, *args, timeout: float = None) -> Optional[asyncpg.Record]:
        """Execute a query and return a single row."""
        start_time = time.time()
        
        try:
            await self.check_connection()
            async with self._operation_semaphore:
                result = await safe_pool_execute(
                    self.pool,
                    lambda conn: conn.fetchrow(query, *args, timeout=timeout)
                )
            
            # Log success
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Fetchrow query", duration_ms, self.logger)
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name, 
                    "Fetchrow query", 
                    duration_ms, 
                    True
                )
                
            return result
        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            log_db_error("Fetchrow query", e, self.logger)
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name,
                    "Fetchrow query",
                    duration_ms,
                    False,
                    str(e)
                )
            
            return None
    
    async def executemany(self, query: str, args, timeout: float = None) -> str:
        """Execute a query with multiple sets of parameters."""
        start_time = time.time()
        
        try:
            await self.check_connection()
            async with self._operation_semaphore:
                result = await safe_pool_execute(
                    self.pool,
                    lambda conn: conn.executemany(query, args, timeout=timeout)
                )
            
            # Log success
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Executemany query", duration_ms, self.logger)
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name, 
                    "Executemany query", 
                    duration_ms, 
                    True
                )
                
            return result
        except Exception as e:
            # Log error
            duration_ms = (time.time() - start_time) * 1000
            log_db_error("Executemany query", e, self.logger)
            
            if get_connection_monitor():
                get_connection_monitor().log_operation(
                    self.owner_name,
                    "Executemany query",
                    duration_ms,
                    False,
                    str(e)
                )
            
            return ""
    
    async def transaction(self):
        """Return a transaction object to be used in an async with statement."""
        await self.check_connection()
        if get_connection_monitor():
            get_connection_monitor().log_operation(
                self.owner_name,
                "Start transaction",
                0,
                True
            )
        return self.pool.acquire()
