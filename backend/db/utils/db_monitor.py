import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

class DBConnectionMonitor:
    """Monitors database connections across the application."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the monitor."""
        if cls._instance is None:
            cls._instance = DBConnectionMonitor()
        return cls._instance
    
    def __init__(self):
        """Initialize the connection monitor."""
        self.logger = logging.getLogger(__name__)
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.operations: List[Dict[str, Any]] = []
        self.max_operations_history = 100
        self._monitoring_task = None
        self.start_time = datetime.now()
    
    def register_connection(self, owner_name: str, db_instance):
        """
        Register a new database connection.
        
        Args:
            owner_name: Identifier for the connection owner
            db_instance: The Database instance being used
        """
        self.connections[owner_name] = {
            'instance': db_instance,
            'created_at': datetime.now(),
            'last_used': datetime.now(),
            'operation_count': 0,
            'errors': 0,
            'status': 'active',
            'pool_id': id(db_instance.pool) if hasattr(db_instance, 'pool') and db_instance.pool else None
        }
        self.logger.info(f"Registered new connection for {owner_name}")
    
    def update_connection_usage(self, owner_name: str):
        """
        Update the last used timestamp for a connection.
        
        Args:
            owner_name: Identifier for the connection owner
        """
        if owner_name in self.connections:
            self.connections[owner_name]['last_used'] = datetime.now()
            self.connections[owner_name]['operation_count'] += 1
    
    def log_operation(self, owner_name: str, operation: str, 
                     duration_ms: float, success: bool = True, 
                     details: Optional[str] = None):
        """
        Log a database operation.
        
        Args:
            owner_name: Identifier for the connection owner
            operation: Description of the operation
            duration_ms: Time taken in milliseconds
            success: Whether the operation was successful
            details: Additional details about the operation
        """
        self.operations.append({
            'owner': owner_name,
            'operation': operation,
            'timestamp': datetime.now(),
            'duration_ms': duration_ms,
            'success': success,
            'details': details
        })
        
        # Trim history if needed
        if len(self.operations) > self.max_operations_history:
            self.operations = self.operations[-self.max_operations_history:]
        
        if not success and owner_name in self.connections:
            self.connections[owner_name]['errors'] += 1
    
    def log_connection_closed(self, owner_name: str):
        """
        Log when a connection is closed.
        
        Args:
            owner_name: Identifier for the connection owner
        """
        if owner_name in self.connections:
            self.connections[owner_name]['status'] = 'closed'
            self.logger.info(f"Connection closed for {owner_name}")
    
    def get_connection_stats(self):
        """
        Get statistics about all database connections.
        
        Returns:
            Dictionary with connection statistics and history
        """
        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        return {
            'total_connections': len(self.connections),
            'active_connections': sum(1 for c in self.connections.values() 
                                     if c['status'] == 'active'),
            'total_operations': sum(c['operation_count'] for c in self.connections.values()),
            'error_count': sum(c['errors'] for c in self.connections.values()),
            'connections': {name: {k: v for k, v in info.items() if k != 'instance'} 
                           for name, info in self.connections.items()},
            'recent_operations': self.operations[-10:],  # Last 10 operations
            'uptime': uptime_str,
            'monitor_start_time': self.start_time
        }
    
    def get_connection_by_id(self, connection_id: str):
        """
        Get connection information by ID.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Connection information or None if not found
        """
        if connection_id in self.connections:
            conn_info = self.connections[connection_id]
            # Don't return the actual instance in the result
            return {k: v for k, v in conn_info.items() if k != 'instance'}
        return None
    
    def log_error(self, owner_name: str, error: Exception, 
                 operation: Optional[str] = None):
        """
        Log a database error.
        
        Args:
            owner_name: Identifier for the connection owner
            error: The exception that occurred
            operation: Description of the operation that failed
        """
        error_details = {
            'error_type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc()
        }
        
        if owner_name in self.connections:
            self.connections[owner_name]['errors'] += 1
            self.connections[owner_name]['last_error'] = error_details
        
        self.log_operation(
            owner_name=owner_name,
            operation=operation or "Database operation",
            duration_ms=0,
            success=False,
            details=str(error)
        )
        
        self.logger.error(f"Database error in {owner_name}: {str(error)}")
    
    async def start_monitoring(self):
        """Start periodic monitoring of connections."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            self.logger.info("Database connection monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring task."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            self.logger.info("Database connection monitoring stopped")
    
    async def _monitor_loop(self):
        """Background task to check connection status."""
        try:
            while True:
                for name, conn_info in list(self.connections.items()):
                    if conn_info['status'] == 'active':
                        db = conn_info['instance']
                        try:
                            # Check if the connection is still alive
                            if (hasattr(db, 'pool') and db.pool and not db.pool._closed and 
                                hasattr(db.pool, 'free_size')):
                                # Update pool status info
                                conn_info['pool_status'] = {
                                    'free_size': db.pool.free_size,
                                    'size': db.pool.size,
                                    'min_size': db.pool.minsize,
                                    'max_size': db.pool.maxsize
                                }
                            else:
                                conn_info['status'] = 'disconnected'
                                self.logger.warning(f"Connection for {name} appears to be disconnected")
                        except Exception as e:
                            conn_info['status'] = 'error'
                            self.logger.error(f"Error checking connection for {name}: {str(e)}")
                
                # Log overall status periodically
                active = sum(1 for c in self.connections.values() if c['status'] == 'active')
                total = len(self.connections)
                self.logger.info(f"Connection status: {active}/{total} active connections")
                
                # Clean up connections not used for a long time
                current_time = datetime.now()
                for name, conn_info in list(self.connections.items()):
                    if (conn_info['status'] == 'closed' and 
                        (current_time - conn_info['last_used']).total_seconds() > 3600):
                        # Remove connections closed for more than an hour
                        del self.connections[name]
                        self.logger.info(f"Removed stale connection for {name}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
        except asyncio.CancelledError:
            self.logger.info("Connection monitoring task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in connection monitoring loop: {str(e)}")
            self.logger.debug(traceback.format_exc())

# Global function to get the monitor instance
def get_connection_monitor():
    """Get the global connection monitor instance."""
    return DBConnectionMonitor.get_instance() 