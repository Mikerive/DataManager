# Database Connection Monitoring and Error Handling

This directory contains the core database components for the AlgoTrader application, including a newly implemented database connection monitoring and standardized error handling system.

## Key Components

### 1. Database Class

- `Database.py` - The base class for database connections, now enhanced with connection monitoring and improved error handling.
- Each instance has a unique `owner_name` to track the source of database operations.
- Supports automatic reconnection and detailed operation logging.

### 2. Connection Monitoring

- `db_monitor.py` - Contains the `DBConnectionMonitor` class that tracks all database connections throughout the application.
- Features:
  - Real-time connection status tracking
  - Operation timing and statistics
  - Error logging and diagnostics
  - Connection pool utilization monitoring

### 3. Standardized Error Handling

- `db_utils.py` - Provides standardized utilities for database operations:
  - `handle_db_operation` - A wrapper for database operations with consistent error handling
  - `safe_pool_execute` - Safe execution with retry functionality
  - Error formatting and detailed logging

### 4. System Status Dashboard

- A dedicated Streamlit dashboard page for real-time monitoring
- Displays active connections, operation history, and performance metrics
- Includes controls for starting/stopping monitoring

## Usage

### Monitoring Database Operations

The monitoring system automatically tracks:
- Connection creation and closure
- Query execution time and success/failure
- Connection pool utilization
- Error patterns and frequencies

### Implementing Error Handling

Each database-related class should follow this pattern:

```python
from backend.db.db_utils import handle_db_operation

# Public method with error handling
async def get_data(self, id):
    return await handle_db_operation(
        "ClassNameOrOwner",
        "Operation description",
        lambda: self._get_data(id),
        default_value=None  # Return value in case of error
    )

# Private implementation method
async def _get_data(self, id):
    # Actual implementation
    db = await self._get_db()
    result = await db.fetchrow("SELECT * FROM table WHERE id = $1", id)
    return result
```

### Best Practices

1. **Consistent Naming**: Use the `_method_name` convention for internal implementation methods
2. **Owner Identification**: Always set a descriptive `owner_name` when creating Database instances
3. **Resource Cleanup**: Implement `close_connection()` methods in all database classes
4. **Default Values**: Provide appropriate default values for error cases
5. **Explicit Error Handling**: Use try/except in sensitive operations with specific error handling

## Monitoring Dashboard

A dedicated "System Status" dashboard provides real-time visibility into database operations:

- **Overview**: Connection counts, operation totals, error rates
- **Connections**: Details on each active connection including status and utilization
- **Operations**: Recent database operations with timing information
- **Controls**: Buttons to start/stop monitoring with auto-refresh option

## Developer Tools

- `update_db_classes.py` - A utility script to assist in updating existing database classes to use the new error handling pattern
- Can scan codebase for database classes and suggest appropriate modifications

## Connection Management Philosophy

The core philosophy of this system is:

1. **Independence**: Each service manages its own database connection
2. **Transparency**: All database operations are visible and trackable
3. **Resilience**: Automatic reconnection and retry logic for transient failures
4. **Consistency**: Standardized error handling across all components

By following these principles, database connection issues are easier to diagnose and the system becomes more resilient to transient database failures, especially in the Streamlit environment where request handling can be unpredictable. 