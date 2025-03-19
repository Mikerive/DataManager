# AlgoTrader System Documentation

This document provides an overview of the system architecture, folder structure, and key components of the AlgoTrader application.

## Folder Structure

```
AlgoTrader/
├── backend/
│   ├── endpoints/              # FastAPI route definitions
│   │   ├── tickers_endpoint.py # Endpoints for ticker operations
│   │   └── processed_data_endpoint.py # Endpoints for processed data
│   ├── functions/              # Core functionality modules
│   │   ├── data/               # Data source implementations
│   │   │   ├── AlphaVantage.py # AlphaVantage API interface
│   │   │   ├── datacache.py    # Base cache implementation
│   │   │   └── Processing/     # Data processing modules
│   │   │       └── RawDataHandler.py # Raw data handling and caching
│   │   ├── db/                 # Database interaction modules
│   │   │   ├── database.py     # Base database class
│   │   │   ├── tickersDb.py    # Ticker database operations
│   │   │   └── RawDataDb.py    # Raw data database operations
│   │   └── preprocessing/      # Data preprocessing modules
│   │       └── entropy_bars.py # Sequential entropy bars implementation
│   └── services/               # Business logic services
│       └── tickerService.py    # Service for ticker operations
```

## Key Components

### Data Loading

- **AlphaVantage**: Implementation for fetching financial data from AlphaVantage API
  - Uses pandas DataFrames for data structure
  - Provides methods for ticker listings and time series data
  - Includes rate limiting (75 calls per minute) for API compliance
  - Supports intraday 1-minute data retrieval

### Database Management

- **Database**: Base class for database connections and common operations
- **TickersDb**: Specialized database operations for ticker data
  - Methods for CRUD operations on tickers table
  - Batch operations for efficient data storage
  - Stores IPO dates for historical data retrieval

### Data Processing

- **RawDataHandler**: Handles fetching and caching of raw financial data
  - Three-tier data retrieval strategy (memory, database, API)
  - Supports bulk data operations
  - Automatic historical data import from IPO date

### Services

- **TickerService**: Business logic for ticker operations
  - Coordinates between data loaders and database
  - Handles downloading and storing ticker data
  - Includes methods for time series data retrieval and processing

### Endpoints

- **TickersEndpoint**: REST API endpoints for ticker operations
  - Initialize ticker database
  - CRUD operations for tickers
  - Time series data retrieval

### Preprocessing

- **Sequential Entropy Bars**: Information-theoretic bars based on entropy
  - Both Shannon and Tsallis entropy calculation methods
  - NumPy-optimized implementation for large datasets
  - Adaptive threshold based on running average of daily entropy
  - Key parameters:
    - `window`: Controls rolling window size for entropy calculation (default: 100)
    - `method`: Entropy calculation method ('shannon' or 'tsallis')
    - `price_column`: Column to use for price data (default: 'close')
    - `daily_ratio`: Ratio of daily entropy to use as threshold
    - `avg_window`: Number of days for running average (default: 200)

## Design Principles

1. **Service-Oriented Architecture**: Separate concerns into distinct layers
   - Endpoints handle HTTP requests/responses
   - Services contain business logic
   - Functions provide core functionality

2. **Object-Oriented Design**: Organize code by objects and classes
   - Use inheritance for common functionality
   - Encapsulate related operations within appropriate classes

3. **Asynchronous Operations**: Use async/await for non-blocking I/O
   - Efficient handling of network requests
   - Background tasks for long-running operations

4. **NumPy and Pandas Optimization**: Leverage libraries for efficient data processing
   - Pandas DataFrames for data manipulation and analysis
   - NumPy for high-performance numerical operations
   - Vectorized operations for performance

5. **Batch Processing**: Handle large datasets efficiently
   - Process data in manageable batches
   - Optimized database operations

6. **Absolute Imports**: Use absolute imports to ensure clear module dependencies
   - Reduces import ambiguity
   - Improves code maintainability
