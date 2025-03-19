# AlgoTrader Dashboard

Interactive dashboard for visualizing and analyzing raw and processed market data.

## Features

- **Raw Data Dashboard**: Visualize OHLCV data, daily statistics, and volume analysis
- **Processed Data Dashboard**: Analyze features, correlations, and distributions in processed data
- **Interactive Filtering**: Filter data by ticker, date range, and bar type
- **Data Quality Monitoring**: Track data quality metrics and identify issues

## Running the Dashboard

### Option 1: Using the run script

```bash
python -m backend.dashboard.run_dashboard
```

This will:
- Launch the Streamlit server
- Open the dashboard in your default browser at http://localhost:8501
- Set up all required imports correctly

### Option 2: Direct Streamlit command

```bash
# From project root directory
streamlit run backend/dashboard/Home.py
```

## Dashboard Pages

### Home Page

The home page provides an overview of your database, including:
- Number of raw and processed data tables
- Available tickers and bar types
- Quick links to the raw and processed data dashboards

### Raw Data Page

The Raw Data page allows you to:
- Select a ticker from the available raw data tables
- View OHLCV candlestick charts with volume information
- Analyze daily statistics and returns
- Explore volume distribution and patterns
- Search and filter the raw data table

### Processed Data Page

The Processed Data page enables you to:
- Select a ticker ID and bar type
- View time series of key features
- Analyze feature distributions and statistics
- Explore feature correlations and relationships
- Search and filter the processed data table

## Extending the Dashboard

### Adding a New Page

1. Create a new file in the `pages` directory with a name starting with a number (for ordering)
2. Import required modules including `utils`
3. Set up the page config and sidebar
4. Implement your visualization and analysis functions

### Adding New Visualizations

1. Add new visualization functions to the appropriate page module
2. Add checkbox options in the sidebar to toggle these visualizations
3. Use the Plotly library for consistent look and feel

## Requirements

- Python 3.7+
- Streamlit
- Plotly
- Pandas
- NumPy
- AsyncPG (for database connection)

## Future Enhancements

- Live data updates and real-time visualization
- Trading strategy backtesting visualization
- Performance metrics dashboard
- User authentication and multi-user support 