# AlgoTrader Dashboard Components

This directory contains reusable UI components for the AlgoTrader dashboard. These components are designed to be modular and provide consistent behavior across different pages of the dashboard.

## Components

### Enhanced Time Series Chart (`enhanced_time_series.py`)

A sophisticated time series chart component that provides advanced features for visualizing financial data with improved performance and user experience.

#### Key Features:

1. **Range Selector** - Quick time range selection buttons (1h, 6h, 1d, 1w, 1m, 3m, 6m, YTD, 1y, All)
2. **Range Slider** - Interactive slider for selecting date/time ranges
3. **Horizontal Scrollbar** - Allows horizontal scrolling through large datasets
4. **Performance Optimizations**:
   - Data downsampling for large datasets
   - Time-based aggregation (hour, day, week, month)
   - Efficient rendering with UI state preservation

#### Available Chart Types:

1. **Time Series Chart** - Simple bar or line charts for any time series data
2. **OHLCV Chart** - Candlestick chart with volume subplot

#### Usage Examples:

1. Basic time series chart:

```python
from components.enhanced_time_series import display_enhanced_chart

# Simple usage with default options
display_enhanced_chart(
    df,  # DataFrame with 'timestamp' column and data to plot
    ticker_name="AAPL",  # Optional ticker name to show in title
    chart_type='time_series',
    y_column='volume'  # Column to plot on y-axis
)

# Advanced usage with custom options
display_enhanced_chart(
    df,
    ticker_name="AAPL",
    chart_type='time_series',
    y_column='close',
    key_prefix='my_chart',  # Unique prefix for Streamlit widget keys
    show_advanced_options=True  # Show UI controls for customization
)
```

2. OHLCV chart:

```python
from components.enhanced_time_series import display_enhanced_chart

# Display OHLCV chart with candlesticks and volume
display_enhanced_chart(
    df,  # DataFrame with OHLCV data columns
    ticker_name="AAPL",
    table_name="raw_data_AAPL",  # Optional, used if ticker_name not provided
    chart_type='ohlcv',
    key_prefix='ohlcv_chart',
    show_advanced_options=True
)
```

#### Advanced Options:

When `show_advanced_options=True`, users will see a control panel with:

- Range selector toggle
- Range slider toggle
- Horizontal scrollbar toggle
- Max data points control (for performance)
- Time aggregation selector (None, Hour, Day, Week, Month)

#### Functions:

The module provides three main functions:

1. `create_enhanced_time_series()` - Creates a simple time series bar chart
2. `create_enhanced_ohlcv_chart()` - Creates a candlestick chart with volume subplot
3. `display_enhanced_chart()` - High-level function that displays either chart type with interactive controls

#### Performance Considerations:

- For datasets larger than 10,000 points, downsampling is automatically applied
- Time aggregation significantly improves performance and clarity for large datasets
- The horizontal scrollbar works best with the range slider enabled 