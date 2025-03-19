import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_enhanced_time_series(
    df, 
    title="Time Series Data",
    y_column='volume',
    ticker_name=None,
    show_range_selector=True,
    show_range_slider=True,
    show_horizontal_scrollbar=True,
    height=600,
    color_bars_by_change=True,
    max_points=10000,
    aggregation=None
):
    """
    Create an enhanced time series plot with range selector, slider, and horizontal scrolling.
    
    Args:
        df (pd.DataFrame): DataFrame with timestamp column and data to plot
        title (str): Chart title
        y_column (str): Column to plot on the y-axis (e.g., 'volume', 'close', etc.)
        ticker_name (str, optional): Ticker name to include in the title
        show_range_selector (bool): Whether to show the range selector buttons
        show_range_slider (bool): Whether to show the range slider
        show_horizontal_scrollbar (bool): Whether to enable horizontal scrollbar
        height (int): Height of the chart in pixels
        color_bars_by_change (bool): Color bars based on up/down changes
        max_points (int): Maximum number of data points to display (for performance)
        aggregation (str, optional): Aggregation method ('hour', 'day', 'week', 'month')
    
    Returns:
        plotly.graph_objects.Figure: The enhanced time series chart
    """
    if df.empty:
        st.warning("No data available for plotting.")
        return None
    
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_dtype(plot_df['timestamp']):
        plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
    
    # Apply aggregation if specified
    if aggregation:
        # Set timestamp as index for resampling
        plot_df = plot_df.set_index('timestamp')
        
        # Define resample frequency based on aggregation parameter
        freq_map = {
            'hour': 'H',
            'day': 'D',
            'week': 'W',
            'month': 'M'
        }
        freq = freq_map.get(aggregation.lower(), 'D')  # Default to daily if not recognized
        
        # Perform resampling with appropriate aggregation methods
        agg_methods = {}
        numeric_columns = plot_df.select_dtypes(include=[np.number]).columns
        
        # Standard OHLCV aggregation
        if 'open' in numeric_columns:
            agg_methods['open'] = 'first'
        if 'high' in numeric_columns:
            agg_methods['high'] = 'max'
        if 'low' in numeric_columns:
            agg_methods['low'] = 'min'
        if 'close' in numeric_columns:
            agg_methods['close'] = 'last'
        if 'volume' in numeric_columns:
            agg_methods['volume'] = 'sum'
        
        # Add any other numeric columns with 'mean' as default aggregation
        for col in numeric_columns:
            if col not in agg_methods:
                agg_methods[col] = 'mean'
        
        # Perform resampling
        try:
            plot_df = plot_df.resample(freq).agg(agg_methods).reset_index()
            logger.info(f"Aggregated data to {aggregation} intervals, resulting in {len(plot_df)} points")
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
            st.warning(f"Could not apply {aggregation} aggregation: {str(e)}")
    
    # Apply downsampling if needed
    if max_points > 0 and len(plot_df) > max_points:
        # Sort by timestamp to ensure proper sampling
        plot_df = plot_df.sort_values('timestamp')
        
        # Calculate sampling factor
        sampling_factor = len(plot_df) // max_points
        
        # Apply sampling
        plot_df = plot_df.iloc[::sampling_factor].copy()
        logger.info(f"Downsampled from {len(df)} to {len(plot_df)} points")
    
    # Format title to include ticker
    display_title = title
    if ticker_name:
        display_title = f"{ticker_name} - {title}"
    
    # Create figure
    fig = go.Figure()
    
    # Determine bar colors if needed
    if color_bars_by_change and y_column in plot_df.columns:
        # For volume or price data, color based on change
        colors = []
        for i in range(len(plot_df)):
            if i > 0:
                # Use close price to determine color if available
                if 'close' in plot_df.columns and 'open' in plot_df.columns:
                    if plot_df['close'].iloc[i] > plot_df['open'].iloc[i]:
                        colors.append('rgba(0, 128, 0, 0.7)')  # Green for up
                    else:
                        colors.append('rgba(255, 0, 0, 0.7)')  # Red for down
                # Otherwise, use the y-column itself for comparison with previous value
                elif plot_df[y_column].iloc[i] > plot_df[y_column].iloc[i-1]:
                    colors.append('rgba(0, 128, 0, 0.7)')  # Green for up
                else:
                    colors.append('rgba(255, 0, 0, 0.7)')  # Red for down
            else:
                colors.append('rgba(0, 128, 0, 0.7)')  # Default green for first bar
    else:
        # Default color
        colors = 'rgba(0, 128, 0, 0.7)'
    
    # Add bar trace
    fig.add_trace(
        go.Bar(
            x=plot_df['timestamp'],
            y=plot_df[y_column],
            name=y_column.capitalize(),
            marker_color=colors,
            hovertemplate='%{x}<br>%{y}<extra></extra>'
        )
    )
    
    # Configure range selector buttons if enabled
    range_selector = dict(
        buttons=list([
            dict(count=1, label="1h", step="hour", stepmode="backward"),
            dict(count=6, label="6h", step="hour", stepmode="backward"),
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ]),
        x=0.01,
        y=1.05,
        bgcolor='rgba(150, 200, 250, 0.4)',
        font=dict(size=12)
    ) if show_range_selector else None
    
    # Configure range slider if enabled
    range_slider = dict(visible=True) if show_range_slider else dict(visible=False)
    
    # Update layout
    fig.update_layout(
        title=display_title,
        xaxis=dict(
            rangeselector=range_selector,
            rangeslider=range_slider,
            type="date",
            # Enable panning when horizontal scrolling is enabled
            fixedrange=not show_horizontal_scrollbar
        ),
        yaxis_title=y_column.capitalize(),
        height=height,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        # Performance optimizations
        uirevision='constant',  # Preserves UI state on updates
    )
    
    return fig

def create_enhanced_ohlcv_chart(
    df, 
    ticker_name=None,
    table_name=None,
    show_range_selector=True,
    show_range_slider=True,
    show_horizontal_scrollbar=True,
    height=600,
    max_points=10000,
    aggregation=None
):
    """
    Create an enhanced OHLCV chart with candlesticks and volume bars.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        ticker_name (str, optional): Ticker name to display
        table_name (str, optional): Table name (used if ticker_name not provided)
        show_range_selector (bool): Whether to show range selector buttons
        show_range_slider (bool): Whether to show the range slider
        show_horizontal_scrollbar (bool): Whether to enable horizontal scrollbar
        height (int): Height of the chart in pixels
        max_points (int): Maximum number of data points to display
        aggregation (str, optional): Aggregation method ('hour', 'day', 'week', 'month')
    
    Returns:
        tuple: (plotly figure, statistics dictionary)
    """
    if df.empty:
        st.warning("No data available for plotting OHLCV chart.")
        return None, {}
    
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_dtype(plot_df['timestamp']):
        plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
    
    # Apply aggregation if specified (similar to the time series function)
    if aggregation:
        plot_df = plot_df.set_index('timestamp')
        freq_map = {'hour': 'H', 'day': 'D', 'week': 'W', 'month': 'M'}
        freq = freq_map.get(aggregation.lower(), 'D')
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        try:
            plot_df = plot_df.resample(freq).agg(agg_dict).reset_index()
            logger.info(f"Aggregated OHLCV data to {aggregation} intervals, resulting in {len(plot_df)} points")
        except Exception as e:
            logger.error(f"Error during OHLCV aggregation: {str(e)}")
            st.warning(f"Could not apply {aggregation} aggregation: {str(e)}")
    
    # Apply downsampling if needed
    if max_points > 0 and len(plot_df) > max_points:
        plot_df = plot_df.sort_values('timestamp')
        sampling_factor = len(plot_df) // max_points
        plot_df = plot_df.iloc[::sampling_factor].copy()
        logger.info(f"Downsampled OHLCV data from {len(df)} to {len(plot_df)} points")
    
    # Extract ticker from table_name if not provided
    if not ticker_name and table_name:
        import re
        match = re.search(r'raw_data_([A-Z0-9_]+)', table_name)
        ticker_name = match.group(1) if match else table_name
    
    # Calculate volume statistics for display
    zero_volume_count = (plot_df['volume'] == 0).sum()
    total_rows = len(plot_df)
    zero_volume_percent = (zero_volume_count / total_rows) * 100 if total_rows > 0 else 0
    avg_volume = plot_df['volume'].mean()
    max_volume = plot_df['volume'].max()
    
    # Create a copy for volume display (replacing zeros with small values for visibility)
    plot_df_with_volume = plot_df.copy()
    plot_df_with_volume.loc[plot_df_with_volume['volume'] == 0, 'volume'] = 1
    
    # Calculate volume color based on price change
    colors = []
    for i in range(len(plot_df_with_volume)):
        if i > 0:
            if plot_df_with_volume['close'].iloc[i] > plot_df_with_volume['close'].iloc[i-1]:
                colors.append('rgba(0, 128, 0, 0.7)')  # Green for up
            else:
                colors.append('rgba(255, 0, 0, 0.7)')  # Red for down
        else:
            colors.append('rgba(0, 128, 0, 0.7)')  # Default green for first bar
    
    # Create subplot with 2 rows with improved height ratio
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,  # Increased spacing between subplots
        subplot_titles=(f'{ticker_name} Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Add candlestick chart for OHLC
    fig.add_trace(
        go.Candlestick(
            x=plot_df['timestamp'],
            open=plot_df['open'], 
            high=plot_df['high'],
            low=plot_df['low'], 
            close=plot_df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add volume bar chart with colored bars
    fig.add_trace(
        go.Bar(
            x=plot_df_with_volume['timestamp'],
            y=plot_df_with_volume['volume'],
            name='Volume',
            marker=dict(color=colors)
        ),
        row=2, col=1
    )
    
    # Configure range selector buttons if enabled
    range_selector = dict(
        buttons=list([
            dict(count=1, label="1h", step="hour", stepmode="backward"),
            dict(count=6, label="6h", step="hour", stepmode="backward"),
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ]),
        x=0.01,
        y=1.15,  # Increased from 1.05 to create more space at the top
        bgcolor='rgba(150, 200, 250, 0.4)',
        font=dict(size=12)
    ) if show_range_selector else None
    
    # Configure range slider if enabled
    range_slider = dict(
        visible=True,
        thickness=0.1  # Reduced thickness to prevent overlap
    ) if show_range_slider else dict(visible=False)
    
    # Update layout with improved margins
    fig.update_layout(
        title={
            'text': f'{ticker_name} OHLCV Chart',
            'y': 0.95,  # Changed from 2 to 0.95 - must be between 0 and 1
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(
            rangeselector=range_selector,
            rangeslider=range_slider,
            type="date",
            fixedrange=not show_horizontal_scrollbar
        ),
        height=height,
        margin=dict(t=100, b=80, l=50, r=50),  # Increased top margin to 100px
        showlegend=False,
        uirevision='constant',  # Preserves UI state on updates
    )
    
    # Add more space below volume subplot for the rangeslider
    fig.update_yaxes(
        type="log", 
        row=2, 
        col=1,
        domain=[0.15, 0.3]  # Adjust domain to prevent overlap with rangeslider
    )
    
    return fig, {
        'zero_volume_percent': zero_volume_percent,
        'avg_volume': avg_volume,
        'max_volume': max_volume,
        'total_rows': total_rows
    }

def create_categorical_chart(
    df, 
    title="Categorical Chart",
    y_column='volume',
    x_column='timestamp',
    ticker_name=None,
    height=400,
    color_bars_by_change=False
):
    """
    Create a simple bar chart for categorical data (like days of week).
    
    Args:
        df (pd.DataFrame): DataFrame with data to plot
        title (str): Chart title
        y_column (str): Column to plot on the y-axis
        x_column (str): Column to use for x-axis categories
        ticker_name (str, optional): Ticker name to include in the title
        height (int): Height of the chart in pixels
        color_bars_by_change (bool): Color bars based on values
    
    Returns:
        plotly.graph_objects.Figure: The bar chart
    """
    if df.empty:
        st.warning("No data available for plotting.")
        return None
    
    # Make a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Format title to include ticker
    display_title = title
    if ticker_name:
        display_title = f"{ticker_name} - {title}"
    
    # Create figure
    fig = go.Figure()
    
    # Determine bar colors if needed
    if color_bars_by_change and y_column in plot_df.columns:
        # Sort values for consistent coloring
        plot_df = plot_df.sort_values(y_column)
        
        # Generate color gradient based on values
        values = plot_df[y_column]
        min_val = values.min()
        max_val = values.max()
        norm_values = [(val - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for val in values]
        
        # Generate colors from red to green
        colors = [
            f'rgba({int(255 * (1 - nv))}, {int(255 * nv)}, 0, 0.7)'
            for nv in norm_values
        ]
        
        # Restore original order
        plot_df = plot_df.sort_index()
    else:
        # Default color
        colors = 'rgba(26, 118, 255, 0.7)'
    
    # Add bar trace
    fig.add_trace(
        go.Bar(
            x=plot_df[x_column],
            y=plot_df[y_column],
            name=y_column.capitalize(),
            marker_color=colors,
            hovertemplate='%{x}<br>%{y}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=display_title,
        xaxis_title=x_column.capitalize().replace('_', ' '),
        yaxis_title=y_column.capitalize().replace('_', ' '),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        # Performance optimizations
        uirevision='constant'  # Preserves UI state on updates
    )
    
    return fig

def display_enhanced_chart(
    df, 
    ticker_name=None, 
    table_name=None,
    chart_type='ohlcv',
    y_column='volume',
    x_column='timestamp',  # Added x_column parameter
    key_prefix='',
    show_advanced_options=True
):
    """
    Display an enhanced chart with options for customization.
    
    Args:
        df (pd.DataFrame): DataFrame with data to plot
        ticker_name (str, optional): Ticker name to display
        table_name (str, optional): Table name for the data
        chart_type (str): Type of chart ('ohlcv', 'time_series', or 'categorical')
        y_column (str): Column to plot on y-axis for time_series charts
        x_column (str): Column to use for x-axis (default 'timestamp')
        key_prefix (str): Prefix for Streamlit widget keys to avoid duplicates
        show_advanced_options (bool): Whether to show advanced chart options
    
    Returns:
        dict: Chart statistics
    """
    if df.empty:
        st.warning(f"No data available for {ticker_name or table_name}")
        return {}
    
    # Check if we're dealing with categorical data
    is_categorical = False
    if x_column in df.columns:
        # Check if the column contains string values like day names
        if df[x_column].dtype == 'object' or pd.api.types.is_string_dtype(df[x_column]):
            is_categorical = True
            chart_type = 'categorical'  # Force categorical chart type
    
    # Show advanced options if enabled (only for time series and OHLCV)
    chart_options = {}
    settings_updated = False
    
    if show_advanced_options and not is_categorical:
        with st.expander("Chart Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Range controls
                chart_options["show_range_selector"] = st.checkbox(
                    "Show Range Selector", 
                    value=True,
                    key=f"{key_prefix}_show_range_selector"
                )
                chart_options["show_range_slider"] = st.checkbox(
                    "Show Range Slider", 
                    value=True,
                    key=f"{key_prefix}_show_range_slider"
                )
                chart_options["show_horizontal_scrollbar"] = st.checkbox(
                    "Show Horizontal Scrollbar", 
                    value=True,
                    key=f"{key_prefix}_show_scrollbar"
                )
            
            with col2:
                # Performance options
                chart_options["max_points"] = st.number_input(
                    "Max Data Points (0 = all)",
                    min_value=0,
                    max_value=50000,
                    value=10000,
                    step=1000,
                    key=f"{key_prefix}_max_points"
                )
                
                chart_options["aggregation"] = st.selectbox(
                    "Time Aggregation",
                    options=["None", "Hour", "Day", "Week", "Month"],
                    key=f"{key_prefix}_aggregation"
                )
            
            # Add apply button for chart settings
            settings_updated = st.button("Apply Chart Settings", key=f"{key_prefix}_apply_settings")
            
            # Track settings in session state to detect changes
            if f"{key_prefix}_settings" not in st.session_state:
                st.session_state[f"{key_prefix}_settings"] = chart_options.copy()
            elif settings_updated:
                # Update stored settings when button is clicked
                st.session_state[f"{key_prefix}_settings"] = chart_options.copy()
                st.rerun()  # Rerun the app to apply changes
    else:
        # Default options if not showing advanced controls
        chart_options = {
            "show_range_selector": True,
            "show_range_slider": True,
            "show_horizontal_scrollbar": True,
            "max_points": 10000,
            "aggregation": "None"
        }
    
    # Use stored settings if they exist and we're not using new ones
    if f"{key_prefix}_settings" in st.session_state and not settings_updated:
        chart_options = st.session_state[f"{key_prefix}_settings"]
    
    # Process aggregation option
    aggregation = None
    if chart_options["aggregation"] != "None":
        aggregation = chart_options["aggregation"].lower()
    
    # Create and display chart based on chart type
    if chart_type == 'categorical':
        # For categorical data like day of week
        fig = create_categorical_chart(
            df,
            y_column=y_column,
            x_column=x_column,
            ticker_name=ticker_name,
            height=400
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False
            })
            return {"total_rows": len(df)}
    elif chart_type == 'ohlcv':
        fig, stats = create_enhanced_ohlcv_chart(
            df,
            ticker_name=ticker_name,
            table_name=table_name,
            show_range_selector=chart_options["show_range_selector"],
            show_range_slider=chart_options["show_range_slider"],
            show_horizontal_scrollbar=chart_options["show_horizontal_scrollbar"],
            max_points=chart_options["max_points"],
            aggregation=aggregation,
            height=650  # Increased height to accommodate all elements
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,  # Enable scroll zoom for better navigation
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['pan2d'] if chart_options["show_horizontal_scrollbar"] else [],
                'displaylogo': False
            })
            
            # Removed duplicate statistics display - these are now shown in streamlit_dashboard.py
            
            return stats
    else:
        fig = create_enhanced_time_series(
            df,
            y_column=y_column,
            ticker_name=ticker_name,
            show_range_selector=chart_options["show_range_selector"],
            show_range_slider=chart_options["show_range_slider"],
            show_horizontal_scrollbar=chart_options["show_horizontal_scrollbar"],
            max_points=chart_options["max_points"],
            aggregation=aggregation
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,  # Enable scroll zoom for better navigation
                'displayModeBar': True,
                'modeBarButtonsToAdd': ['pan2d'] if chart_options["show_horizontal_scrollbar"] else [],
                'displaylogo': False
            })
            return {"total_rows": len(df)}
    
    return {} 