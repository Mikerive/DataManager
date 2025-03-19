import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to sys.path so we can import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import utils
# Import our new enhanced chart component
from components.enhanced_time_series import display_enhanced_chart

# Import services
from backend.services.RawDataService.RawDataService import RawDataService

# Set page config
st.set_page_config(
    page_title="Raw Data - AlgoTrader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Initialize services
_raw_data_service = None

def initialize_services():
    """Initialize service instances if they haven't been already."""
    global _raw_data_service
    
    if _raw_data_service is None:
        # Get service instances if they were initialized in utils
        _raw_data_service = utils._raw_data_service
        
        # If services are still not initialized, create them
        if _raw_data_service is None:
            _raw_data_service = RawDataService()
            
    return _raw_data_service is not None

# Sidebar for filtering data
def show_sidebar():
    st.sidebar.markdown("## Filter Options")
    
    # Initialize services and get database info
    try:
        # Initialize services
        if not initialize_services():
            st.sidebar.error("Failed to initialize services")
            return minimal_options()
        
        global _raw_data_service
        
        # Use service to get database info - connection managed internally by service
        tables_info, raw_data_tables, _ = utils.run_async(_raw_data_service.get_database_info)
        
        # Format table names for display
        table_options = {t.replace('raw_data_', '').upper(): t for t in raw_data_tables}
    except Exception as e:
        st.sidebar.error(f"Error connecting to database: {str(e)}")
        st.sidebar.error("Please check your database configuration and ensure PostgreSQL is running.")
        # Return minimal options to avoid downstream errors
        return minimal_options()
    
    # Track if any settings have changed
    settings_changed = False
    
    # Initialize session state for tracking changes
    if "previous_settings" not in st.session_state:
        st.session_state.previous_settings = {
            "selected_ticker_name": None,
            "start_date": None,
            "end_date": None
        }
    
    # Select ticker
    selected_ticker_name = st.sidebar.selectbox(
        "Select Ticker",
        options=list(table_options.keys()),
        index=0 if table_options else None
    )
    
    # Check if ticker changed
    if selected_ticker_name != st.session_state.previous_settings["selected_ticker_name"]:
        settings_changed = True
        st.session_state.previous_settings["selected_ticker_name"] = selected_ticker_name
    
    selected_table = table_options[selected_ticker_name] if selected_ticker_name else None
    
    # Date range selector
    st.sidebar.markdown("## Date Range")
    
    # Default date range - last 30 days
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=30)
    
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", default_end_date)
    
    # Check if dates changed
    if start_date != st.session_state.previous_settings["start_date"] or end_date != st.session_state.previous_settings["end_date"]:
        settings_changed = True
        st.session_state.previous_settings["start_date"] = start_date
        st.session_state.previous_settings["end_date"] = end_date
    
    # Button to update chart
    update_chart = st.sidebar.button("Update Chart", key="update_chart_button") or settings_changed
    
    # Analysis options
    st.sidebar.markdown("## Display Options")
    
    # Which sections to show
    show_ohlcv = st.sidebar.checkbox("Show OHLCV Chart", value=True)
    show_daily_stats = st.sidebar.checkbox("Show Daily Statistics", value=True)
    show_volume_analysis = st.sidebar.checkbox("Show Volume Analysis", value=True)
    show_data_table = st.sidebar.checkbox("Show Data Table", value=False)
    
    return {
        "selected_ticker_name": selected_ticker_name,
        "selected_table": selected_table,
        "start_date": start_date,
        "end_date": end_date,
        "update_chart": update_chart,
        "show_ohlcv": show_ohlcv,
        "show_daily_stats": show_daily_stats,
        "show_volume_analysis": show_volume_analysis,
        "show_data_table": show_data_table
    }

def minimal_options():
    """Return minimal options to avoid downstream errors."""
    return {
        "selected_ticker_name": None,
        "selected_table": None,
        "start_date": datetime.now() - timedelta(days=30),
        "end_date": datetime.now(),
        "update_chart": False,
        "show_ohlcv": True,
        "show_daily_stats": True,
        "show_volume_analysis": True,
        "show_data_table": False
    }

def get_data_for_table(selected_table, start_date, end_date):
    """Get OHLCV data for the selected table and date range using the service layer."""
    if not selected_table:
        return pd.DataFrame()
    
    try:
        # Initialize services if not done already
        if not initialize_services():
            st.error("Failed to initialize services")
            return pd.DataFrame()
            
        global _raw_data_service
        
        # Extract ticker from the table name
        if selected_table.startswith('raw_data_'):
            # Use the service to get price data - connection managed internally by service
            df = utils.run_async(
                _raw_data_service.get_price_data,
                selected_table,
                start_date=start_date,
                end_date=end_date + timedelta(days=1) if end_date else None
            )
            return df
        else:
            # Fallback to direct database access for other tables
            # This should be rare and ideally moved to a service as well
            df = utils.run_async(
                utils.get_price_data,
                selected_table,
                start_date=start_date,
                end_date=end_date + timedelta(days=1) if end_date else None
            )
            return df
            
    except Exception as e:
        st.error(f"Error retrieving data: {str(e)}")
        return pd.DataFrame()

def show_ohlcv_chart(df, selected_ticker_name, selected_table):
    """Show OHLCV chart for the selected ticker using the enhanced chart component."""
    st.markdown('<div class="sub-header">OHLCV Chart</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning(f"No data available for {selected_ticker_name}")
        return
    
    # Use the enhanced chart component instead of the original function
    stats = display_enhanced_chart(
        df,
        ticker_name=selected_ticker_name,
        table_name=selected_table,
        chart_type='ohlcv',
        key_prefix='ohlcv',
        show_advanced_options=True
    )
    
    if not stats:
        st.warning(f"Could not create chart for {selected_ticker_name}")
        return
    
    # Display statistics below the chart
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Data Points", f"{stats.get('total_rows', 0):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Zero Volume %", f"{stats.get('zero_volume_percent', 0):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Avg Volume", f"{stats.get('avg_volume', 0):.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Max Volume", f"{stats.get('max_volume', 0):,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

def show_daily_statistics(df, selected_ticker_name):
    """Show daily statistics for the selected ticker."""
    st.markdown('<div class="sub-header">Daily Statistics</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning(f"No data available for {selected_ticker_name}")
        return
    
    # Calculate daily statistics
    daily_stats = utils.calculate_daily_stats(df)
    
    if daily_stats.empty:
        st.warning("Not enough data to calculate daily statistics")
        return
    
    # Create a bar chart of daily returns using the enhanced time series
    daily_returns_df = daily_stats.reset_index()
    daily_returns_df.rename(columns={'day': 'timestamp'}, inplace=True)
    
    # Use the enhanced chart component for daily returns
    st.markdown("### Daily Returns (%)")
    display_enhanced_chart(
        daily_returns_df,
        ticker_name=selected_ticker_name,
        chart_type='time_series',
        y_column='daily_return',
        key_prefix='daily_returns',
        show_advanced_options=False
    )
    
    # Display daily stats table
    st.markdown("### Daily OHLCV Summary")
    
    # Format the daily stats for display
    formatted_stats = daily_stats.copy()
    formatted_stats['daily_return'] = formatted_stats['daily_return'].round(2).astype(str) + '%'
    formatted_stats['daily_range'] = formatted_stats['daily_range'].round(2).astype(str) + '%'
    
    # Rename columns for better display
    formatted_stats = formatted_stats.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'bar_count': 'Bar Count',
        'daily_return': 'Return (%)',
        'daily_range': 'Range (%)'
    })
    
    st.dataframe(formatted_stats, use_container_width=True)

def show_volume_analysis(df, selected_ticker_name):
    """Show volume analysis for the selected ticker."""
    st.markdown('<div class="sub-header">Volume Analysis</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning(f"No data available for {selected_ticker_name}")
        return
    
    # Create volume distribution chart
    non_zero_volume = df[df['volume'] > 0]['volume']
    
    if len(non_zero_volume) == 0:
        st.warning("No non-zero volume data available for analysis")
        return
    
    fig_volume_dist = go.Figure()
    
    # Add histogram for volume distribution
    fig_volume_dist.add_trace(
        go.Histogram(
            x=non_zero_volume,
            nbinsx=50,
            marker_color='rgba(0, 128, 0, 0.7)'
        )
    )
    
    zero_volume_count = (df['volume'] == 0).sum()
    
    fig_volume_dist.update_layout(
        title=f"{selected_ticker_name} Volume Distribution (excluding {zero_volume_count} zero values)",
        xaxis_title="Volume",
        yaxis_title="Frequency",
        height=400
    )
    
    # Log scale option
    use_log_scale = st.checkbox("Use Log Scale for Volume Distribution", value=True)
    if use_log_scale:
        fig_volume_dist.update_xaxes(type="log")
    
    st.plotly_chart(fig_volume_dist, use_container_width=True)
    
    # Show summary statistics for volume
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Volume by day of week
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
        volume_by_day = df.groupby('day_of_week')['volume'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
        ])
        
        # Convert to dataframe for enhanced chart
        volume_by_day_df = volume_by_day.reset_index()
        
        st.markdown("#### Average Volume by Day of Week")
        display_enhanced_chart(
            volume_by_day_df, 
            chart_type='categorical',  # Changed to categorical chart type
            y_column='volume',
            x_column='day_of_week',  # Specify the x-column instead of using 'timestamp'
            key_prefix='volume_by_day',
            show_advanced_options=False
        )
    
    with col2:
        # Volume by hour of day
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        volume_by_hour = df.groupby('hour')['volume'].mean()
        
        # Convert to dataframe for enhanced chart
        volume_by_hour_df = volume_by_hour.reset_index()
        
        st.markdown("#### Average Volume by Hour of Day")
        # Use categorical chart for hour data instead of direct Plotly
        display_enhanced_chart(
            volume_by_hour_df,
            chart_type='categorical',
            y_column='volume',
            x_column='hour',
            key_prefix='volume_by_hour',
            show_advanced_options=False
        )
        
    with col3:
        # Volume and price correlation
        df_corr = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df_corr['price_change'] = df_corr['close'] - df_corr['open']
        df_corr['price_change_pct'] = (df_corr['price_change'] / df_corr['open']) * 100
        
        fig_corr = go.Figure(data=[
            go.Scatter(
                x=df_corr['price_change_pct'],
                y=df_corr['volume'],
                mode='markers',
                marker=dict(
                    color=df_corr['price_change_pct'],
                    colorscale='RdBu',
                    size=8,
                    colorbar=dict(title="Price Change (%)"),
                    opacity=0.7
                )
            )
        ])
        fig_corr.update_layout(
            title="Volume vs Price Change",
            xaxis_title="Price Change (%)",
            yaxis_title="Volume",
            height=300
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def show_data_table(df, selected_ticker_name):
    """Show data table for the selected ticker."""
    st.markdown('<div class="sub-header">Data Table</div>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning(f"No data available for {selected_ticker_name}")
        return
    
    # Add search and filter options
    search_term = st.text_input("Search (Timestamp Format: YYYY-MM-DD HH:MM:SS)")
    
    filtered_df = df
    if search_term:
        # Apply search to timestamp column
        filtered_df = df[df['timestamp'].astype(str).str.contains(search_term)]
    
    # Display data with pagination
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100])
    total_pages = (len(filtered_df) + page_size - 1) // page_size
    
    if total_pages > 0:
        page_num = st.slider("Page", 1, max(1, total_pages), 1)
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        display_df = filtered_df.iloc[start_idx:end_idx].copy()
        
        # Format display columns
        display_df = display_df.rename(columns={
            'timestamp': 'Timestamp',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        st.dataframe(display_df, use_container_width=True)
        st.text(f"Showing {start_idx+1}-{end_idx} of {len(filtered_df)} rows")
    else:
        st.info("No data to display with the current filters.")

def main():
    # Page title
    st.markdown('<div class="main-header">Raw Market Data</div>', unsafe_allow_html=True)
    
    try:
        # Show sidebar and get filter options
        options = show_sidebar()
        
        if not options or not options["selected_table"]:
            if not options:
                st.error("Failed to initialize dashboard components. Check the database connection.")
            else:
                st.warning("Please select a ticker from the sidebar")
            return
        
        # Store selections in session state for persistence
        if "data_cache" not in st.session_state:
            st.session_state.data_cache = {}
        
        # Create a cache key based on selected table and date range
        cache_key = f"{options['selected_table']}_{options['start_date']}_{options['end_date']}"
        
        # Check if we need to update the data (new selection or update button pressed)
        update_data = (
            options["update_chart"] or 
            cache_key not in st.session_state.data_cache
        )
        
        # Get data for the selected ticker
        if update_data:
            df = get_data_for_table(
                options["selected_table"],
                options["start_date"],
                options["end_date"]
            )
            # Cache the data
            st.session_state.data_cache[cache_key] = df
        else:
            # Use cached data
            df = st.session_state.data_cache.get(cache_key, pd.DataFrame())
        
        # Show ticker information
        st.markdown(f'<div class="sub-header">{options["selected_ticker_name"]} - Overview</div>', unsafe_allow_html=True)
        st.markdown(f"""
        * **Date Range:** {options["start_date"].strftime("%Y-%m-%d")} to {options["end_date"].strftime("%Y-%m-%d")}
        * **Data Points:** {len(df):,}
        * **Trading Days:** {df['timestamp'].dt.date.nunique() if not df.empty else 0}
        """)
        
        # Show the selected components
        if options["show_ohlcv"]:
            show_ohlcv_chart(df, options["selected_ticker_name"], options["selected_table"])
        
        if options["show_daily_stats"]:
            show_daily_statistics(df, options["selected_ticker_name"])
        
        if options["show_volume_analysis"]:
            show_volume_analysis(df, options["selected_ticker_name"])
        
        if options["show_data_table"]:
            show_data_table(df, options["selected_ticker_name"])
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # Cleanup is not necessary here - service manages its own connections
        pass
    
    # Footer
    st.markdown("---")
    st.markdown("Raw Data Dashboard | Data updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Cleanup resources if the app is shutting down
        if _raw_data_service is not None:
            try:
                utils.run_async(_raw_data_service.cleanup)
            except Exception as e:
                st.error(f"Failed to cleanup resources: {str(e)}")
                pass 