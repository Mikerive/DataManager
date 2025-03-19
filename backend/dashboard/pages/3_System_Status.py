import streamlit as st
import pandas as pd
import asyncio
import time
from datetime import datetime
import altair as alt
import json

# Import our custom modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from db.utils.db_monitor import get_connection_monitor
from dashboard.utils import run_async, load_css

# Page config
st.set_page_config(
    page_title="System Status",
    page_icon="🔍",
    layout="wide"
)

# Apply custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Initialize connection monitor if not already done
_connection_monitor = get_connection_monitor()

# Page header
st.markdown("<h1 class='main-header'>System Status Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Database Connection Monitoring and System Health</p>", unsafe_allow_html=True)

async def get_system_status():
    """Get the current system status."""
    monitor = get_connection_monitor()
    
    # Make sure monitoring is running
    await monitor.start_monitoring()
    
    # Get current stats
    return monitor.get_connection_stats()

def format_timestamp(timestamp):
    """Format a timestamp for display."""
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp

# Run async function to get status
status_data = run_async(get_system_status())

# Create tabs for different status views
status_tab, connections_tab, operations_tab = st.tabs(["System Overview", "Database Connections", "Recent Operations"])

with status_tab:
    # Display system overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Connections", status_data.get('active_connections', 0))
    
    with col2:
        st.metric("Total Operations", status_data.get('total_operations', 0))
    
    with col3:
        st.metric("Error Count", status_data.get('error_count', 0))
    
    with col4:
        st.metric("Uptime", status_data.get('uptime', 'N/A'))
    
    st.markdown("### System Information")
    
    # Display monitor start time
    start_time = status_data.get('monitor_start_time')
    if start_time:
        st.info(f"Monitoring active since: {format_timestamp(start_time)}")

    # Refresh button
    if st.button("Refresh Status Data"):
        st.experimental_rerun()

with connections_tab:
    st.markdown("### Database Connections")
    
    # Create a DataFrame for connections
    connections = status_data.get('connections', {})
    if connections:
        connection_data = []
        for name, info in connections.items():
            # Format data for display
            status_class = "status-green" if info.get('status') == 'active' else "status-red"
            
            # Create record
            connection_data.append({
                "Owner": name,
                "Status": info.get('status', 'unknown'),
                "Created": format_timestamp(info.get('created_at')),
                "Last Used": format_timestamp(info.get('last_used')),
                "Operations": info.get('operation_count', 0),
                "Errors": info.get('errors', 0),
                "Pool ID": info.get('pool_id', 'N/A')
            })
        
        if connection_data:
            df = pd.DataFrame(connection_data)
            st.dataframe(df)
        else:
            st.info("No active database connections found.")
    else:
        st.info("No database connections have been registered yet.")
    
    # If we have pool status information, display it
    has_pool_status = False
    pool_status_data = []
    
    for name, info in connections.items():
        if 'pool_status' in info:
            has_pool_status = True
            pool_status = info['pool_status']
            pool_status_data.append({
                "Owner": name,
                "Free Size": pool_status.get('free_size', 'N/A'),
                "Size": pool_status.get('size', 'N/A'),
                "Min Size": pool_status.get('min_size', 'N/A'),
                "Max Size": pool_status.get('max_size', 'N/A'),
                "Utilization (%)": round((1 - (pool_status.get('free_size', 0) / 
                                            pool_status.get('size', 1))) * 100, 2)
                                    if pool_status.get('size', 0) > 0 else 0
            })
    
    if has_pool_status and pool_status_data:
        st.markdown("### Connection Pool Status")
        pool_df = pd.DataFrame(pool_status_data)
        st.dataframe(pool_df)
        
        # Create a visualization for pool utilization
        if len(pool_status_data) > 0:
            chart_data = pd.DataFrame({
                'Owner': [d['Owner'] for d in pool_status_data],
                'Utilization': [d['Utilization (%)'] for d in pool_status_data]
            })
            
            chart = alt.Chart(chart_data).mark_bar().encode(
                x='Owner',
                y=alt.Y('Utilization:Q', scale=alt.Scale(domain=[0, 100])),
                color=alt.condition(
                    alt.datum.Utilization > 80,
                    alt.value('red'),
                    alt.value('green')
                ),
                tooltip=['Owner', 'Utilization']
            ).properties(
                title='Connection Pool Utilization',
                width=600
            )
            
            st.altair_chart(chart, use_container_width=True)

with operations_tab:
    st.markdown("### Recent Database Operations")
    
    operations = status_data.get('recent_operations', [])
    if operations:
        # Convert to DataFrame for display
        ops_data = []
        for op in operations:
            ops_data.append({
                "Timestamp": format_timestamp(op.get('timestamp')),
                "Owner": op.get('owner', 'unknown'),
                "Operation": op.get('operation', 'unknown'),
                "Duration (ms)": op.get('duration_ms', 0),
                "Success": "✓" if op.get('success', False) else "✗",
                "Details": op.get('details', '')
            })
        
        if ops_data:
            ops_df = pd.DataFrame(ops_data)
            st.dataframe(ops_df)
        else:
            st.info("No operations have been recorded yet.")
    else:
        st.info("No database operations have been recorded yet.")
    
    # Add a chart for operation duration if we have data
    if operations and len(operations) > 0:
        # Only include successful operations for the chart
        success_ops = [op for op in operations if op.get('success', False)]
        if success_ops:
            chart_data = pd.DataFrame({
                'Timestamp': [format_timestamp(op.get('timestamp')) for op in success_ops],
                'Duration (ms)': [op.get('duration_ms', 0) for op in success_ops],
                'Owner': [op.get('owner', 'unknown') for op in success_ops]
            })
            
            # Sort by timestamp for proper ordering
            chart_data = chart_data.sort_values('Timestamp')
            
            chart = alt.Chart(chart_data).mark_line(point=True).encode(
                x='Timestamp:N',
                y='Duration (ms):Q',
                color='Owner:N',
                tooltip=['Timestamp', 'Duration (ms)', 'Owner']
            ).properties(
                title='Operation Duration Over Time',
                width=600
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)

# Bottom section with controls
st.markdown("---")
st.markdown("### Monitoring Controls")

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Monitoring"):
        run_async(get_connection_monitor().start_monitoring())
        st.success("Database connection monitoring started.")

with col2:
    if st.button("Stop Monitoring"):
        run_async(get_connection_monitor().stop_monitoring())
        st.success("Database connection monitoring stopped.")

# Auto-refresh option
auto_refresh = st.checkbox("Enable auto-refresh (30s)", value=False)

if auto_refresh:
    st.warning("Auto-refresh is enabled. The page will refresh every 30 seconds.")
    time.sleep(2)  # Brief pause to show the message
    st.experimental_rerun() 