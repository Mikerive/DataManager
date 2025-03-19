import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils

# Import services
from backend.services.RawDataService.RawDataService import RawDataService
from backend.services.DataIntegrityService.DataIntegrityService import DataIntegrityService

# Set page config
st.set_page_config(
    page_title="AlgoTrader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
st.markdown(utils.load_css(), unsafe_allow_html=True)

# Initialize services
_raw_data_service = None
_data_integrity_service = None

def initialize_services():
    """Initialize service instances if they haven't been already."""
    global _raw_data_service, _data_integrity_service
    
    if _raw_data_service is None:
        # Get service instances if they were initialized in utils
        _raw_data_service = utils._raw_data_service
        
        # If services are still not initialized, create them
        if _raw_data_service is None:
            _raw_data_service = RawDataService()
            utils._raw_data_service = _raw_data_service
    
    if _data_integrity_service is None:
        _data_integrity_service = DataIntegrityService()
        utils._data_integrity_service = _data_integrity_service
            
    return _raw_data_service is not None and _data_integrity_service is not None

def main():
    # Main header
    st.markdown('<div class="main-header">AlgoTrader Dashboard</div>', unsafe_allow_html=True)
    
    try:
        # Initialize services
        if not initialize_services():
            st.error("Failed to initialize services")
            return
        
        # Explicitly ensure both services are connected before proceeding
        # This addresses the "NoneType has no attribute 'acquire'" error when switching between pages
        utils.run_async(utils.ensure_service_connected, _data_integrity_service)
        utils.run_async(utils.ensure_service_connected, _raw_data_service)
        
        # Get database info - now using DataIntegrityService
        tables_info = utils.run_async(_data_integrity_service.get_tables_info)
        
        # Extract raw and processed data tables
        raw_data_tables = [t for t in tables_info.keys() if t.startswith('raw_data_') and t != 'raw_data_template']
        processed_data_tables = [t for t in tables_info.keys() if t.startswith('processed_')]
        
        # Dashboard cards - 2 columns
        col1, col2 = st.columns(2)
        
        # Raw Data Dashboard Card
        with col1:
            st.markdown('<div class="sub-header">Raw Market Data</div>', unsafe_allow_html=True)
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            
            # Count of raw data tables
            raw_count = len(raw_data_tables)
            raw_rows = sum(tables_info[t].get('row_count', 0) for t in raw_data_tables) if tables_info else 0
            
            st.markdown(f"### {raw_count} Raw Data Tables")
            st.markdown(f"**{raw_rows:,}** total price records")
            
            # List tickers
            tickers = [t.replace('raw_data_', '').upper() for t in raw_data_tables]
            st.markdown("**Available Tickers:** " + ", ".join(tickers))
            
            st.markdown("""
            * OHLCV tick data visualization  
            * Volume analysis
            * Data quality monitoring
            """)
            
            # Button to navigate to raw data page
            st.markdown('<a href="./pages/1_Raw_Data" target="_self"><button style="background-color:#1E88E5; color:white; border:none; border-radius:5px; padding:10px 20px; cursor:pointer;">Explore Raw Data</button></a>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Processed Data Dashboard Card
        with col2:
            st.markdown('<div class="sub-header">Processed Data</div>', unsafe_allow_html=True)
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            
            # Count of processed data tables
            proc_count = len(processed_data_tables)
            proc_rows = sum(tables_info[t].get('row_count', 0) for t in processed_data_tables) if tables_info else 0
            
            st.markdown(f"### {proc_count} Processed Data Tables")
            st.markdown(f"**{proc_rows:,}** total feature records")
            
            # List types
            bar_types = set()
            for table in processed_data_tables:
                parts = table.split('_')
                if len(parts) >= 4:  # Check if there are at least 4 parts
                    bar_types.add(parts[3])
            
            # Only show bar types if we found any
            if bar_types:
                st.markdown("**Bar Types:** " + ", ".join(bar_types))
            else:
                st.markdown("**Bar Types:** None found")
            
            st.markdown("""
            * Feature exploration
            * Time/Tick/Volume bars  
            * Statistical properties
            """)
            
            # Button to navigate to processed data page
            st.markdown('<a href="./pages/3_Processed_Data" target="_self"><button style="background-color:#FF5722; color:white; border:none; border-radius:5px; padding:10px 20px; cursor:pointer;">Explore Processed Data</button></a>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add Data Management Card
        st.markdown('<div class="sub-header">Data Management</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### Download Market Data")
            st.markdown("""
            * Download ticker data from Tiingo
            * View download progress in real-time
            * Download recent data or full history
            """)
            st.markdown('<a href="./pages/2_Data_Management" target="_self"><button style="background-color:#10B981; color:white; border:none; border-radius:5px; padding:10px 20px; cursor:pointer;">Manage Data</button></a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### Data Integrity Checks")
            st.markdown("""
            * Verify data quality & completeness
            * Identify gaps in price history
            * Detect anomalies & outliers
            """)
            st.markdown('<a href="./pages/2_Data_Management?section=Data+Integrity+Check" target="_self"><button style="background-color:#8B5CF6; color:white; border:none; border-radius:5px; padding:10px 20px; cursor:pointer;">Check Data Integrity</button></a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # System Overview
        st.markdown('<div class="sub-header">System Overview</div>', unsafe_allow_html=True)
        
        # Display database stats in 3 metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Tables", len(tables_info) if tables_info else 0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            total_rows = sum(tables_info[t].get('row_count', 0) for t in tables_info.keys()) if tables_info else 0
            st.metric("Total Data Points", f"{total_rows:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent Downloads
        st.markdown('<div class="sub-header">Recent Downloads</div>', unsafe_allow_html=True)
        
        # Get actual download history from service
        downloads = _raw_data_service.get_download_status()
        
        if downloads:
            # Convert to DataFrame for display
            updates_df = pd.DataFrame([
                {
                    "Timestamp": datetime.fromisoformat(d.get("start_time")).strftime("%Y-%m-%d %H:%M:%S") if d.get("start_time") else "",
                    "Ticker": d.get("ticker", "").upper(),
                    "Status": d.get("status", ""),
                    "Records": f"{d.get('total_records', 0):,}",
                    "Description": f"Downloaded {d.get('ticker', '').upper()} data"
                }
                for d in downloads[:5]  # Show only the 5 most recent
            ])
            st.dataframe(updates_df, use_container_width=True, hide_index=True)
        else:
            # Placeholder for recent updates
            updates = [
                {"Timestamp": "2025-03-12 12:36:00", "Description": "Created Volume-based chart visualizations"},
                {"Timestamp": "2025-03-12 12:00:00", "Description": "Added extended hours support to data fetching"},
                {"Timestamp": "2025-03-11 15:30:00", "Description": "Initialized database with AAPL, MSFT, GOOG, AMZN, META"}
            ]
            
            updates_df = pd.DataFrame(updates)
            st.dataframe(updates_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("AlgoTrader Dashboard | Data updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.markdown('<p class="data-source">Data source: Tiingo | Dashboard created with Streamlit</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Cleanup resources if the app is shutting down
        if '_raw_data_service' in globals() and _raw_data_service is not None:
            try:
                utils.run_async(_raw_data_service.cleanup)
            except Exception as e:
                st.error(f"Failed to cleanup raw data service: {str(e)}")
                
        if '_data_integrity_service' in globals() and _data_integrity_service is not None:
            try:
                utils.run_async(_data_integrity_service.cleanup)
            except Exception as e:
                st.error(f"Failed to cleanup data integrity service: {str(e)}")
                pass 