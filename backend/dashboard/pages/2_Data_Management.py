import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import time
import json
import asyncio

# Add the parent directory to sys.path so we can import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import utils
from components.enhanced_time_series import display_enhanced_chart

# Import services
from backend.services.RawDataService.RawDataService import RawDataService
from backend.services.DataIntegrityService.DataIntegrityService import DataIntegrityService

# Set page config
st.set_page_config(
    page_title="Data Management - AlgoTrader",
    page_icon="🗄️",
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
    
    if _data_integrity_service is None:
        _data_integrity_service = DataIntegrityService()
            
    return _raw_data_service is not None and _data_integrity_service is not None

# Sidebar for navigation
def show_sidebar():
    st.sidebar.markdown("## Data Management")
    
    # Initialize services
    if not initialize_services():
        st.sidebar.error("Failed to initialize services")
        return None
    
    # Navigation options
    selected_page = st.sidebar.radio(
        "Select Operation",
        options=["Download Data", "Data Integrity Check", "Download History"]
    )
    
    return selected_page

def download_data_page():
    """Page for downloading ticker data."""
    st.markdown('<div class="main-header">Download Market Data</div>', unsafe_allow_html=True)
    
    # Form for ticker input
    with st.form(key="download_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
        
        with col2:
            download_type = st.radio(
                "Download Type",
                options=["Recent Data", "Full History"],
                horizontal=True
            )
        
        with col3:
            if download_type == "Recent Data":
                days_back = st.number_input("Days Back", min_value=1, max_value=365, value=30)
            
            include_extended_hours = st.checkbox("Include Extended Hours", value=True)
        
        submit_button = st.form_submit_button(label="Download Data")
    
    # Start download when form is submitted
    if submit_button:
        if not ticker:
            st.error("Please enter a ticker symbol")
            return
        
        try:
            # Initialize progress indicators
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_container = st.empty()
            log_container = st.empty()
            result_container = st.empty()
            
            # Start the download
            with st.spinner(f"Downloading data for {ticker}..."):
                # Call the service to start the download
                full_history = (download_type == "Full History")
                days = days_back if download_type == "Recent Data" else 0
                
                download_result = utils.run_async(
                    _raw_data_service.download_ticker_data,
                    ticker=ticker,
                    days_back=days,
                    include_extended_hours=include_extended_hours,
                    full_history=full_history
                )
                
                download_id = download_result["download_id"]
                
                # Poll for updates until complete
                while True:
                    status = _raw_data_service.get_download_status(download_id)
                    progress = status.get("progress", 0)
                    status_text = status.get("status", "")
                    
                    # Update progress bar
                    progress_bar.progress(progress)
                    
                    # Update status text
                    status_html = f"""
                    <div class="download-status">
                        <p><strong>Status:</strong> {status_text}</p>
                        <p><strong>Progress:</strong> {progress}%</p>
                    </div>
                    """
                    status_container.markdown(status_html, unsafe_allow_html=True)
                    
                    # Update logs
                    logs = _raw_data_service.get_download_logs(download_id)
                    log_text = "\n".join([f"{datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} - {log.get('level', 'INFO')} - {log['message']}" for log in logs[-10:]])
                    log_container.code(log_text)
                    
                    # Check if complete
                    if status_text in ["completed", "failed"]:
                        break
                    
                    # Sleep briefly before polling again
                    time.sleep(0.5)
                
                # Show final result
                if status.get("status") == "completed":
                    result_container.success(f"Download completed for {ticker}. Added {status.get('total_records', 0)} records.")
                    
                    # Offer to check data integrity
                    if st.button("Check Data Integrity"):
                        st.session_state["check_ticker"] = ticker
                        st.experimental_rerun()
                else:
                    result_container.error(f"Download failed: {status.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

def check_data_integrity_page():
    """Page for checking data integrity."""
    st.markdown('<div class="main-header">Data Integrity Check</div>', unsafe_allow_html=True)
    
    # Form for ticker input
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Check if we have a ticker from the download page
        default_ticker = st.session_state.get("check_ticker", "AAPL")
        ticker = st.text_input("Ticker Symbol", value=default_ticker).upper()
        
        if st.button("Check Integrity"):
            if not ticker:
                st.error("Please enter a ticker symbol")
                return
            
            try:
                with st.spinner(f"Analyzing data for {ticker}..."):
                    # Run the analysis
                    results = utils.run_async(
                        _data_integrity_service.analyze_ticker_data,
                        ticker
                    )
                    
                    # Store results in session state
                    st.session_state["integrity_results"] = results
                    
            except Exception as e:
                st.error(f"Error analyzing data: {str(e)}")
                return
    
    # Display results if available
    if "integrity_results" in st.session_state:
        results = st.session_state["integrity_results"]
        
        if not results.get("data_available", False):
            st.warning(f"No data available for {results.get('ticker', ticker)}. {results.get('message', '')}")
            return
        
        # Overview metrics
        st.markdown("### Data Overview")
        
        # Data quality score with color-coded gauge
        quality_score = results.get("data_quality_score", 0)
        score_color = "red" if quality_score < 50 else "orange" if quality_score < 80 else "green"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Data Quality Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": score_color},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 80], "color": "gray"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": quality_score
                }
            }
        ))
        
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{results['row_count']:,}")
            
        with col2:
            date_range = results.get("date_range", {})
            days_covered = date_range.get("days_covered", 0)
            st.metric("Days Covered", f"{days_covered:,}")
            
        with col3:
            trading_days = results.get("trading_days", 0)
            coverage_pct = results.get("unique_days", 0) / trading_days * 100 if trading_days > 0 else 0
            st.metric("Trading Days Coverage", f"{coverage_pct:.1f}%")
            
        with col4:
            volume_stats = results.get("volume_statistics", {})
            zero_volume_pct = volume_stats.get("zero_volume_percentage", 0)
            st.metric("Zero Volume %", f"{zero_volume_pct:.1f}%")
        
        # Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["Price Statistics", "Data Gaps", "Outliers"])
        
        with tab1:
            price_stats = results.get("price_statistics", {})
            
            st.markdown("### Price Statistics")
            
            # Price range summary
            price_summary = pd.DataFrame({
                "Min": [
                    price_stats.get("min_price", {}).get("open", 0),
                    price_stats.get("min_price", {}).get("high", 0),
                    price_stats.get("min_price", {}).get("low", 0),
                    price_stats.get("min_price", {}).get("close", 0)
                ],
                "Max": [
                    price_stats.get("max_price", {}).get("open", 0),
                    price_stats.get("max_price", {}).get("high", 0),
                    price_stats.get("max_price", {}).get("low", 0),
                    price_stats.get("max_price", {}).get("close", 0)
                ],
                "Mean": [
                    price_stats.get("mean_price", {}).get("open", 0),
                    price_stats.get("mean_price", {}).get("high", 0),
                    price_stats.get("mean_price", {}).get("low", 0),
                    price_stats.get("mean_price", {}).get("close", 0)
                ],
                "Median": [
                    price_stats.get("median_price", {}).get("open", 0),
                    price_stats.get("median_price", {}).get("high", 0),
                    price_stats.get("median_price", {}).get("low", 0),
                    price_stats.get("median_price", {}).get("close", 0)
                ]
            }, index=["Open", "High", "Low", "Close"])
            
            st.dataframe(price_summary)
            
            # Price issues
            price_issues = price_stats.get("price_issues", [])
            if price_issues:
                st.markdown("### Price Issues Detected")
                
                for issue in price_issues:
                    issue_type = issue.get("type", "")
                    description = issue.get("description", "")
                    count = issue.get("count", 0)
                    
                    st.markdown(f"**{description}**: Found {count:,} occurrences")
                    
                    # Show examples if available
                    examples = issue.get("examples", [])
                    if examples:
                        examples_df = pd.DataFrame(examples)
                        st.markdown("Examples:")
                        st.dataframe(examples_df)
            else:
                st.success("No price issues detected")
        
        with tab2:
            data_gaps = results.get("data_gaps", {})
            
            st.markdown("### Data Gaps Analysis")
            
            if data_gaps.get("has_gaps", False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Gaps", f"{data_gaps.get('total_gaps', 0):,}")
                
                with col2:
                    st.metric("Largest Gap (min)", f"{data_gaps.get('largest_gap_minutes', 0):,}")
                
                with col3:
                    st.metric("Missing Minutes", f"{data_gaps.get('total_missing_minutes', 0):,}")
                
                # List significant gaps
                gaps_list = data_gaps.get("gaps_list", [])
                if gaps_list:
                    st.markdown("### Significant Gaps")
                    
                    gaps_df = pd.DataFrame([
                        {
                            "Start Time": gap.get("start"),
                            "End Time": gap.get("end"),
                            "Duration (min)": gap.get("duration_minutes")
                        }
                        for gap in gaps_list
                    ])
                    
                    st.dataframe(gaps_df)
            else:
                st.success("No significant data gaps detected")
        
        with tab3:
            outliers = results.get("outliers", {})
            
            st.markdown("### Outlier Detection")
            
            total_outliers = outliers.get("total_outliers", 0)
            if total_outliers > 0:
                st.warning(f"Detected {total_outliers:,} outliers in the data")
                
                # Price outliers
                price_outliers = outliers.get("price_outliers", {})
                if price_outliers:
                    st.markdown("#### Price Outliers")
                    
                    for col, outlier_info in price_outliers.items():
                        st.markdown(f"**{col.capitalize()}**: {outlier_info.get('count', 0):,} outliers ({outlier_info.get('percentage', 0):.2f}%)")
                        
                        examples = outlier_info.get("examples", [])
                        if examples:
                            examples_df = pd.DataFrame(examples)
                            st.dataframe(examples_df)
                
                # Volume outliers
                volume_outliers = outliers.get("volume_outliers", {})
                if volume_outliers:
                    st.markdown("#### Volume Outliers")
                    st.markdown(f"**{volume_outliers.get('count', 0):,}** outliers ({volume_outliers.get('percentage', 0):.2f}%)")
                    
                    examples = volume_outliers.get("examples", [])
                    if examples:
                        examples_df = pd.DataFrame(examples)
                        st.dataframe(examples_df)
            else:
                st.success("No significant outliers detected")

def download_history_page():
    """Page showing download history."""
    st.markdown('<div class="main-header">Download History</div>', unsafe_allow_html=True)
    
    # Get all downloads from the service
    downloads = _raw_data_service.get_download_status()
    
    if not downloads:
        st.info("No download history available yet.")
        return
    
    # Display downloads as a table
    st.markdown("### Recent Downloads")
    
    # Convert to DataFrame for display
    downloads_df = pd.DataFrame([
        {
            "Ticker": d.get("ticker", ""),
            "Start Time": datetime.fromisoformat(d.get("start_time")).strftime("%Y-%m-%d %H:%M:%S") if d.get("start_time") else "",
            "End Time": datetime.fromisoformat(d.get("end_time")).strftime("%Y-%m-%d %H:%M:%S") if d.get("end_time") else "",
            "Status": d.get("status", ""),
            "Progress": f"{d.get('progress', 0)}%",
            "Records": d.get("total_records", 0),
            "Download ID": d.get("download_id", "")
        }
        for d in downloads.values()
    ])
    
    st.dataframe(downloads_df, use_container_width=True)
    
    # Allow user to select a download to view logs
    st.markdown("### Download Logs")
    
    selected_download = st.selectbox(
        "Select Download to View Logs",
        options=downloads_df["Download ID"].tolist(),
        format_func=lambda x: f"{downloads_df[downloads_df['Download ID']==x]['Ticker'].iloc[0]} - {downloads_df[downloads_df['Download ID']==x]['Start Time'].iloc[0]}"
    )
    
    if selected_download:
        logs = _raw_data_service.get_download_logs(selected_download)
        
        if logs:
            log_text = "\n".join([f"{datetime.fromisoformat(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} - {log.get('level', 'INFO')} - {log['message']}" for log in logs])
            st.code(log_text)
        else:
            st.info("No logs available for this download.")

def main():
    """Main function for the data management dashboard."""
    try:
        # Show sidebar and get selected page
        selected_page = show_sidebar()
        
        if not selected_page:
            st.error("Failed to initialize dashboard. Check your database connection.")
            return
        
        # Show the selected page
        if selected_page == "Download Data":
            download_data_page()
        elif selected_page == "Data Integrity Check":
            check_data_integrity_page()
        elif selected_page == "Download History":
            download_history_page()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # Cleanup resources if needed
        pass
    
    # Footer
    st.markdown("---")
    st.markdown("Data Management Dashboard | Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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
                st.error(f"Failed to cleanup raw data service: {str(e)}")
                
        if _data_integrity_service is not None:
            try:
                utils.run_async(_data_integrity_service.cleanup)
            except Exception as e:
                st.error(f"Failed to cleanup data integrity service: {str(e)}")
                pass 