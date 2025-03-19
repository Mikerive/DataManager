import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import time

from backend.db.Database import Database
from backend.db.models.RawData import RawData
from backend.db.utils.db_utils import log_db_error, log_db_success, get_tables_info


class DataIntegrityService:
    """
    Service for checking the integrity and quality of ticker data.
    
    This service performs various checks on data to ensure it meets quality standards:
    - Identifies missing time periods
    - Detects outliers in price and volume
    - Checks for data completeness
    - Verifies data consistency
    """
    
    def __init__(self):
        """Initialize the data integrity service."""
        self.logger = logging.getLogger(__name__)
        self._is_connected = False
        self._db = None
    
    async def _ensure_connected(self):
        """Ensure database connection is established."""
        # Check both the flag and whether the pool is actually valid
        if not self._is_connected or self._db is None or not hasattr(self._db, 'pool') or self._db.pool is None:
            try:
                # Create a new Database instance directly instead of using the singleton
                self._db = Database(owner_name="DataIntegrityService")
                await self._db.connect()
                self._is_connected = True
                self.logger.info("Database connection established")
            except Exception as e:
                self.logger.error(f"Error connecting to database: {str(e)}")
                raise
    
    async def cleanup(self):
        """Cleanup resources, including database connections."""
        if self._is_connected and self._db is not None:
            try:
                await self._db.close()
                self._db = None
                self._is_connected = False
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error disconnecting from database: {str(e)}")
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def analyze_ticker_data(self, ticker_or_table: str) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of ticker data quality and integrity.
        
        Args:
            ticker_or_table: Ticker symbol or table name (e.g., 'AAPL' or 'raw_data_AAPL')
            
        Returns:
            Dictionary with analysis results
        """
        try:
            start_time = time.time()
            
            await self._ensure_connected()
            
            table_name = ticker_or_table
            if not ticker_or_table.startswith('raw_data_'):
                table_name = f"raw_data_{ticker_or_table}"
            
            # Get ticker statistics
            stats = await RawData.get_ticker_statistics(table_name.replace('raw_data_', ''))
            
            if not stats['exists'] or stats['count'] == 0:
                return {
                    'ticker': ticker_or_table,
                    'exists': False,
                    'data_available': False,
                    'message': 'No data available for this ticker'
                }
            
            # Get all data for the ticker
            df = await RawData.get_price_data(
                table_name.replace('raw_data_', ''), 
                start_date=stats['min_date'],
                end_date=stats['max_date'],
                limit=100000  # Set a high limit to get as much data as possible
            )
            
            if df.empty:
                return {
                    'ticker': ticker_or_table,
                    'exists': True,
                    'data_available': False,
                    'message': 'Table exists but no data could be retrieved'
                }
            
            # Basic statistics
            results = {
                'ticker': ticker_or_table,
                'exists': True,
                'data_available': True,
                'row_count': len(df),
                'date_range': {
                    'start_date': df['timestamp'].min(),
                    'end_date': df['timestamp'].max(),
                    'days_covered': (df['timestamp'].max() - df['timestamp'].min()).days
                },
                'unique_days': df['timestamp'].dt.date.nunique(),
                'trading_days': self._count_trading_days(df['timestamp'].min(), df['timestamp'].max()),
                'price_statistics': self._calculate_price_statistics(df),
                'volume_statistics': self._calculate_volume_statistics(df),
                'data_gaps': self._identify_data_gaps(df),
                'outliers': self._identify_outliers(df),
                'data_quality_score': 0  # Will be calculated below
            }
            
            # Calculate overall data quality score
            results['data_quality_score'] = self._calculate_quality_score(df, results)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success(f"Analyze ticker data for {ticker_or_table}", duration_ms, self.logger)
            
            return results
        except Exception as e:
            log_db_error(f"Analyze ticker data for {ticker_or_table}", e, self.logger)
            return {
                'ticker': ticker_or_table,
                'exists': False,
                'data_available': False,
                'message': f'Error analyzing ticker data: {str(e)}'
            }
    
    def _count_trading_days(self, start_date: datetime, end_date: datetime) -> int:
        """Count the number of trading days (weekdays) in the date range."""
        days = (end_date - start_date).days + 1
        weeks = days // 7
        extra_days = days % 7
        
        # Count weekdays (Monday = 0, Sunday = 6)
        weekdays = min(5, extra_days)
        if start_date.weekday() > end_date.weekday():
            weekdays = max(0, weekdays - 2)
        elif start_date.weekday() + extra_days > 5:
            weekdays = max(0, 5 - start_date.weekday())
        
        return weeks * 5 + weekdays
    
    def _calculate_price_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for price data."""
        return {
            'min_price': {
                'open': df['open'].min(),
                'high': df['high'].min(),
                'low': df['low'].min(),
                'close': df['close'].min(),
            },
            'max_price': {
                'open': df['open'].max(),
                'high': df['high'].max(),
                'low': df['low'].max(),
                'close': df['close'].max(),
            },
            'mean_price': {
                'open': df['open'].mean(),
                'high': df['high'].mean(),
                'low': df['low'].mean(),
                'close': df['close'].mean(),
            },
            'median_price': {
                'open': df['open'].median(),
                'high': df['high'].median(),
                'low': df['low'].median(),
                'close': df['close'].median(),
            },
            'zero_or_negative_prices': {
                'open': (df['open'] <= 0).sum(),
                'high': (df['high'] <= 0).sum(),
                'low': (df['low'] <= 0).sum(),
                'close': (df['close'] <= 0).sum(),
            },
            'price_issues': self._check_price_issues(df)
        }
    
    def _check_price_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for common price issues like high-low inversion."""
        issues = []
        
        # Check for high < low (price inversion)
        inversion_mask = df['high'] < df['low']
        inversions = df[inversion_mask]
        if not inversions.empty:
            issues.append({
                'type': 'price_inversion',
                'description': 'High price is less than low price',
                'count': len(inversions),
                'examples': inversions.head(5).to_dict('records') if len(inversions) > 0 else []
            })
        
        # Check for open outside high-low range
        open_out_of_range = df[(df['open'] > df['high']) | (df['open'] < df['low'])]
        if not open_out_of_range.empty:
            issues.append({
                'type': 'open_out_of_range',
                'description': 'Open price is outside the high-low range',
                'count': len(open_out_of_range),
                'examples': open_out_of_range.head(5).to_dict('records') if len(open_out_of_range) > 0 else []
            })
        
        # Check for close outside high-low range
        close_out_of_range = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
        if not close_out_of_range.empty:
            issues.append({
                'type': 'close_out_of_range',
                'description': 'Close price is outside the high-low range',
                'count': len(close_out_of_range),
                'examples': close_out_of_range.head(5).to_dict('records') if len(close_out_of_range) > 0 else []
            })
        
        return issues
    
    def _calculate_volume_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for volume data."""
        return {
            'min_volume': df['volume'].min(),
            'max_volume': df['volume'].max(),
            'mean_volume': df['volume'].mean(),
            'median_volume': df['volume'].median(),
            'zero_volume_count': (df['volume'] == 0).sum(),
            'zero_volume_percentage': (df['volume'] == 0).mean() * 100,
            'negative_volume_count': (df['volume'] < 0).sum()
        }
    
    def _identify_data_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify gaps in the time series data."""
        df = df.sort_values('timestamp')
        
        # Assume 1-minute data for intraday analysis
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60
        
        # Find gaps larger than expected interval (1 minute)
        gaps = df[df['time_diff'] > 1].copy()
        
        if gaps.empty:
            return {
                'has_gaps': False,
                'total_gaps': 0,
                'largest_gap': 0,
                'total_missing_minutes': 0,
                'gaps_list': []
            }
        
        # Calculate statistics about gaps
        total_missing_minutes = gaps['time_diff'].sum() - len(gaps)
        largest_gap = gaps['time_diff'].max()
        
        # Create a list of significant gaps (more than 10 minutes)
        significant_gaps = gaps[gaps['time_diff'] > 10].copy()
        gaps_list = []
        
        for _, row in significant_gaps.iterrows():
            gap_start = row['timestamp'] - timedelta(minutes=int(row['time_diff']))
            gaps_list.append({
                'start': gap_start,
                'end': row['timestamp'],
                'duration_minutes': row['time_diff']
            })
        
        return {
            'has_gaps': True,
            'total_gaps': len(gaps),
            'largest_gap_minutes': largest_gap,
            'total_missing_minutes': total_missing_minutes,
            'gaps_list': gaps_list[:10]  # Limit to top 10 gaps
        }
    
    def _identify_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify outliers in price and volume data."""
        results = {
            'price_outliers': {},
            'volume_outliers': {},
            'total_outliers': 0
        }
        
        # Check for price outliers using Z-score
        for col in ['open', 'high', 'low', 'close']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3].copy()  # More than 3 standard deviations
            
            if not outliers.empty:
                results['price_outliers'][col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'examples': outliers.head(5).to_dict('records') if len(outliers) > 0 else []
                }
                results['total_outliers'] += len(outliers)
        
        # Check for volume outliers
        z_scores = np.abs((df['volume'] - df['volume'].mean()) / df['volume'].std())
        outliers = df[z_scores > 3].copy()
        
        if not outliers.empty:
            results['volume_outliers'] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100,
                'examples': outliers.head(5).to_dict('records') if len(outliers) > 0 else []
            }
            results['total_outliers'] += len(outliers)
        
        return results
    
    def _calculate_quality_score(self, df: pd.DataFrame, results: Dict[str, Any]) -> float:
        """Calculate an overall data quality score from 0-100."""
        score = 100.0
        
        # Deduct for missing data relative to expected trading days
        if results['trading_days'] > 0:
            coverage_ratio = results['unique_days'] / results['trading_days']
            score -= max(0, (1 - coverage_ratio) * 30)  # Deduct up to 30 points
        
        # Deduct for gaps
        if results['data_gaps']['has_gaps']:
            gap_penalty = min(25, results['data_gaps']['total_gaps'] / 10)
            score -= gap_penalty
        
        # Deduct for price issues
        price_issues_count = sum(issue['count'] for issue in results['price_statistics']['price_issues'])
        if price_issues_count > 0:
            price_penalty = min(20, price_issues_count / len(df) * 100)
            score -= price_penalty
        
        # Deduct for zero volumes
        if results['volume_statistics']['zero_volume_percentage'] > 0:
            volume_penalty = min(15, results['volume_statistics']['zero_volume_percentage'] / 5)
            score -= volume_penalty
        
        # Deduct for outliers
        if results['outliers']['total_outliers'] > 0:
            outlier_penalty = min(10, results['outliers']['total_outliers'] / len(df) * 50)
            score -= outlier_penalty
        
        return max(0, round(score, 1))
    
    async def get_tables_info(self):
        """
        Get information about all tables in the database, focusing on data tables.
        
        Returns:
            Dictionary of table information indexed by table name
        """
        try:
            start_time = time.time()
            
            await self._ensure_connected()
            
            # Use the centralized utility function from db_utils
            tables_info = await get_tables_info(self._db)
            
            duration_ms = (time.time() - start_time) * 1000
            log_db_success("Get tables info", duration_ms, self.logger)
            
            return tables_info
        except Exception as e:
            log_db_error("Get tables info", e, self.logger)
            return {} 