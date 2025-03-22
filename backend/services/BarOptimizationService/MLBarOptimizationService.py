import logging
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from backend.services.BarProcessingService.BarProcessingService import BarProcessingService
from backend.models.db.ProcessedData import ProcessedData
from backend.models.db.RawData import RawData
from backend.models.db.Tickers import Tickers
from backend.services.BarOptimizationService.utils.BarOptimizer import BarOptimizer

class MLBarOptimizationService:
    """
    Machine Learning enhanced service for optimizing and analyzing different types of bars.
    
    This class provides machine learning methods for efficient parameter searches, 
    including Bayesian optimization techniques.
    """
    
    def __init__(self, 
                 output_dir: str = "optimization_results",
                 bar_processing_service: Optional[BarProcessingService] = None,
                 raw_data_db: Optional[RawData] = None,
                 processed_data_db: Optional[ProcessedData] = None,
                 tickers_db: Optional[Tickers] = None,
                 n_init_points: int = 5,
                 n_iterations: int = 10):
        """
        Initialize the MLBarOptimizationService.
        
        Args:
            output_dir: Base directory for output files
            bar_processing_service: Service for processing bars with different parameters
            raw_data_db: Database for raw market data
            processed_data_db: Database for processed bar data
            tickers_db: Database for ticker information
            n_init_points: Number of initial random points for Bayesian optimization
            n_iterations: Number of optimization iterations
        """
        # Initialize instance variables directly without calling super().__init__
        self.output_dir = output_dir
        self.bar_processing_service = bar_processing_service
        self.raw_data_db = raw_data_db
        self.processed_data_db = processed_data_db
        self.tickers_db = tickers_db
        
        self.logger = logging.getLogger(__name__)
        self.n_init_points = n_init_points
        self.n_iterations = n_iterations
        self.scaler = MinMaxScaler()
        
        # Create an instance of BarOptimizer for the actual optimization work
        self.bar_optimizer = BarOptimizer()
        
        # Define valid bar types
        self.valid_bar_types = ['volume', 'tick', 'dollar', 'price', 'time', 'entropy', 'information']
        
        # Define default metrics
        self.default_metrics = ['bar_count', 'price_efficiency', 'serial_correlation']
    
    def optimize_bars_ml(self, 
                      df: pd.DataFrame, 
                      symbol: str,
                      bar_type: str, 
                      param_range: Dict[str, Tuple[float, float]],
                      metrics: List[str] = None,
                      method: str = 'bayesian',
                      report: bool = True) -> Dict[str, Any]:
        """
        Optimize bars using machine learning optimization techniques.
        
        Args:
            df: Raw price data
            symbol: Symbol to optimize for
            bar_type: Type of bar to optimize
            param_range: Dictionary with parameter name as key and range tuple (min, max) as value
            metrics: Metrics to evaluate (default: all available)
            method: Optimization method ('bayesian', 'random', 'evolutionary')
            report: Whether to generate a report
            
        Returns:
            Dictionary with optimization results
        """
        # Validate inputs
        if df.empty:
            error_msg = "Input DataFrame is empty"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
            
        if bar_type not in self.valid_bar_types:
            error_msg = f"Invalid bar type: {bar_type}. Valid types: {self.valid_bar_types}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
            
        if not param_range:
            error_msg = "Parameter range must be provided"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
            
        self.logger.info(f"Starting ML optimization for {symbol} {bar_type} bars")
        self.logger.info(f"Parameter range: {param_range}")
        
        try:
            # Set optimization method attribute for reporting
            self.optimization_method = method
            
            # Use the BarOptimizer to perform the actual optimization
            optimization_result = self.bar_optimizer.optimize_bar_parameters(
                df=df,
                bar_calculator=self.bar_processing_service.bar_calculator,
                bar_type=bar_type,
                param_range=param_range,
                method=method,
                n_init_points=self.n_init_points,
                n_iterations=self.n_iterations
            )
            
            if not optimization_result.get('success', False):
                return optimization_result
                
            # Add symbol and bar_type to the result
            optimization_result['symbol'] = symbol
            optimization_result['bar_type'] = bar_type
                
            # Generate report if requested
            if report:
                report_file = self._generate_ml_optimization_report(
                    symbol=symbol,
                    bar_type=bar_type,
                    optimization_result=optimization_result
                )
                optimization_result['report_file'] = report_file
                
            return optimization_result
                
        except Exception as e:
            error_msg = f"Error during ML optimization: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {'success': False, 'error': error_msg}
    
    def _generate_ml_optimization_report(self,
                                      symbol: str,
                                      bar_type: str,
                                      optimization_result: Dict[str, Any]) -> str:
        """
        Generate a report for the optimization results.
        
        Args:
            symbol: Symbol being optimized
            bar_type: Type of bar being optimized
            optimization_result: Optimization result dictionary
            
        Returns:
            Path to the generated report file
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import base64
            from io import BytesIO
            
            # Create a directory for the report if it doesn't exist
            report_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'reports', 'optimization', symbol)
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{symbol}_{bar_type}_ml_optimization_{timestamp}.html"
            report_path = os.path.join(report_dir, report_filename)
            
            # Extract data for plotting
            all_results = optimization_result.get('all_results', [])
            if not all_results:
                self.logger.warning("No results to generate report")
                return None
                
            # Extract parameters and scores
            params = [result['param'] for result in all_results]
            scores = [result['score'] for result in all_results]
            
            # Create a DataFrame for plotting
            results_df = pd.DataFrame({
                'parameter': params,
                'score': scores
            })
            
            # Get the best parameter
            best_param = optimization_result.get('best_param')
            best_score = optimization_result.get('best_score')
            
            # Generate report content
            with open(report_path, 'w') as f:
                f.write(f"""
                <html>
                <head>
                    <title>ML Bar Optimization Report: {symbol} {bar_type}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #2c3e50; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .result-box {{ 
                            border: 1px solid #ddd; 
                            padding: 15px; 
                            margin: 10px 0; 
                            border-radius: 5px;
                            background-color: #f9f9f9;
                        }}
                        .best-result {{ 
                            border: 2px solid #27ae60; 
                            background-color: #eafaf1;
                        }}
                        .metrics-table {{ 
                            border-collapse: collapse; 
                            width: 100%; 
                            margin: 15px 0;
                        }}
                        .metrics-table th, .metrics-table td {{ 
                            border: 1px solid #ddd; 
                            padding: 8px; 
                            text-align: left;
                        }}
                        .metrics-table th {{ 
                            background-color: #f2f2f2; 
                        }}
                        .plot-container {{ 
                            width: 100%; 
                            height: 400px; 
                            margin: 20px 0;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>ML Bar Optimization Report</h1>
                        <h2>Symbol: {symbol} | Bar Type: {bar_type}</h2>
                        
                        <div class="result-box best-result">
                            <h3>Best Result</h3>
                            <p>Parameter Value: <strong>{best_param}</strong></p>
                            <p>Score: <strong>{best_score:.4f}</strong></p>
                        </div>
                        
                        <h3>Optimization Method</h3>
                        <p>Method: {self.optimization_method}</p>
                        <p>Initial Points: {self.n_init_points}</p>
                        <p>Iterations: {self.n_iterations}</p>
                        
                        <h3>Parameter vs Score Plot</h3>
                        <div class="plot-container">
                            <img src="data:image/png;base64,{self._generate_parameter_plot(results_df, best_param, best_score)}" width="100%">
                        </div>
                        
                        <h3>Top 5 Results</h3>
                        <table class="metrics-table">
                            <tr>
                                <th>Rank</th>
                                <th>Parameter</th>
                                <th>Score</th>
                                <th>Metrics</th>
                            </tr>
                """)
                
                # Sort results by score and get top 5
                sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:5]
                
                # Add top 5 results to the report
                for i, result in enumerate(sorted_results):
                    metrics_str = "<br>".join([f"{k}: {v:.4f}" for k, v in result['metrics'].items()])
                    f.write(f"""
                            <tr>
                                <td>{i+1}</td>
                                <td>{result['param']}</td>
                                <td>{result['score']:.4f}</td>
                                <td>{metrics_str}</td>
                            </tr>
                    """)
                
                f.write("""
                        </table>
                    </div>
                </body>
                </html>
                """)
            
            self.logger.info(f"Optimization report generated: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating optimization report: {str(e)}")
            return None
            
    def _generate_parameter_plot(self, results_df: pd.DataFrame, best_param: float, best_score: float) -> str:
        """
        Generate a plot of parameter vs. score and encode it as base64.
        
        Args:
            results_df: DataFrame with parameters and scores
            best_param: Best parameter value
            best_score: Best score
            
        Returns:
            Base64 encoded image
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import base64
        from io import BytesIO
        
        # Set the style
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Plot parameter vs score
        ax = sns.scatterplot(x='parameter', y='score', data=results_df, alpha=0.7)
        
        # Highlight the best parameter
        plt.scatter([best_param], [best_score], color='red', s=100, label='Best Parameter')
        
        # Add a lowess trendline if we have enough points
        if len(results_df) >= 10:
            sns.regplot(x='parameter', y='score', data=results_df, scatter=False, 
                        lowess=True, line_kws={'color': 'green', 'lw': 2}, ax=ax)
        
        # Set labels and title
        plt.xlabel('Parameter Value')
        plt.ylabel('Score')
        plt.title('Parameter vs. Optimization Score')
        plt.legend()
        plt.tight_layout()
        
        # Convert plot to base64 encoding
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        # Encode the image as base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_base64
    
    def suggest_parameter_ranges(self, bar_type: str) -> Dict[str, Tuple[float, float]]:
        """
        Suggest parameter ranges for a given bar type to be used in optimization.
        
        Args:
            bar_type: Type of bar to optimize
            
        Returns:
            Dict with parameter name as key and range tuple (min, max) as value
        """
        if bar_type == 'volume':
            # Volume ratios are typically larger values
            return {'param': (5000, 100000)}
        elif bar_type == 'tick':
            # Tick counts are integers
            return {'param': (50, 1000)}
        elif bar_type == 'entropy':
            # Entropy ratios are multiples of the rolling average
            return {'param': (0.5, 5.0)}
        elif bar_type == 'price':
            # Price ratios are typically small absolute changes
            return {'param': (0.01, 0.5)}
        elif bar_type == 'dollar':
            # Dollar ratios are typically larger values
            return {'param': (10000, 500000)}
        elif bar_type == 'information':
            # Information ratios are similar to entropy
            return {'param': (0.5, 5.0)}
        elif bar_type == 'time':
            # Time bars in minutes
            return {'param': (1, 60)}
        else:
            # Default range
            return {'param': (1, 100)}

    async def apply_optimization_results(self, symbol: str, optimization_result: Dict[str, Any]) -> bool:
        """
        Apply optimization results to the bar processing service.
        
        Args:
            symbol: Symbol to apply optimization to
            optimization_result: Optimization result dictionary
            
        Returns:
            Boolean indicating success
        """
        if not optimization_result.get('success', False):
            return False
            
        bar_type = optimization_result.get('bar_type')
        param_value = optimization_result.get('best_param')
        
        if bar_type is None or param_value is None:
            return False
            
        try:
            # Apply optimization based on bar type
            if bar_type == 'volume':
                return await self.bar_processing_service.update_volume_ratio(symbol, param_value)
            elif bar_type == 'tick':
                return await self.bar_processing_service.update_tick_ratio(symbol, int(param_value))
            elif bar_type == 'entropy':
                return await self.bar_processing_service.update_entropy_ratio(symbol, param_value)
            elif bar_type == 'price':
                return await self.bar_processing_service.update_price_ratio(symbol, param_value)
            elif bar_type == 'dollar':
                return await self.bar_processing_service.update_dollar_ratio(symbol, param_value)
            elif bar_type == 'information':
                return await self.bar_processing_service.update_information_ratio(symbol, param_value)
            elif bar_type == 'time':
                return await self.bar_processing_service.update_timeframe(symbol, str(int(param_value)))
            else:
                self.logger.warning(f"Cannot apply optimization for unsupported bar type: {bar_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error applying optimization results: {str(e)}")
            return False
