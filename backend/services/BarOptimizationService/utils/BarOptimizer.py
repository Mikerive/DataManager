import numpy as np
import pandas as pd
import logging
import scipy.stats as stats
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, field


@dataclass
class BarOptimizationResult:
    """
    Class representing the results of bar optimization.
    
    Stores the parameter value, quality score, detailed metrics,
    and other metadata about a particular parameter configuration.
    
    Attributes:
        symbol: Trading symbol (e.g., 'AAPL')
        bar_type: Type of bar (e.g., 'volume', 'tick', 'entropy', 'price')
        ratio: The parameter value used (e.g., volume ratio, tick threshold)
        score: Overall quality score (higher is better)
        metrics: Dictionary of individual metric scores
        sample_size: Number of bars in the sample
        timestamp: When the optimization was performed
    """
    symbol: str
    bar_type: str
    ratio: float
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    
    def __post_init__(self):
        """Validate and initialize the result object after creation."""
        # Ensure metrics contains all required fields
        required_metrics = [
            'normality', 'independence', 'info_efficiency',
            'sampling_efficiency', 'variance_stability', 'prediction_power'
        ]
        
        for metric in required_metrics:
            if metric not in self.metrics:
                self.metrics[metric] = 0.0
                
        # Calculate overall score if not provided
        if self.score == 0.0 and self.metrics:
            self.score = sum(self.metrics.values()) / len(self.metrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'bar_type': self.bar_type,
            'ratio': self.ratio,
            'score': self.score,
            'metrics': self.metrics,
            'sample_size': self.sample_size,
            'timestamp': self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BarOptimizationResult':
        """Create a result object from a dictionary."""
        # Convert timestamp string to Timestamp if needed
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = pd.Timestamp(data['timestamp'])
            
        return cls(**data)
    
    def __lt__(self, other: 'BarOptimizationResult') -> bool:
        """Less than comparison based on score for sorting."""
        return self.score < other.score
        
    def get_best_metrics(self) -> List[str]:
        """Return list of metrics where this result performs best."""
        if not self.metrics:
            return []
            
        return [
            metric for metric, value in self.metrics.items() 
            if value >= 0.7 * max(self.metrics.values())
        ]
    
    def get_worst_metrics(self) -> List[str]:
        """Return list of metrics where this result performs worst."""
        if not self.metrics:
            return []
            
        return [
            metric for metric, value in self.metrics.items() 
            if value <= 0.3 * max(self.metrics.values())
        ]
    
    def format_summary(self) -> str:
        """Return a formatted summary of the optimization result."""
        strong_points = self.get_best_metrics()
        weak_points = self.get_worst_metrics()
        
        summary = [
            f"Optimization Result for {self.symbol} {self.bar_type} bars:",
            f"Parameter value: {self.ratio:.6f}",
            f"Overall score: {self.score:.4f}",
            f"Sample size: {self.sample_size} bars",
            f"Timestamp: {self.timestamp}",
            "\nDetailed metrics:"
        ]
        
        for metric, value in self.metrics.items():
            summary.append(f"  - {metric.replace('_', ' ').title()}: {value:.4f}")
            
        if strong_points:
            summary.append("\nStrong points: " + ", ".join(m.replace('_', ' ').title() for m in strong_points))
            
        if weak_points:
            summary.append("Weak points: " + ", ".join(m.replace('_', ' ').title() for m in weak_points))
            
        return "\n".join(summary)


class BarOptimizer:
    """
    Advanced bar optimizer that implements research-based metrics to evaluate bar quality.
    
    This class calculates various quantitative metrics based on financial research:
    - Normality of returns
    - Serial correlation of returns
    - Information efficiency
    - Sampling efficiency (volume/time balance)
    - Variance stability
    - Prediction accuracy
    
    References:
    - Lopez de Prado, M. (2018). Advances in Financial Machine Learning
    - Chan, E. (2013). Algorithmic Trading: Winning Strategies and Their Rationale
    - Aït-Sahalia, Y., & Jacod, J. (2014). High-Frequency Financial Econometrics
    """
    
    def __init__(self):
        """Initialize the BarOptimizer with default configuration."""
        self.logger = logging.getLogger(__name__)
        
    def evaluate_normality(self, returns: np.ndarray) -> float:
        """
        Evaluate the normality of returns using the Jarque-Bera test.
        
        The JB test measures deviation from normality based on skewness and kurtosis.
        Lower values indicate more normal distributions, which is generally desirable
        for statistical analysis and risk modeling.
        
        Args:
            returns: Array of returns calculated from bar prices
            
        Returns:
            float: Normality score (higher is better, range 0-1)
        """
        if len(returns) < 20:  # Increased minimum sample size for better statistical validity
            self.logger.debug(f"Insufficient data points for normality test: {len(returns)} < 20")
            return 0.0
            
        try:
            # Clean data - remove NaNs and Infs
            clean_returns = returns[np.isfinite(returns)]
            if len(clean_returns) < 20:
                self.logger.debug(f"Insufficient clean data points after filtering: {len(clean_returns)} < 20")
                return 0.0
                
            # Log data characteristics for diagnostics
            self.logger.debug(f"Returns stats: mean={np.mean(clean_returns):.6f}, std={np.std(clean_returns):.6f}, min={np.min(clean_returns):.6f}, max={np.max(clean_returns):.6f}")
            
            # Calculate skewness and kurtosis for diagnostics
            skew = stats.skew(clean_returns)
            kurt = stats.kurtosis(clean_returns)
            self.logger.debug(f"Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}")
            
            # Calculate Jarque-Bera statistic and p-value
            jb_stat, p_value = stats.jarque_bera(clean_returns)
            self.logger.debug(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={p_value:.6f}")
            
            # Apply a scaled transformation to the p-value to create a more nuanced score
            # For very small p-values (high non-normality), we'll still give some small positive score
            normality_score = 0.05 + 0.95 * min(p_value, 1.0)
            
            return max(0.01, normality_score)  # Ensure score is always at least slightly positive
        except Exception as e:
            self.logger.warning(f"Error in normality evaluation: {str(e)}")
            # Fall back to a simpler normality test using the moments
            try:
                clean_returns = returns[np.isfinite(returns)]
                if len(clean_returns) < 10:
                    return 0.0
                
                # Calculate moments
                skew = abs(stats.skew(clean_returns))
                kurt = abs(stats.kurtosis(clean_returns) - 3)  # Excess kurtosis compared to normal dist.
                
                # Convert to a 0-1 score (0 = highly non-normal, 1 = perfectly normal)
                # For normal distribution, skew=0 and excess_kurt=0
                skew_score = np.exp(-2 * skew)
                kurt_score = np.exp(-0.5 * kurt)
                
                return 0.5 * (skew_score + kurt_score)
            except Exception as e2:
                self.logger.warning(f"Error in fallback normality evaluation: {str(e2)}")
                return 0.1  # Return a small non-zero value as default
            
    def evaluate_serial_correlation(self, returns: np.ndarray) -> float:
        """
        Evaluate serial correlation of returns using multiple methods.
        
        Ideally, returns should be uncorrelated for efficient markets.
        Higher p-values indicate less serial correlation (more independent).
        
        Args:
            returns: Array of returns calculated from bar prices
            
        Returns:
            float: Serial correlation score (higher is better, range 0-1)
        """
        if len(returns) < 20:  # Increased minimum sample for better reliability
            self.logger.debug(f"Insufficient data points for serial correlation test: {len(returns)} < 20")
            return 0.0
            
        try:
            # Clean data - remove NaNs and Infs
            clean_returns = returns[np.isfinite(returns)]
            if len(clean_returns) < 20:
                self.logger.debug(f"Insufficient clean data points after filtering: {len(clean_returns)} < 20")
                return 0.0
            
            # Log data characteristics for diagnostics
            self.logger.debug(f"Returns stats for serial correlation: mean={np.mean(clean_returns):.6f}, std={np.std(clean_returns):.6f}")
            
            # Use multiple lag values to get a more comprehensive assessment
            lags = [1, 2, 3]
            
            # Method 1: Ljung-Box test
            try:
                lb_stats, p_values = stats.acorr_ljungbox(clean_returns, lags=lags, return_df=False)
                lb_score = min(np.mean(p_values), 1.0)  # Average p-value across lags
                self.logger.debug(f"Ljung-Box test: mean p-value={lb_score:.6f}")
            except Exception as e:
                self.logger.debug(f"Ljung-Box test failed: {str(e)}")
                lb_score = 0.0
            
            # Method 2: Simple autocorrelations
            try:
                acorr_values = [np.abs(pd.Series(clean_returns).autocorr(lag=lag)) for lag in lags if lag < len(clean_returns)]
                if acorr_values:
                    mean_acorr = np.mean(acorr_values)
                    # Transform: low autocorrelation → high score
                    acorr_score = np.exp(-5 * mean_acorr)  # Exponential transformation
                    self.logger.debug(f"Autocorrelation: mean={mean_acorr:.6f}, score={acorr_score:.6f}")
                else:
                    acorr_score = 0.0
            except Exception as e:
                self.logger.debug(f"Autocorrelation calculation failed: {str(e)}")
                acorr_score = 0.0
            
            # Combine scores with weights
            combined_score = 0.0
            weights = 0.0
            
            if lb_score > 0:
                combined_score += 0.7 * lb_score
                weights += 0.7
                
            if acorr_score > 0:
                combined_score += 0.3 * acorr_score
                weights += 0.3
                
            # Normalize by weights or return default if all methods failed
            if weights > 0:
                return combined_score / weights
            else:
                return 0.1  # Return a small non-zero value as default
                
        except Exception as e:
            self.logger.warning(f"Error in serial correlation evaluation: {str(e)}")
            return 0.1  # Return a small non-zero value as default
    
    def evaluate_information_efficiency(self, returns: np.ndarray) -> float:
        """
        Evaluate information efficiency using variance ratio.
        
        Ideal bars should capture price information efficiently, with variance
        scaling approximately linearly with time.
        
        Args:
            returns: Array of returns calculated from bar prices
            
        Returns:
            float: Information efficiency score (higher is better, range 0-1)
        """
        if len(returns) < 20:
            return 0.0
            
        try:
            # Create variance ratios at different timeframes
            chunk_size = max(5, len(returns) // 4)
            variances = []
            
            for i in range(0, len(returns), chunk_size):
                chunk = returns[i:i+chunk_size]
                if len(chunk) >= 5:  # Minimum size for meaningful variance
                    variances.append(np.var(chunk))
            
            if not variances:
                return 0.0
                
            # Calculate coefficient of variation of the variances
            # Lower CV indicates more stable variance across time
            variance_cv = np.std(variances) / np.mean(variances)
            
            # Convert to 0-1 score (higher is better)
            return np.exp(-variance_cv)
        except:
            return 0.0
            
    def evaluate_sampling_efficiency(self, bars_df: pd.DataFrame) -> float:
        """
        Evaluate efficiency of sampling in terms of volume/time balance.
        
        Good bars should create a balanced sampling that's not overly concentrated
        in high-volume periods or stretched across low-volume periods.
        
        Args:
            bars_df: DataFrame with bar data including volume and timestamps
            
        Returns:
            float: Sampling efficiency score (higher is better, range 0-1)
        """
        try:
            if bars_df.empty or len(bars_df) < 2:
                return 0.0
                
            bars_df = bars_df.copy()
            bars_df['time_delta'] = (bars_df['end_time'] - bars_df['start_time']).dt.total_seconds()
            
            # Calculate coefficient of variation (CV) for time and volume, protecting against division by zero
            time_mean = bars_df['time_delta'].mean()
            volume_mean = bars_df['volume'].mean()
            
            if time_mean <= 0 or np.isnan(time_mean):
                time_cv = 1.0  # Default to high variability when mean is invalid
            else:
                time_cv = bars_df['time_delta'].std() / time_mean
                
            if volume_mean <= 0 or np.isnan(volume_mean):
                volume_cv = 1.0  # Default to high variability when mean is invalid
            else:
                volume_cv = bars_df['volume'].std() / volume_mean
            
            # More balanced sampling has lower CV 
            # Normalize to 0-1 range (using exp(-x) to convert from "lower is better" to "higher is better")
            time_score = np.exp(-time_cv)
            volume_score = np.exp(-volume_cv)
            
            # Combine time and volume scores
            combined_score = (time_score + volume_score) / 2
            
            return max(0.0, min(1.0, combined_score))  # Ensure output is in range 0-1
        except Exception as e:
            self.logger.warning(f"Error in evaluate_sampling_efficiency: {str(e)}")
            return 0.0  # Safe default on error
            
    def evaluate_variance_stability(self, returns: np.ndarray) -> float:
        """
        Evaluate stability of variance across the bar sample.
        
        Ideal bars should maintain consistent statistical properties across different
        market regimes and time periods.
        
        Args:
            returns: Array of returns calculated from bar prices
            
        Returns:
            float: Variance stability score (higher is better, range 0-1)
        """
        if len(returns) < 20:
            return 0.0
            
        try:
            # Split returns into chunks and calculate variance in each
            chunk_size = max(5, len(returns) // 4)
            variances = []
            
            for i in range(0, len(returns), chunk_size):
                chunk = returns[i:i+chunk_size]
                if len(chunk) >= 5:  # Minimum size for meaningful variance
                    variances.append(np.var(chunk))
            
            if not variances:
                return 0.0
                
            # Calculate coefficient of variation of the variances
            # Lower CV indicates more stable variance across time
            variance_cv = np.std(variances) / np.mean(variances)
            
            # Convert to 0-1 score (higher is better)
            return np.exp(-variance_cv)
        except:
            return 0.0
            
    def evaluate_prediction_power(self, returns: np.ndarray) -> float:
        """
        Evaluate predictive power of the bars using autocorrelation of volatility.
        
        Good bars should reveal underlying volatility patterns that have
        some predictive power.
        
        Args:
            returns: Array of returns calculated from bar prices
            
        Returns:
            float: Prediction power score (higher is better, range 0-1)
        """
        if len(returns) < 30:
            return 0.0
            
        try:
            # Calculate absolute returns (proxy for volatility)
            abs_returns = np.abs(returns)
            
            # Calculate autocorrelation of absolute returns (volatility clustering)
            # Using lag-1 autocorrelation
            if len(abs_returns) > 1:
                acf = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
                
                # Convert to 0-1 score (positive autocorrelation is good for prediction)
                return max(0.0, min(acf, 1.0))
            else:
                return 0.0
        except:
            return 0.0
    
    def calculate_all_metrics(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all quality metrics for a given set of bars.
        
        Args:
            bars_df: DataFrame with bar data
            
        Returns:
            Dict mapping metric names to their values
        """
        if bars_df.empty:
            return {
                'normality': 0.0,
                'independence': 0.0,
                'info_efficiency': 0.0,
                'sampling_efficiency': 0.0,
                'variance_stability': 0.0,
                'prediction_power': 0.0,
                'overall_score': 0.0
            }
            
        # Calculate returns for metrics that need them
        if 'close' in bars_df.columns:
            returns = np.log(bars_df['close'] / bars_df['close'].shift(1)).dropna().values
        else:
            returns = np.zeros(0)
            
        # Calculate individual metrics
        normality = self.evaluate_normality(returns)
        independence = self.evaluate_serial_correlation(returns)
        info_efficiency = self.evaluate_information_efficiency(returns)
        sampling_efficiency = self.evaluate_sampling_efficiency(bars_df)
        variance_stability = self.evaluate_variance_stability(returns)
        prediction_power = self.evaluate_prediction_power(returns)
        
        # Calculate weighted overall score
        weights = {
            'normality': 0.15,
            'independence': 0.2,
            'info_efficiency': 0.2,
            'sampling_efficiency': 0.2,
            'variance_stability': 0.15,
            'prediction_power': 0.1
        }
        
        metrics = {
            'normality': normality,
            'independence': independence,
            'info_efficiency': info_efficiency,
            'sampling_efficiency': sampling_efficiency,
            'variance_stability': variance_stability,
            'prediction_power': prediction_power
        }
        
        overall_score = sum(metrics[k] * weights[k] for k in metrics)
        metrics['overall_score'] = overall_score
        
        return metrics
        
    def compare_bar_parameters(
        self, 
        symbol: str, 
        bar_type: str,
        bars_dict: Dict[float, pd.DataFrame]
    ) -> List[BarOptimizationResult]:
        """
        Compare the quality of bars generated with different parameter values.
        
        Args:
            symbol: Trading symbol (e.g. 'AAPL')
            bar_type: Type of bar ('volume', 'tick', 'entropy', 'price')
            bars_dict: Dictionary mapping parameter values to bar DataFrames
            
        Returns:
            List of BarOptimizationResult objects sorted by quality (best first)
        """
        if not bars_dict:
            logging.warning(f"No bars provided for comparison for {symbol}")
            return []
            
        results = []
        
        for ratio, bars_df in bars_dict.items():
            # Skip if no bars were generated
            if bars_df.empty:
                logging.warning(f"No bars generated for {symbol} with {bar_type} ratio {ratio}")
                continue
                
            # Extract returns for evaluation
            if 'returns' in bars_df.columns:
                returns = bars_df['returns'].values
            elif 'close' in bars_df.columns:
                # Calculate returns if not provided
                returns = np.diff(np.log(bars_df['close'].values))
            else:
                logging.warning(f"No price data found in bars for {symbol} with {bar_type} ratio {ratio}")
                continue
                
            # Skip if not enough data points
            if len(returns) < 30:
                logging.warning(f"Insufficient data for {symbol} with {bar_type} ratio {ratio}: {len(returns)} points")
                continue
            
            # Calculate metrics
            metrics = {
                'normality': self.evaluate_normality(returns),
                'independence': self.evaluate_serial_correlation(returns),
                'info_efficiency': self.evaluate_information_efficiency(returns),
                'sampling_efficiency': self.evaluate_sampling_efficiency(bars_df),
                'variance_stability': self.evaluate_variance_stability(returns),
                'prediction_power': self.evaluate_prediction_power(returns)
            }
            
            # Calculate overall score
            score = sum(metrics.values()) / len(metrics)
            
            # Create result object
            result = BarOptimizationResult(
                symbol=symbol,
                bar_type=bar_type,
                ratio=ratio,
                score=score,
                metrics=metrics,
                sample_size=len(bars_df)
            )
            
            results.append(result)
            
        # Sort results by score (descending)
        results.sort(reverse=True)
        
        return results
        
    def find_optimal_parameters(
        self, 
        symbol: str,
        bar_types: List[str],
        bars_data: Dict[str, Dict[float, pd.DataFrame]]
    ) -> Dict[str, BarOptimizationResult]:
        """
        Find optimal parameters for multiple bar types.
        
        Args:
            symbol: Symbol identifier
            bar_types: List of bar types to optimize
            bars_data: Nested dictionary mapping bar types to ratio->DataFrame mappings
            
        Returns:
            Dict mapping bar types to their optimal parameters
        """
        optimal_params = {}
        
        for bar_type in bar_types:
            if bar_type not in bars_data:
                continue
                
            bars_dict = bars_data[bar_type]
            results = self.compare_bar_parameters(symbol, bar_type, bars_dict)
            
            if results:
                optimal_params[bar_type] = results[0]  # Take the best result
                
                self.logger.info(
                    f"Optimal {bar_type} bars for {symbol}: "
                    f"ratio={results[0].ratio:.4f}, "
                    f"score={results[0].score:.4f}"
                )
            
        return optimal_params

    def bayesian_optimization(
        self,
        df: pd.DataFrame,
        bar_calculator,
        bar_type: str,
        param_range: Dict[str, Tuple[float, float]],
        n_init_points: int = 5,
        n_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize bar parameters using Bayesian optimization with Gaussian Processes.
        
        Args:
            df: DataFrame with validated price data
            bar_calculator: Calculator that can generate bars with different parameters 
            bar_type: Type of bar to optimize (e.g., 'volume', 'tick', 'entropy')
            param_range: Dictionary with parameter name as key and range tuple (min, max) as value
            n_init_points: Number of initial random points to evaluate
            n_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimization results and best parameters
        """
        from scipy.optimize import minimize
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        import numpy as np
        
        # Generate initial random points within the range
        param_name = list(param_range.keys())[0]
        min_param, max_param = param_range[param_name]
        
        # Store all evaluation results
        all_results = []
        
        # Initialize with random points
        X_samples = np.random.uniform(min_param, max_param, size=(n_init_points, 1))
        y_samples = np.zeros(n_init_points)
        
        # Evaluate initial random points
        for i, param_value in enumerate(X_samples.flatten()):
            self.logger.info(f"Evaluating initial sample {i+1}/{n_init_points}: {param_value}")
            
            # For integer parameters like tick count, round to nearest integer
            if bar_type == 'tick':
                param_value = int(round(param_value))
                
            # Generate bars using the provided parameter
            bars_df = self._generate_bars_with_param(df, bar_calculator, bar_type, param_value)
            
            # Calculate metrics for these bars
            metrics = self.calculate_all_metrics(bars_df)
            score = metrics['overall_score']
            
            all_results.append({
                'param': param_value,
                'score': score,
                'metrics': metrics
            })
            
            y_samples[i] = score
            
        # Reshape for scikit-learn
        X_samples = X_samples.reshape(-1, 1)
        
        # Initialize and fit GP regression model
        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1),
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        # Fit the model to our initial samples
        gp.fit(X_samples, y_samples)
        
        # Run Bayesian optimization iterations
        best_score = max(y_samples)
        best_param = X_samples[np.argmax(y_samples)].item()
        
        self.logger.info(f"Initial best score: {best_score} with param {best_param}")
        
        # Function to maximize (acquisition function - Expected Improvement)
        def expected_improvement(x):
            x = x.reshape(-1, 1)
            mu, sigma = gp.predict(x, return_std=True)
            
            # If sigma is zero, we don't expect improvement
            if sigma == 0:
                return 0
                
            # Expected improvement formula
            imp = mu - best_score
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei  # Negative because we're minimizing
        
        # Run iterations
        for iteration in range(n_iterations):
            self.logger.info(f"Bayesian optimization iteration {iteration+1}/{n_iterations}")
            
            # Find the next point to evaluate by maximizing the acquisition function
            x_next = None
            
            # Try different starting points to avoid getting stuck in local optima
            best_ei = float('inf')
            
            for _ in range(10):  # Try 10 different starting points
                # Random starting point within the range
                x_start = np.random.uniform(min_param, max_param)
                
                # Optimize acquisition function
                res = minimize(
                    expected_improvement,
                    x_start,
                    bounds=[(min_param, max_param)],
                    method='L-BFGS-B'
                )
                
                if res.fun < best_ei:
                    best_ei = res.fun
                    x_next = res.x[0]
            
            # For integer parameters, round to nearest integer
            if bar_type == 'tick':
                x_next = int(round(x_next))
            
            # Generate bars using the next parameter value
            bars_df = self._generate_bars_with_param(df, bar_calculator, bar_type, x_next)
            
            # Calculate metrics for these bars
            metrics = self.calculate_all_metrics(bars_df)
            score = metrics['overall_score']
            
            # Store the result
            all_results.append({
                'param': x_next,
                'score': score,
                'metrics': metrics
            })
            
            # Update our samples
            X_samples = np.vstack((X_samples, [[x_next]]))
            y_samples = np.append(y_samples, score)
            
            # Update GP model
            gp.fit(X_samples, y_samples)
            
            # Update best score and parameter
            if score > best_score:
                best_score = score
                best_param = x_next
                self.logger.info(f"New best score: {best_score} with param {best_param}")
        
        # Create final result
        optimization_result = {
            'best_param': best_param,
            'best_score': best_score,
            'all_results': all_results,
            'success': True,
            'timestamp': pd.Timestamp.now().isoformat()
        }
            
        return optimization_result
    
    def _generate_bars_with_param(self, df: pd.DataFrame, bar_calculator, bar_type: str, param_value: float) -> pd.DataFrame:
        """
        Generate bars using the specific parameter value and bar type.
        
        Args:
            df: Input raw price data
            bar_calculator: Object that can generate different types of bars
            bar_type: Type of bar to generate
            param_value: Parameter value to use
            
        Returns:
            DataFrame with generated bars
        """
        try:
            # Handle different bar types
            if bar_type == 'volume':
                return bar_calculator.process_volume_bars(df, [param_value])
            elif bar_type == 'tick':
                return bar_calculator.process_tick_bars(df, [int(param_value)])
            elif bar_type == 'entropy':
                return bar_calculator.process_entropy_bars(df, [param_value])
            elif bar_type == 'dollar':
                return bar_calculator.process_dollar_bars(df, [param_value])
            elif bar_type == 'price':
                return bar_calculator.process_price_bars(df, [param_value])
            elif bar_type == 'information':
                return bar_calculator.process_information_bars(df, [param_value])
            elif bar_type == 'time':
                return bar_calculator.process_time_bars(df, int(param_value))
            else:
                self.logger.warning(f"Unsupported bar type: {bar_type}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error generating {bar_type} bars with param {param_value}: {str(e)}")
            return pd.DataFrame()

    def random_search(
        self,
        df: pd.DataFrame,
        bar_calculator,
        bar_type: str,
        param_range: Dict[str, Tuple[float, float]],
        n_samples: int = 15
    ) -> Dict[str, Any]:
        """
        Optimize bar parameters using random search.
        
        Args:
            df: DataFrame with validated price data
            bar_calculator: Calculator that can generate bars with different parameters
            bar_type: Type of bar to optimize
            param_range: Dictionary with parameter name as key and range tuple (min, max) as value
            n_samples: Number of random samples to evaluate
            
        Returns:
            Dictionary with optimization results and best parameters
        """
        import numpy as np
        
        param_name = list(param_range.keys())[0]
        min_param, max_param = param_range[param_name]
        
        # Generate random parameter values
        params = np.random.uniform(min_param, max_param, n_samples)
        
        # For integer parameters, round to integers
        if bar_type == 'tick':
            params = np.round(params).astype(int)
            
        # Store all evaluation results
        all_results = []
        best_score = -float('inf')
        best_param = None
        
        # Evaluate all parameter values
        for i, param_value in enumerate(params):
            self.logger.info(f"Random search evaluation {i+1}/{n_samples}: {param_value}")
            
            # Generate bars using the parameter
            bars_df = self._generate_bars_with_param(df, bar_calculator, bar_type, param_value)
            
            # Calculate metrics for these bars
            metrics = self.calculate_all_metrics(bars_df)
            score = metrics['overall_score']
            
            all_results.append({
                'param': param_value,
                'score': score,
                'metrics': metrics
            })
            
            # Update best result if better
            if score > best_score:
                best_score = score
                best_param = param_value
                self.logger.info(f"New best score: {best_score} with param {best_param}")
                
        # Create final result
        optimization_result = {
            'best_param': best_param,
            'best_score': best_score,
            'all_results': all_results,
            'success': True,
            'timestamp': pd.Timestamp.now().isoformat()
        }
            
        return optimization_result
        
    def evolutionary_optimization(
        self,
        df: pd.DataFrame,
        bar_calculator,
        bar_type: str,
        param_range: Dict[str, Tuple[float, float]],
        population_size: int = 5,
        generations: int = 3
    ) -> Dict[str, Any]:
        """
        Optimize bar parameters using evolutionary optimization (differential evolution).
        
        Args:
            df: DataFrame with validated price data
            bar_calculator: Calculator that can generate bars with different parameters
            bar_type: Type of bar to optimize
            param_range: Dictionary with parameter name as key and range tuple (min, max) as value
            population_size: Size of the population for differential evolution
            generations: Number of generations to evolve
            
        Returns:
            Dictionary with optimization results and best parameters
        """
        from scipy.optimize import differential_evolution
        
        param_name = list(param_range.keys())[0]
        min_param, max_param = param_range[param_name]
        bounds = [(min_param, max_param)]
        
        # Store all evaluation results
        all_results = []
        
        # Define objective function
        def objective(params):
            param_value = params[0]
            
            # For integer parameters, round to integers
            if bar_type == 'tick':
                param_value = int(round(param_value))
                
            self.logger.info(f"Evaluating parameter: {param_value}")
            
            # Generate bars using the parameter
            bars_df = self._generate_bars_with_param(df, bar_calculator, bar_type, param_value)
            
            # Calculate metrics for these bars
            metrics = self.calculate_all_metrics(bars_df)
            score = metrics['overall_score']
            
            all_results.append({
                'param': param_value,
                'score': score,
                'metrics': metrics
            })
            
            # Return negative score because scipy.optimize minimizes
            return -score
            
        # Run differential evolution
        result = differential_evolution(
            objective, 
            bounds=bounds,
            maxiter=generations,
            popsize=population_size,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42
        )
        
        # Extract best parameter and score
        best_param = result.x[0]
        # For integer parameters, ensure final result is integer
        if bar_type == 'tick':
            best_param = int(round(best_param))
            
        best_score = -result.fun
        
        # Create final result
        optimization_result = {
            'best_param': best_param,
            'best_score': best_score,
            'all_results': all_results,
            'success': True,
            'timestamp': pd.Timestamp.now().isoformat()
        }
            
        return optimization_result

    def optimize_bar_parameters(
        self,
        df: pd.DataFrame,
        bar_calculator,
        bar_type: str,
        param_range: Dict[str, Tuple[float, float]],
        method: str = 'bayesian',
        n_init_points: int = 5,
        n_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        High-level entry point for bar parameter optimization.
        
        Args:
            df: Input DataFrame with raw price data
            bar_calculator: Object that can generate different types of bars
            bar_type: Type of bar to optimize ('volume', 'tick', 'entropy', etc.)
            param_range: Dictionary with parameter name as key and range tuple (min, max) as value
            method: Optimization method ('bayesian', 'random', 'evolutionary')
            n_init_points: Number of initial points to evaluate
            n_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        # Validate input data
        if df.empty:
            return {'success': False, 'error': 'Input DataFrame is empty'}
            
        # Choose optimization method based on parameter
        self.logger.info(f"Starting optimization using {method} method for {bar_type} bars")
        
        if method == 'bayesian':
            return self.bayesian_optimization(
                df, 
                bar_calculator,
                bar_type, 
                param_range,
                n_init_points,
                n_iterations
            )
        elif method == 'random':
            return self.random_search(
                df, 
                bar_calculator,
                bar_type, 
                param_range,
                n_samples=n_init_points + n_iterations
            )
        elif method == 'evolutionary':
            return self.evolutionary_optimization(
                df, 
                bar_calculator,
                bar_type, 
                param_range,
                population_size=n_init_points,
                generations=n_iterations
            )
        else:
            error_msg = f"Unsupported optimization method: {method}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
