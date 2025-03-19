# Bar Processing Service

This service handles the processing of raw financial data into various types of bars (OHLCV) using different methodologies.

## Structure

The service is organized as follows:

```
BarProcessingService/
├── BarProcessingService.py    # Main service class
├── utils/                     # Utility functions for different bar calculations
│   ├── __init__.py            # Package exports
│   ├── bar_types.py           # Bar calculation functions for each type
│   └── data_utils.py          # Data preparation and validation utilities
├── components/                # Optional additional components
└── tests/                     # Unit tests
```

## Bar Types

The service supports various bar types:

1. **Volume Bars**: Based on cumulative volume thresholds
2. **Tick Bars**: Based on fixed number of price updates
3. **Time Bars**: Regular time-based bars (e.g., 1-minute, 5-minute)
4. **Price Bars**: Based on absolute price changes
5. **Dollar Bars**: Based on cumulative dollar volume
6. **Entropy Bars**: Based on information content (Shannon or Tsallis entropy)
7. **Information Bars**: Based on cumulative absolute log returns

## Usage

### Basic Example

```python
# Initialize the service for a specific ticker
service = BarProcessingService(ticker="AAPL")

# Load data for a specific time range
await service.load_data(
    start_time=datetime(2022, 1, 1),
    end_time=datetime(2022, 12, 31)
)

# Process volume bars with a ratio of 10
volume_bars = await service.process_bar(bar_type="volume", ratio=10)

# Process multiple bar types with different ratios
results = await service.process_multiple_bars({
    "volume": [10, 20, 50],
    "tick": [100, 200],
    "time": [5, 15, 60]  # 5-min, 15-min, 60-min
})
```

### Accessing and Managing Data

```python
# Change ticker and load new data
await service.change_ticker("MSFT", load_data=True, 
                          start_time=datetime(2022, 1, 1),
                          end_time=datetime(2022, 12, 31))

# Update calculation parameters
await service.update_params(window_size=200, calculation_method="tsallis")

# Get statistics about cached data
stats = await service.get_statistics()

# Clear cache to free memory
service.clear_cache()
```

## Advanced Usage

For entropy bars with specific parameters:

```python
entropy_bars = await service.process_bar(
    bar_type="entropy", 
    ratio=5,
    window_size=150,  # Override default window size
    method="shannon",  # Use Shannon entropy method
    avg_window=300     # Use 300-period average window
)
```

## Notes

- Raw data and processed bars are cached in memory for efficiency
- Processed bars are also cached to avoid redundant calculations
- The service automatically handles data validation and preparation
- All bar calculations use optimized numpy operations for performance 

## Cython Optimization

For high-performance bar calculations, the service includes Cython-optimized implementations of all bar type calculations. These offer significant performance improvements, especially for entropy and information bars which require complex calculations.

### Compiling the Cython Extensions

To compile the Cython extensions:

1. Ensure you have the required dependencies:
   ```bash
   pip install cython numpy setuptools
   ```

2. From the `BarProcessingService` directory, run:
   ```bash
   python setup.py build_ext --inplace
   ```

3. The compiled extensions will be placed in the `utils` directory and automatically used by the service.

### Benefits of Cython Optimization

- **Performance**: 10-100x faster bar calculations compared to pure Python
- **Reduced Memory**: More efficient memory usage for large datasets
- **CPU Utilization**: Better use of CPU cache and vectorized operations
- **GIL Release**: Allows for true parallel execution in multi-threaded environments

### Fallback Mechanism

If the Cython extensions are not available (e.g., not compiled or on an unsupported platform), the service will automatically fall back to the pure Python implementations.

```python
# Check if using optimized functions
import logging
logging.basicConfig(level=logging.INFO)
from backend.services.BarProcessingService.utils import calculate_volume_bars
# Will log whether using Cython or Python implementation
``` 