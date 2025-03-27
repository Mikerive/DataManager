# BarProcessingService C++ Implementation

This directory contains the C++ implementation of the bar calculation functionality for the BarProcessingService. The implementation provides high-performance calculations for various types of bars used in financial time series analysis.

## Architecture Overview

```
┌─────────────────┐          ┌─────────────────┐
│  Python Layer   │          │  bar_params.h   │
│  (Wrapper)      │◄────────►│  (Parameters)   │
└────────┬────────┘          └─────────────────┘
         │
         ▼
┌─────────────────┐          ┌─────────────────┐
│ bar_calculator.h│          │   bar_result.h  │
│  (Main Logic)   │◄────────►│  (Results)      │
└────────┬────────┘          └────────┬────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────┐
│             Specialized Calculators          │
├─────────────┬─────────────┬─────────────────┤
│  Volume     │    Tick     │     Time        │
│  Calculator │  Calculator │   Calculator    │
├─────────────┴─────────────┴─────────────────┤
│           Entropy Calculator                 │
│           (Uses Utilities)                   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│              Utility Functions               │
│     (entropy_utils.h/cpp: optimized          │
│      window calculations)                    │
└─────────────────────────────────────────────┘
```

## Data Flow

1. **Input Data Flow**:
   - Raw market data (timestamps, OHLCV) flows from Python → bar_calculator_wrapper.py → C++ BarCalculator
   - BarCalculator stores data in memory-efficient vectors
   - Calculation parameters flow through BarParams objects

2. **Processing Flow**:
   - BarCalculator routes data to specialized calculators based on bar type
   - Each calculator implements optimized algorithms for its specific bar type
   - Moving windows are calculated in-place to avoid unnecessary memory operations
   - Results are accumulated in BarResult objects

3. **Output Flow**:
   - BarResult objects convert C++ data to Python-compatible formats
   - Result data flows back to Python as DataFrames through wrapper functions

## Directory Structure

```
cpp/
├── bar_calculator.h          # Main calculator class header
├── bar_calculator.cpp        # Main calculator class implementation
├── bar_params.h              # Parameters for bar calculations
├── bar_result.h              # Result structure for bar data
├── bar_result.cpp            # Result implementation
├── bar_calculator_module.cpp # Python binding module
├── calculators/              # Specialized bar calculators
│   ├── base_calculator.h     # Base interface for all calculators
│   ├── volume_bar_calculator.h
│   ├── volume_bar_calculator.cpp
│   ├── tick_bar_calculator.h
│   ├── tick_bar_calculator.cpp
│   ├── time_bar_calculator.h
│   ├── time_bar_calculator.cpp
│   ├── entropy_bar_calculator.h
│   └── entropy_bar_calculator.cpp
└── utils/                    # Utility functions
    ├── entropy_utils.h       # Entropy calculation utilities
    └── entropy_utils.cpp     # Entropy calculation implementation
```

## File Roles and Data Flow

### Core Components

- **`bar_calculator.h/cpp`**: 
  - **Data Flow**: Receives raw market data from Python → distributes to specialized calculators → collects results
  - **Role**: Central hub managing all calculation types and data transfer between Python and C++

- **`bar_params.h`**: 
  - **Data Flow**: Python parameters → C++ calculation configuration
  - **Role**: Structure defining all parameters needed for different bar calculations

- **`bar_result.h/cpp`**: 
  - **Data Flow**: Calculated bar data → Python-compatible format
  - **Role**: Stores calculated bar data and handles conversion to Python objects

### Specialized Calculators

- **`volume_bar_calculator.h/cpp`**: 
  - **Data Flow**: Receives price and volume data → accumulates volumes → creates bars when threshold reached
  - **Role**: Specialized implementation for volume-based bars

- **`tick_bar_calculator.h/cpp`**: 
  - **Data Flow**: Receives price data → counts ticks → creates bars after fixed number of ticks
  - **Role**: Specialized implementation for tick-based bars

- **`time_bar_calculator.h/cpp`**: 
  - **Data Flow**: Receives time and price data → segments by time intervals → aggregates price data
  - **Role**: Specialized implementation for time-based bars

- **`entropy_bar_calculator.h/cpp`**: 
  - **Data Flow**: Receives price data → calculates price changes → computes rolling entropy → creates bars when threshold reached
  - **Role**: Specialized implementation for entropy-based bars, with optimized moving window calculations

### Utility Functions

- **`entropy_utils.h/cpp`**: 
  - **Data Flow**: Receives price changes → computes various entropy measures using optimized window algorithms
  - **Role**: Contains optimized implementations of entropy calculations using efficient moving window techniques

## Moving Window Optimizations

The implementation employs several key optimizations for calculations involving moving windows:

### 1. In-place Window Calculations

Rather than repeatedly copying data for each window calculation, the implementation uses in-place window processing:

```
Window 1: [a b c d e] → Calculate
Window 2:   [b c d e f] → Calculate 
```

This is implemented efficiently by:
```cpp
// Instead of creating a new vector for each window
for (size_t i = window_size; i < data_size; i++) {
    // Use the existing data with different start/end indices
    double metric = calculate_for_window(&data[i - window_size + 1], window_size);
    // Use the calculated metric
}
```

### 2. Reuse of Intermediate Calculations

For entropy calculations, intermediate values are reused when possible:

- Price changes are calculated once and reused for all entropy calculations
- Absolute values are computed once per window
- Probability distributions are computed efficiently by normalizing window values

### 3. Memory Access Optimization

The implementation carefully manages memory access patterns:

- Data is stored contiguously in vectors for cache-friendly access
- Pointers to arrays are used for fast direct access when calculating window statistics
- Batch processing combines multiple calculations to maximize data locality

### 4. Batch Processing

The `batch_process` method allows processing multiple bar types in a single pass through the data:

```
       ┌─> Volume Bars
Data ──┼─> Tick Bars
       ├─> Time Bars
       └─> Entropy Bars
```

This avoids multiple passes through the same data, significantly improving performance for multi-bar analysis.

## Integration with Python

The C++ implementation is integrated with Python using pybind11. The `bar_calculator_module.cpp` file defines the Python bindings, exposing the C++ classes and functions to Python.

Python code can use this implementation through the `bar_calculator_wrapper.py` module, which provides a user-friendly interface to the C++ functionality.

## Performance Considerations

This C++ implementation is designed for high performance when processing large datasets:

1. **Efficient Memory Management**: Minimizes unnecessary data copying and allocations.
2. **Batch Processing**: Processes multiple bar types in a single pass over the data.
3. **Optimized Algorithms**: Uses efficient algorithms for bar calculations, especially for entropy calculations.
4. **Moving Window Optimizations**: Implements efficient moving window calculations that avoid redundant operations.

## Building and Using the C++ Code

To build the C++ extension:

1. Make sure you have a C++ compiler installed (MSVC for Windows, GCC/Clang for Unix).
2. Install pybind11: `pip install pybind11`
3. Run the setup script: `python setup.py build_ext --inplace`

### Using in Python

After building, you can use the C++ implementation in Python:

```python
from backend.services.BarProcessingService.bar_calculator_wrapper import BarCalculator
import pandas as pd

# Create a BarCalculator instance
calculator = BarCalculator()

# Set the data (using a DataFrame with OHLCV columns)
calculator.set_data(my_dataframe)

# Calculate various bar types
volume_bars = calculator.calculate_volume_bars(1000)
tick_bars = calculator.calculate_tick_bars(50)
time_bars = calculator.calculate_time_bars(300)  # 5-minute bars
entropy_bars = calculator.calculate_entropy_bars(
    0.8, window_size=100, method="shannon"
)

# Batch process multiple bar types
bars_dict = calculator.batch_calculate([
    {'bar_type': 'volume', 'ratio': 1000},
    {'bar_type': 'tick', 'ratio': 50},
    {'bar_type': 'time', 'ratio': 300},
    {'bar_type': 'entropy', 'ratio': 0.8, 'window_size': 100, 'method': 'shannon'}
])
```

## Troubleshooting

If you encounter issues building the extension:

1. Check that pybind11 is installed.
2. Ensure you have a compatible C++ compiler (supporting C++14 or later).
3. On Windows, make sure Visual Studio build tools are installed and in the PATH.

For runtime issues:
1. Ensure the extension is built with the correct Python version.
2. Check for compatibility issues between the C++ extension and the Python interpreter.

## Adaptive Thresholds

All bar types (except Time Bars) use adaptive thresholds calculated from the average of recent data:

- **Volume Bars**: The threshold is the average volume of the last N bars multiplied by a ratio
- **Tick Bars**: The threshold is the average tick count of the last N bars multiplied by a ratio
- **Entropy Bars**: The threshold is the average entropy of the last N bars multiplied by a ratio

### Parameters

- `ratio`: The multiplier applied to the calculated average (default: 1.0)
- `lookback_window`: The number of previous bars to include in the average calculation (default: 20)

## Example Usage

```cpp
// Set parameters for volume bars
BarParams params(BarType::Volume, 1.5);  // 1.5x the average volume
params.lookback_window = 20;  // Use last 20 bars for average

// Calculate volume bars
VolumeBarCalculator calculator;
BarResult result = calculator.calculate(timestamps, opens, highs, lows, closes, volumes, params);
```

## Adding New Bar Types

To add a new bar type:

1. Add the type to the `BarType` enum in `bar_params.h`
2. Create a new calculator class that inherits from `BaseCalculator`
3. Implement the `calculate` method
4. Use the `calculate_adaptive_threshold` method to determine the threshold based on historical data

## Timestamp Preservation

The implementation preserves timestamp information throughout the bar calculation process. Each bar contains:

- `timestamp`: The timestamp of the bar (typically the last data point in the bar)
- `start_time`: The timestamp of the first data point in the bar
- `end_time`: The timestamp of the last data point in the bar

This allows for accurate backtesting and analysis. 