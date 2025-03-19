# Bar Processing Service Utilities

This directory contains utility functions for the Bar Processing Service, including both Python and Cython-optimized implementations.

## Contents

- `__init__.py`: Package initialization that prioritizes Cython imports when available
- `bar_types.py`: Pure Python implementations of bar calculation functions
- `bar_types_cy.pyx`: Cython-optimized implementations of bar calculation functions
- `data_utils.py`: Utility functions for data preparation and validation

## Using Cython Optimizations

The package automatically attempts to use Cython-optimized functions when available. If the Cython extensions are not built, it falls back to the pure Python implementations.

### Optimizations Applied

The Cython implementation uses several optimization techniques:

1. **Static Typing**: Data types are explicitly declared to avoid Python's dynamic typing overhead
2. **Bounds Checking Removal**: Unnecessary bounds checking is disabled with `@cython.boundscheck(False)`
3. **Wrap-Around Behavior Removal**: Negative indexing is disabled with `@cython.wraparound(False)`
4. **Fast Division**: C-style division is enabled with `@cython.cdivision(True)`
5. **Memory Views**: NumPy arrays are accessed via memory views for direct memory access
6. **C Math Functions**: Uses C math functions via `libc.math` instead of Python/NumPy functions

### Benchmarking

Performance improvements vary by bar type and data size:

| Bar Type | Typical Speedup |
|----------|----------------|
| Volume Bars | 5-10x |
| Tick Bars | 5-15x |
| Price Bars | 10-20x |
| Dollar Bars | 10-20x |
| Entropy Bars | 30-100x |
| Information Bars | 20-50x |

The largest speedups are seen with entropy bars due to the computational complexity of entropy calculations.

### Implementation Notes

- Time bars cannot be fully optimized with Cython due to their reliance on pandas' `resample` function
- All Cython functions provide a compatible API with their Python counterparts
- Python wrapper functions handle the pandas DataFrame interface for ease of use 