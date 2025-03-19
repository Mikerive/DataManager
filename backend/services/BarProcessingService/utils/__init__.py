# Try to import from optimized Cython implementations first
try:
    from .bar_types_cy import (
        calculate_volume_bars,
        calculate_tick_bars,
        calculate_time_bars,
        calculate_price_bars,
        calculate_dollar_bars,
        calculate_entropy_bars,
        calculate_information_bars
    )
    # Log that we're using the optimized versions
    import logging
    logging.getLogger(__name__).info("Using Cython-optimized bar calculation functions")

# Fall back to pure Python implementations if Cython is not available
except ImportError:
    from .bar_types import (
        calculate_volume_bars,
        calculate_tick_bars,
        calculate_time_bars,
        calculate_price_bars,
        calculate_dollar_bars,
        calculate_entropy_bars,
        calculate_information_bars,
        calculate_shannon_entropy,
        calculate_tsallis_entropy
    )
    # Log that we're using the Python versions
    import logging
    logging.getLogger(__name__).warning("Cython-optimized bar calculation functions not available, using pure Python")

# Always import from data_utils
from .data_utils import (
    prepare_dataframe,
    validate_bar_parameters,
    enrich_bar_dataframe,
    merge_bar_dataframes
)

# Define the full list of exports
__all__ = [
    'calculate_volume_bars',
    'calculate_tick_bars',
    'calculate_time_bars',
    'calculate_price_bars',
    'calculate_dollar_bars',
    'calculate_entropy_bars',
    'calculate_information_bars',
    'calculate_shannon_entropy',
    'calculate_tsallis_entropy',
    'prepare_dataframe',
    'validate_bar_parameters',
    'enrich_bar_dataframe',
    'merge_bar_dataframes'
] 