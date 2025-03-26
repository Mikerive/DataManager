#pragma once

#include <vector>
#include <cstdint>
#include "../bar_params.h"
#include "../bar_result.h"

/**
 * Base interface for all bar calculators.
 * Provides a common interface for calculating bars from market data.
 */
class BaseCalculator {
public:
    /**
     * Virtual destructor for proper cleanup in derived classes
     */
    virtual ~BaseCalculator() = default;
    
    /**
     * Calculate bars from market data.
     * 
     * @param timestamps Vector of timestamps
     * @param opens Vector of opening prices
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of closing prices
     * @param volumes Vector of volumes
     * @param params Parameters for bar calculation
     * @return Result of the bar calculation
     */
    virtual BarResult calculate(
        const std::vector<int64_t>& timestamps,
        const std::vector<double>& opens,
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& closes,
        const std::vector<double>& volumes,
        const BarParams& params
    ) = 0;
    
protected:
    /**
     * Helper method to create a new bar with preserved timeframe information.
     * 
     * @param result Reference to the BarResult object to add the bar to
     * @param timestamps Vector of original timestamps
     * @param ts_idx Index of the timestamp in the original array
     * @param start_idx Index of the start time in the original array
     * @param end_idx Index of the end time in the original array
     * @param open Opening price
     * @param high Highest price
     * @param low Lowest price
     * @param close Closing price
     * @param volume Volume
     */
    void add_bar_with_preserved_timeframe(
        BarResult& result,
        const std::vector<int64_t>& timestamps,
        size_t ts_idx, size_t start_idx, size_t end_idx,
        double open, double high, double low, double close, double volume
    ) {
        result.add_bar_with_timestamps(
            ts_idx, start_idx, end_idx,
            timestamps[ts_idx], timestamps[start_idx], timestamps[end_idx],
            open, high, low, close, volume
        );
    }
}; 