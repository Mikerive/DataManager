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
}; 