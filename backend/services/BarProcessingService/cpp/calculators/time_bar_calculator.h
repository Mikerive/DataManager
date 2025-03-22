#pragma once

#include "base_calculator.h"

/**
 * Calculator for time bars.
 * Time bars are created at fixed time intervals.
 */
class TimeBarCalculator : public BaseCalculator {
public:
    /**
     * Constructor
     */
    TimeBarCalculator() = default;
    
    /**
     * Destructor
     */
    ~TimeBarCalculator() override = default;
    
    /**
     * Calculate time bars from market data.
     * 
     * @param timestamps Vector of timestamps (in milliseconds)
     * @param opens Vector of opening prices
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of closing prices
     * @param volumes Vector of volumes
     * @param params Parameters for bar calculation (ratio specifies the time interval in seconds)
     * @return Result containing the time bars
     */
    BarResult calculate(
        const std::vector<int64_t>& timestamps,
        const std::vector<double>& opens,
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& closes,
        const std::vector<double>& volumes,
        const BarParams& params
    ) override;
}; 