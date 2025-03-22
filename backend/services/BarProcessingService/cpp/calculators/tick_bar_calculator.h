#pragma once

#include "base_calculator.h"

/**
 * Calculator for tick bars.
 * Tick bars are created after a fixed number of price updates (ticks).
 */
class TickBarCalculator : public BaseCalculator {
public:
    /**
     * Constructor
     */
    TickBarCalculator() = default;
    
    /**
     * Destructor
     */
    ~TickBarCalculator() override = default;
    
    /**
     * Calculate tick bars from market data.
     * 
     * @param timestamps Vector of timestamps
     * @param opens Vector of opening prices
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of closing prices
     * @param volumes Vector of volumes
     * @param params Parameters for bar calculation (ratio specifies the tick count)
     * @return Result containing the tick bars
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