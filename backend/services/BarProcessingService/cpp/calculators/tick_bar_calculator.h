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

    /**
     * Calculate bars based on tick count
     *
     * @param timestamps Vector of timestamps
     * @param opens Vector of opening prices
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of closing prices
     * @param volumes Vector of volumes
     * @param tick_directions Vector of tick directions
     * @param params Bar parameters
     * @return Calculated bars
     */
    BarResult calculate_bars(
        const std::vector<int64_t>& timestamps,
        const std::vector<double>& opens,
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& closes,
        const std::vector<double>& volumes,
        const std::vector<double>& tick_directions,
        const BarParams& params
    );

    /**
     * Calculate tick bars with adaptive threshold
     * 
     * @param timestamps Vector of timestamps
     * @param opens Vector of open prices
     * @param highs Vector of high prices
     * @param lows Vector of low prices 
     * @param closes Vector of close prices
     * @param volumes Vector of volumes
     * @param tick_directions Vector of tick directions
     * @param threshold_ratio Multiplier for the adaptive threshold
     * @param lookback_window Number of bars to include in the lookback calculation
     * @param result BarResult to store the calculated bars
     */
    void calculate_bars_with_adaptive_threshold(
        const std::vector<int64_t>& timestamps,
        const std::vector<double>& opens,
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& closes,
        const std::vector<double>& volumes,
        const std::vector<double>& tick_directions,
        double threshold_ratio,
        size_t lookback_window,
        BarResult& result
    );
}; 