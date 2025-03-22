#pragma once

#include "base_calculator.h"

/**
 * Calculator for volume bars.
 * Volume bars are created when the cumulative volume since the last bar exceeds a threshold.
 */
class VolumeBarCalculator : public BaseCalculator {
public:
    /**
     * Constructor
     */
    VolumeBarCalculator() = default;
    
    /**
     * Destructor
     */
    ~VolumeBarCalculator() override = default;
    
    /**
     * Calculate volume bars from market data.
     * 
     * @param timestamps Vector of timestamps
     * @param opens Vector of opening prices
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of closing prices
     * @param volumes Vector of volumes
     * @param params Parameters for bar calculation (ratio specifies the volume threshold)
     * @return Result containing the volume bars
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