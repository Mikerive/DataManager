#pragma once

#include "base_calculator.h"
#include <string>

/**
 * Calculator for entropy bars.
 * Entropy bars are created when the price change entropy exceeds a threshold.
 */
class EntropyBarCalculator : public BaseCalculator {
public:
    /**
     * Constructor
     */
    EntropyBarCalculator() = default;
    
    /**
     * Destructor
     */
    ~EntropyBarCalculator() override = default;
    
    /**
     * Calculate entropy bars from market data.
     * 
     * @param timestamps Vector of timestamps
     * @param opens Vector of opening prices
     * @param highs Vector of high prices
     * @param lows Vector of low prices
     * @param closes Vector of closing prices
     * @param volumes Vector of volumes
     * @param params Parameters for bar calculation (ratio specifies the entropy threshold,
     *               window_size for the rolling window, method for entropy type)
     * @return Result containing the entropy bars
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

private:
    /**
     * Calculate the rolling entropy for the entire price series.
     * 
     * @param prices Vector of closing prices
     * @param window_size Size of the rolling window
     * @param method Entropy calculation method ("shannon" or "tsallis")
     * @param q_param Tsallis q-parameter (default: 1.5)
     * @return Vector of entropy values
     */
    std::vector<double> calculate_rolling_entropy(
        const std::vector<double>& prices,
        size_t window_size,
        const std::string& method,
        double q_param = 1.5
    );
}; 