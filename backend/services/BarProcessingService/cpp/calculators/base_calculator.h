#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include "../bar_params.h"
#include "../bar_result.h"
#include "../utils/adaptive_threshold_calculator.h"

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
     * Set the ticker ID for caching purposes
     * 
     * @param ticker_id The ticker identifier
     */
    void set_ticker_id(const std::string& ticker_id) {
        ticker_id_ = ticker_id;
    }
    
    /**
     * Get the current ticker ID
     * 
     * @return The current ticker ID
     */
    const std::string& get_ticker_id() const {
        return ticker_id_;
    }
    
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
    
    /**
     * Calculate an adaptive volume threshold using the shared calculator
     * 
     * @param current_volumes Vector of volume values
     * @param current_idx Current position in the data 
     * @param lookback_window Window size for lookback period
     * @param ratio Multiplier for the threshold
     * @return The adaptive threshold or 0 if in initialization period
     */
    double calculate_adaptive_volume_threshold(
        const std::vector<double>& current_volumes,
        size_t current_idx,
        size_t lookback_window,
        double ratio) 
    {
        // Get the shared calculator instance
        AdaptiveThresholdCalculator& calculator = AdaptiveThresholdCalculator::getInstance();
        
        // Calculate with caching if we have a ticker ID
        return calculator.calculate_volume_threshold(
            current_volumes, current_idx, lookback_window, ratio, this->ticker_id_
        );
    }
    
    /**
     * Calculate an adaptive tick threshold using the shared calculator
     * 
     * @param current_ticks Vector of tick values
     * @param current_idx Current position in the data 
     * @param lookback_window Window size for lookback period
     * @param ratio Multiplier for the threshold
     * @return The adaptive threshold or 0 if in initialization period
     */
    double calculate_adaptive_tick_threshold(
        const std::vector<double>& current_ticks,
        size_t current_idx,
        size_t lookback_window,
        double ratio) 
    {
        // Get the shared calculator instance 
        AdaptiveThresholdCalculator& calculator = AdaptiveThresholdCalculator::getInstance();
        
        // Calculate with caching if we have a ticker ID
        return calculator.calculate_tick_threshold(
            current_ticks, current_idx, lookback_window, ratio, this->ticker_id_
        );
    }
    
    /**
     * Calculate an adaptive entropy threshold using the shared calculator
     * 
     * @param current_entropy Vector of entropy values
     * @param current_idx Current position in the data 
     * @param lookback_window Window size for lookback period
     * @param ratio Multiplier for the threshold
     * @return The adaptive threshold or -1 if in initialization period
     */
    double calculate_adaptive_entropy_threshold(
        const std::vector<double>& current_entropy,
        size_t current_idx,
        size_t lookback_window,
        double ratio) 
    {
        // Get the shared calculator instance
        AdaptiveThresholdCalculator& calculator = AdaptiveThresholdCalculator::getInstance();
        
        // Calculate with caching if we have a ticker ID
        return calculator.calculate_entropy_threshold(
            current_entropy, current_idx, lookback_window, ratio, this->ticker_id_
        );
    }
    
protected:
    /**
     * Ticker identifier for caching purposes
     */
    std::string ticker_id_;
    
    /**
     * Calculate an adaptive threshold based on the average of the last X values.
     * 
     * @param values Vector of values to average
     * @param current_idx Current position in the vector
     * @param lookback_window Number of previous values to include in the average
     * @param ratio Multiplier to apply to the average
     * @return Calculated adaptive threshold
     */
    double calculate_adaptive_threshold(
        const std::vector<double>& values,
        size_t current_idx,
        int lookback_window,
        double ratio
    ) {
        // Ensure we don't try to look back before the start of data
        int effective_window = std::min(
            lookback_window, 
            static_cast<int>(current_idx + 1)
        );
        
        if (effective_window <= 0) {
            return ratio; // Not enough data, use ratio as is
        }
        
        // Calculate the sum for the average
        double sum = 0.0;
        
        // Use std::accumulate for efficiency when possible
        if (effective_window <= static_cast<int>(current_idx + 1)) {
            sum = std::accumulate(
                values.begin() + (current_idx - effective_window + 1),
                values.begin() + current_idx + 1,
                0.0
            );
        } else {
            // Fallback for cases where we can't use a simple range
            for (int i = 0; i < effective_window; ++i) {
                if (current_idx >= i) {
                    sum += values[current_idx - i];
                }
            }
        }
        
        // Calculate average and apply ratio
        return (sum / effective_window) * ratio;
    }
    
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