#include "time_bar_calculator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

BarResult TimeBarCalculator::calculate(
    const std::vector<int64_t>& timestamps,
    const std::vector<double>& opens,
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    const BarParams& params
) {
    // Validate parameters
    if (timestamps.empty() || timestamps.size() != closes.size()) {
        throw std::runtime_error("Invalid data for time bar calculation");
    }
    
    if (params.ratio <= 0) {
        throw std::runtime_error("Time interval must be positive");
    }
    
    // Initialize result
    BarResult result(params.bar_type, params.ratio);
    
    // Since the timestamps are now just indices, we need to use data point count as our interval
    // The ratio parameter is the number of data points (minutes) to include in each bar
    int64_t data_points_per_bar = static_cast<int64_t>(params.ratio);
    
    // Loop through data in chunks of data_points_per_bar
    size_t bar_start_idx = 0;
    double bar_volume = 0.0;
    
    for (size_t i = 0; i < timestamps.size(); ++i) {
        // Accumulate volume
        bar_volume += volumes[i];
        
        // Check if we've reached the interval (number of data points) or end of data
        bool is_last = (i == timestamps.size() - 1);
        bool interval_complete = ((i - bar_start_idx + 1) >= data_points_per_bar);
        
        if (interval_complete || is_last) {
            // If no data in this bar, skip
            if (i < bar_start_idx) {
                continue;
            }
            
            // Add the bar to the result
            result.add_bar(
                i,                      // index
                timestamps[bar_start_idx], // start_time (index of first data point)
                timestamps[i],          // end_time (index of last data point)
                opens[bar_start_idx],   // open
                *std::max_element(&highs[bar_start_idx], &highs[i] + 1), // high
                *std::min_element(&lows[bar_start_idx], &lows[i] + 1),   // low
                closes[i],              // close
                bar_volume              // volume
            );
            
            // Reset for the next bar
            if (interval_complete) {
                // Start the new bar at the next index
                bar_start_idx = i + 1;
                
                // Reset bar volume
                bar_volume = 0.0;
            }
            
            // If we've reached the end, we're done
            if (is_last) {
                break;
            }
        }
    }
    
    return result;
} 