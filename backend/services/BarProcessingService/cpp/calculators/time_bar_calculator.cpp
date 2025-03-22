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
    
    // Time interval in milliseconds (assuming timestamps are in milliseconds)
    int64_t time_interval_ms = static_cast<int64_t>(params.ratio * 1000);
    
    // Start time of the first bar
    int64_t bar_start_time = timestamps[0];
    size_t bar_start_idx = 0;
    double bar_volume = 0.0;
    
    for (size_t i = 0; i < timestamps.size(); ++i) {
        // Accumulate volume
        bar_volume += volumes[i];
        
        // Check if we've reached the time interval or end of data
        bool is_last = (i == timestamps.size() - 1);
        bool interval_complete = (timestamps[i] - bar_start_time >= time_interval_ms);
        
        if (interval_complete || is_last) {
            // If no data in this bar, skip
            if (i < bar_start_idx) {
                continue;
            }
            
            // Add the bar to the result
            result.add_bar(
                i,                      // index
                bar_start_time,         // start_time
                timestamps[i],          // end_time
                opens[bar_start_idx],   // open
                *std::max_element(&highs[bar_start_idx], &highs[i] + 1), // high
                *std::min_element(&lows[bar_start_idx], &lows[i] + 1),   // low
                closes[i],              // close
                bar_volume              // volume
            );
            
            // Reset for the next bar
            if (interval_complete) {
                // Start the new bar at the next interval boundary
                bar_start_time += time_interval_ms;
                
                // If current time is already past the new start time, adjust it
                // (this can happen with irregular data timestamps)
                while (bar_start_time < timestamps[i]) {
                    bar_start_time += time_interval_ms;
                }
                
                // Reset bar data
                bar_start_idx = i + 1;
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