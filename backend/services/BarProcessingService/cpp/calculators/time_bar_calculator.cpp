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
    
    // Time interval in seconds (params.ratio is in seconds)
    int64_t interval_seconds = static_cast<int64_t>(params.ratio);
    
    // Convert to milliseconds for timestamp comparison
    int64_t interval_ms = interval_seconds * 1000;
    
    // Find the first timestamp
    int64_t first_timestamp = timestamps[0];
    
    // Calculate the start time for the first bar
    // This aligns bars to fixed time boundaries (e.g., 5-minute bars start at :00, :05, :10, etc.)
    int64_t current_bar_start_time = first_timestamp;
    int64_t current_bar_end_time = current_bar_start_time + interval_ms;
    
    size_t bar_start_idx = 0;
    double bar_open = opens[0];
    double bar_high = highs[0];
    double bar_low = lows[0];
    double bar_close = closes[0];
    double bar_volume = 0.0;
    
    // Loop through all data points
    for (size_t i = 0; i < timestamps.size(); ++i) {
        int64_t current_time = timestamps[i];
        
        // If this is the first point in a bar or we're in the same bar
        if (i == bar_start_idx || current_time < current_bar_end_time) {
            // Accumulate data for this bar
            if (i == bar_start_idx) {
                bar_open = opens[i];
                bar_high = highs[i];
                bar_low = lows[i];
            } else {
                bar_high = std::max(bar_high, highs[i]);
                bar_low = std::min(bar_low, lows[i]);
            }
            
            bar_close = closes[i];
            bar_volume += volumes[i];
        } else {
            // We've crossed a bar boundary, finish the current bar
            
            // Add the bar with preserved timeframe information
            add_bar_with_preserved_timeframe(
                result,
                timestamps,
                bar_start_idx,                // timestamp index (use start of bar)
                bar_start_idx,                // start time index
                i - 1,                        // end time index (previous point)
                bar_open,                     // open
                bar_high,                     // high
                bar_low,                      // low
                closes[i - 1],                // close (previous point's close)
                bar_volume                    // volume
            );
            
            // Start a new bar
            bar_start_idx = i;
            current_bar_start_time = current_bar_end_time;
            current_bar_end_time = current_bar_start_time + interval_ms;
            
            // If the current timestamp is already past this new bar boundary,
            // adjust the boundaries to catch up (this handles gaps in data)
            while (current_time >= current_bar_end_time) {
                current_bar_start_time = current_bar_end_time;
                current_bar_end_time = current_bar_start_time + interval_ms;
            }
            
            // Initialize the new bar with the current point
            bar_open = opens[i];
            bar_high = highs[i];
            bar_low = lows[i];
            bar_close = closes[i];
            bar_volume = volumes[i];
        }
    }
    
    // Handle the last bar if it has data
    if (bar_volume > 0 && bar_start_idx < timestamps.size()) {
        size_t last_idx = timestamps.size() - 1;
        
        // Add the final bar with preserved timeframe information
        add_bar_with_preserved_timeframe(
            result,
            timestamps,
            bar_start_idx,            // timestamp index (use start of bar)
            bar_start_idx,            // start time index
            last_idx,                 // end time index
            bar_open,                 // open
            bar_high,                 // high
            bar_low,                  // low
            closes[last_idx],         // close
            bar_volume                // volume
        );
    }
    
    // Verify timeframe preservation
    if (!result.verify_timestamps()) {
        throw std::runtime_error("Timeframe integrity verification failed in time bar calculation");
    }
    
    return result;
} 