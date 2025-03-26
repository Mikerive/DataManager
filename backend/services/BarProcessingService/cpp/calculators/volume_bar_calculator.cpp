#include "volume_bar_calculator.h"
#include <algorithm>
#include <stdexcept>

BarResult VolumeBarCalculator::calculate(
    const std::vector<int64_t>& timestamps,
    const std::vector<double>& opens,
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    const BarParams& params
) {
    // Validate parameters
    if (timestamps.empty() || timestamps.size() != volumes.size()) {
        throw std::runtime_error("Invalid data for volume bar calculation");
    }
    
    if (params.ratio <= 0) {
        throw std::runtime_error("Volume threshold must be positive");
    }
    
    // Initialize result
    BarResult result(params.bar_type, params.ratio);
    
    // Volume threshold from parameters
    double volume_threshold = params.ratio;
    
    // Start the first bar
    double current_bar_volume = 0.0;
    size_t bar_start_idx = 0;
    double bar_high = highs[0];
    double bar_low = lows[0];
    
    for (size_t i = 0; i < volumes.size(); ++i) {
        // Update high/low for the current bar
        if (i == bar_start_idx) {
            bar_high = highs[i];
            bar_low = lows[i];
        } else {
            bar_high = std::max(bar_high, highs[i]);
            bar_low = std::min(bar_low, lows[i]);
        }
        
        // Add current volume to the accumulator
        current_bar_volume += volumes[i];
        
        // Check if we've reached the threshold or end of data
        if (current_bar_volume >= volume_threshold || i == volumes.size() - 1) {
            // Add the bar with preserved timeframe information
            add_bar_with_preserved_timeframe(
                result,
                timestamps,
                i,                      // timestamp index (use last point in bar)
                bar_start_idx,          // start_time index
                i,                      // end_time index
                opens[bar_start_idx],   // open
                bar_high,               // high
                bar_low,                // low
                closes[i],              // close
                current_bar_volume      // volume
            );
            
            // Reset for the next bar
            current_bar_volume = 0.0;
            bar_start_idx = i + 1;
            
            // If we've reached the end, we're done
            if (i == volumes.size() - 1) {
                break;
            }
        }
    }
    
    // Verify timeframe preservation
    if (!result.verify_timestamps()) {
        throw std::runtime_error("Timeframe integrity verification failed in volume bar calculation");
    }
    
    return result;
} 