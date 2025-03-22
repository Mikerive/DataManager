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
    
    for (size_t i = 0; i < volumes.size(); ++i) {
        // Add current volume to the accumulator
        current_bar_volume += volumes[i];
        
        // Check if we've reached the threshold or end of data
        if (current_bar_volume >= volume_threshold || i == volumes.size() - 1) {
            // Add the bar to the result
            result.add_bar(
                i,                  // index
                timestamps[bar_start_idx],  // start_time
                timestamps[i],      // end_time
                opens[bar_start_idx],      // open
                *std::max_element(&highs[bar_start_idx], &highs[i] + 1), // high
                *std::min_element(&lows[bar_start_idx], &lows[i] + 1),   // low
                closes[i],          // close
                current_bar_volume  // volume
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
    
    return result;
} 