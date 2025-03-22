#include "tick_bar_calculator.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

BarResult TickBarCalculator::calculate(
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
        throw std::runtime_error("Invalid data for tick bar calculation");
    }
    
    if (params.ratio <= 0) {
        throw std::runtime_error("Tick count must be positive");
    }
    
    // Initialize result
    BarResult result(params.bar_type, params.ratio);
    
    // Tick count from parameters (round to nearest integer)
    int tick_count = static_cast<int>(std::round(params.ratio));
    if (tick_count < 1) tick_count = 1;
    
    // Start the first bar
    size_t bar_start_idx = 0;
    int current_ticks = 0;
    double bar_volume = 0.0;
    
    for (size_t i = 0; i < timestamps.size(); ++i) {
        // Increment tick counter and accumulate volume
        current_ticks++;
        bar_volume += volumes[i];
        
        // Check if we've reached the tick count or end of data
        if (current_ticks >= tick_count || i == timestamps.size() - 1) {
            // Add the bar to the result
            result.add_bar(
                i,                      // index
                timestamps[bar_start_idx],  // start_time
                timestamps[i],          // end_time
                opens[bar_start_idx],          // open
                *std::max_element(&highs[bar_start_idx], &highs[i] + 1), // high
                *std::min_element(&lows[bar_start_idx], &lows[i] + 1),   // low
                closes[i],              // close
                bar_volume              // volume
            );
            
            // Reset for the next bar
            current_ticks = 0;
            bar_volume = 0.0;
            bar_start_idx = i + 1;
            
            // If we've reached the end, we're done
            if (i == timestamps.size() - 1) {
                break;
            }
        }
    }
    
    return result;
} 