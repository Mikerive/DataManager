#include "entropy_bar_calculator.h"
#include "../utils/entropy_utils.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

BarResult EntropyBarCalculator::calculate(
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
        throw std::runtime_error("Invalid data for entropy bar calculation");
    }
    
    if (params.ratio <= 0) {
        throw std::runtime_error("Entropy threshold must be positive");
    }
    
    // Initialize result
    BarResult result(params.bar_type, params.ratio);
    
    // Get the necessary parameters
    double entropy_threshold = params.ratio;
    size_t window_size = params.window_size;
    std::string method = params.method;
    double q_param = params.q_param;
    
    // Need enough data for the window
    if (closes.size() < window_size) {
        return result; // Return empty result
    }
    
    // Calculate price changes
    std::vector<double> price_changes = calculate_price_changes(&closes[0], closes.size());
    
    // Start the first bar
    size_t bar_start_idx = window_size; // Start after we have enough data for first entropy calculation
    double current_entropy = 0.0;
    double bar_volume = 0.0;
    
    for (size_t i = window_size; i < closes.size(); ++i) {
        // Calculate entropy at this point
        current_entropy = calculate_entropy(
            method, price_changes, i, window_size, q_param
        );
        
        // Accumulate volume
        bar_volume += volumes[i];
        
        // Check if we've reached the entropy threshold or end of data
        if (current_entropy >= entropy_threshold || i == closes.size() - 1) {
            // Add the bar to the result
            result.add_bar(
                i,                      // index
                timestamps[bar_start_idx],  // start_time
                timestamps[i],          // end_time
                opens[bar_start_idx],   // open
                *std::max_element(&highs[bar_start_idx], &highs[i] + 1), // high
                *std::min_element(&lows[bar_start_idx], &lows[i] + 1),   // low
                closes[i],              // close
                bar_volume              // volume
            );
            
            // Reset for the next bar
            bar_start_idx = i + 1;
            bar_volume = 0.0;
            
            // If we've reached the end, we're done
            if (i == closes.size() - 1) {
                break;
            }
        }
    }
    
    return result;
}

std::vector<double> EntropyBarCalculator::calculate_rolling_entropy(
    const std::vector<double>& prices,
    size_t window_size,
    const std::string& method,
    double q_param
) {
    // Calculate price changes first
    std::vector<double> price_changes = calculate_price_changes(&prices[0], prices.size());
    
    // Calculate entropy for each point after the initial window
    std::vector<double> entropy_values(prices.size(), 0.0);
    
    for (size_t i = window_size; i < prices.size(); ++i) {
        entropy_values[i] = calculate_entropy(
            method, price_changes, i, window_size, q_param
        );
    }
    
    return entropy_values;
} 