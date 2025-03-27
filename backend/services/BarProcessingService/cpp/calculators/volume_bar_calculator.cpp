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
        throw std::runtime_error("Volume threshold ratio must be positive");
    }
    
    // Initialize result
    BarResult result(params.bar_type, params.ratio);
    
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
        
        // Calculate adaptive threshold based on lookback window
        double volume_threshold = calculate_adaptive_threshold(
            volumes,
            i > params.lookback_window ? i - 1 : 0,  // Look back from previous point if possible
            params.lookback_window,
            params.ratio
        );
        
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

void VolumeBarCalculator::calculate_bars_with_adaptive_threshold(
    const std::vector<int64_t>& timestamps,
    const std::vector<double>& opens,
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    double threshold_ratio,
    size_t lookback_window,
    BarResult& result
) {
    if (volumes.empty()) {
        return;
    }

    // Store volumes for adaptive calculation
    std::vector<double> bar_volumes;
    
    // We'll track cumulative volume between bars
    double cumulative_volume = 0;
    size_t bar_start_idx = 0;
    bool first_bar = true;
    
    for (size_t i = 0; i < volumes.size(); ++i) {
        // Add to running volume
        cumulative_volume += volumes[i];
        
        // Calculate adaptive threshold for this position
        double adaptive_threshold = 0;
        
        // Skip threshold check during initialization period for first bar
        if (first_bar && i < lookback_window) {
            // For first bar, just accumulate volume during initialization
            if (i < volumes.size() - 1) {
                continue;
            }
        }
        
        // Calculate threshold if we have enough bars or we're at the end
        if (!bar_volumes.empty() || i == volumes.size() - 1) {
            adaptive_threshold = calculate_adaptive_volume_threshold(
                bar_volumes, bar_volumes.size(), lookback_window, threshold_ratio
            );
        } else {
            adaptive_threshold = volumes[i] * threshold_ratio; // Initial guess
        }
        
        // Check if we've reached the threshold or end of data
        bool create_bar = (adaptive_threshold > 0 && cumulative_volume >= adaptive_threshold) || 
                           (i == volumes.size() - 1);
        
        if (create_bar) {
            // Calculate price info for this bar
            double bar_open = opens[bar_start_idx];
            double bar_close = closes[i];
            
            double bar_high = highs[bar_start_idx];
            double bar_low = lows[bar_start_idx];
            
            for (size_t j = bar_start_idx + 1; j <= i; ++j) {
                bar_high = std::max(bar_high, highs[j]);
                bar_low = std::min(bar_low, lows[j]);
            }
            
            // Add bar to result
            result.add_bar(
                i,                  // timestamp index 
                bar_start_idx,      // start time index
                i,                  // end time index
                bar_open,           // open price
                bar_high,           // high price
                bar_low,            // low price
                bar_close,          // close price
                cumulative_volume   // volume
            );
            
            // Store the volume of this bar for future adaptive calculations
            bar_volumes.push_back(cumulative_volume);
            
            // Reset for next bar
            cumulative_volume = 0;
            bar_start_idx = i + 1;
            first_bar = false;
        }
    }
}

BarResult VolumeBarCalculator::calculate_bars(
    const std::vector<int64_t>& timestamps,
    const std::vector<double>& opens,
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    const std::vector<double>& tick_directions,
    const BarParams& params
) {
    BarResult result(BarType::Volume, params.ratio);
    
    // Check if we should use adaptive thresholds
    if (params.lookback_window > 0) {
        // Use adaptive threshold calculation
        calculate_bars_with_adaptive_threshold(
            timestamps, opens, highs, lows, closes, volumes,
            params.ratio, params.lookback_window, result
        );
    } else {
        // Use fixed threshold calculation
        double volume_threshold = params.ratio;
        double running_volume = 0;
        int64_t bar_start_idx = 0;
        
        for (size_t i = 0; i < volumes.size(); i++) {
            running_volume += volumes[i];
            
            if (running_volume >= volume_threshold || i == volumes.size() - 1) {
                // We've hit the threshold or the end of the data, create a bar
                
                // Find high and low in the bar
                double bar_high = *std::max_element(highs.begin() + bar_start_idx, highs.begin() + i + 1);
                double bar_low = *std::min_element(lows.begin() + bar_start_idx, lows.begin() + i + 1);
                
                // Add bar to result
                result.add_bar(
                    i,                  // timestamp index
                    bar_start_idx,      // start time index
                    i,                  // end time index
                    opens[bar_start_idx], // open price
                    bar_high,           // high price
                    bar_low,            // low price
                    closes[i],          // close price
                    running_volume      // volume
                );
                
                // Reset for next bar
                running_volume = 0;
                bar_start_idx = i + 1;
            }
        }
    }
    
    return result;
} 