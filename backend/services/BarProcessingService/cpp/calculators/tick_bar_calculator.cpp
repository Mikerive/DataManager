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
        throw std::runtime_error("Tick threshold ratio must be positive");
    }
    
    // Initialize result
    BarResult result(params.bar_type, params.ratio);
    
    // For tick bars, we use a constant vector of 1's to calculate the average tick rate
    std::vector<double> tick_values(timestamps.size(), 1.0);
    
    // Start the first bar
    size_t bar_start_idx = 0;
    int current_ticks = 0;
    double bar_volume = 0.0;
    double bar_high = highs[0];
    double bar_low = lows[0];
    
    for (size_t i = 0; i < timestamps.size(); ++i) {
        // Update high/low for the current bar
        if (i == bar_start_idx) {
            bar_high = highs[i];
            bar_low = lows[i];
        } else {
            bar_high = std::max(bar_high, highs[i]);
            bar_low = std::min(bar_low, lows[i]);
        }
        
        // Calculate adaptive threshold based on lookback window
        // For tick bars, we use a constant vector of 1's to calculate the average tick rate
        double tick_threshold = calculate_adaptive_threshold(
            tick_values,
            i > params.lookback_window ? i - 1 : 0,
            params.lookback_window,
            params.ratio
        );
        
        // Round to nearest integer and enforce minimum of 1
        int tick_count = static_cast<int>(std::round(tick_threshold));
        if (tick_count < 1) tick_count = 1;
        
        // Increment tick counter and accumulate volume
        current_ticks++;
        bar_volume += volumes[i];
        
        // Check if we've reached the tick count or end of data
        if (current_ticks >= tick_count || i == timestamps.size() - 1) {
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
    
    // Verify timeframe preservation
    if (!result.verify_timestamps()) {
        throw std::runtime_error("Timeframe integrity verification failed in tick bar calculation");
    }
    
    return result;
}

void TickBarCalculator::calculate_bars_with_adaptive_threshold(
    const std::vector<int64_t>& timestamps,
    const std::vector<double>& opens,
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    const std::vector<double>& tick_directions,
    double threshold_ratio,
    size_t lookback_window,
    BarResult& result
) {
    if (closes.empty()) {
        return;
    }

    // Store tick counts for adaptive calculation
    std::vector<double> bar_ticks;
    
    // We'll track tick count between bars (each record is one tick)
    int tick_count = 0;
    size_t bar_start_idx = 0;
    double bar_volume = 0;
    bool first_bar = true;
    
    for (size_t i = 0; i < closes.size(); ++i) {
        // Each data point is one tick
        tick_count++;
        bar_volume += volumes[i];
        
        // Skip threshold check during initialization period for first bar
        if (first_bar && i < lookback_window) {
            // For first bar, just accumulate ticks during initialization
            if (i < closes.size() - 1) {
                continue;
            }
        }
        
        // Calculate adaptive threshold for this position
        double adaptive_threshold = 0;
        
        // Calculate threshold if we have enough bars or we're at the end
        if (!bar_ticks.empty() || i == closes.size() - 1) {
            adaptive_threshold = calculate_adaptive_tick_threshold(
                bar_ticks, bar_ticks.size(), lookback_window, threshold_ratio
            );
        } else {
            adaptive_threshold = threshold_ratio; // Initial guess for first bar
        }
        
        // Check if we've reached the threshold or end of data
        bool create_bar = (adaptive_threshold > 0 && tick_count >= adaptive_threshold) || 
                           (i == closes.size() - 1);
        
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
                bar_volume          // volume
            );
            
            // Store the tick count of this bar for future adaptive calculations
            bar_ticks.push_back(static_cast<double>(tick_count));
            
            // Reset for next bar
            tick_count = 0;
            bar_volume = 0;
            bar_start_idx = i + 1;
            first_bar = false;
        }
    }
}

BarResult TickBarCalculator::calculate_bars(
    const std::vector<int64_t>& timestamps,
    const std::vector<double>& opens,
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    const std::vector<double>& tick_directions,
    const BarParams& params
) {
    BarResult result(BarType::Tick, params.ratio);
    
    // Check if we should use adaptive thresholds
    if (params.lookback_window > 0) {
        // Use adaptive threshold calculation
        calculate_bars_with_adaptive_threshold(
            timestamps, opens, highs, lows, closes, volumes, tick_directions,
            params.ratio, params.lookback_window, result
        );
    } else {
        // Use fixed threshold calculation
        int tick_threshold = static_cast<int>(params.ratio);
        int tick_count = 0;
        int64_t bar_start_idx = 0;
        double bar_volume = 0;
        
        for (size_t i = 0; i < closes.size(); i++) {
            tick_count++;
            bar_volume += volumes[i];
            
            if (tick_count >= tick_threshold || i == closes.size() - 1) {
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
                    bar_volume          // volume
                );
                
                // Reset for next bar
                tick_count = 0;
                bar_volume = 0;
                bar_start_idx = i + 1;
            }
        }
    }
    
    return result;
} 