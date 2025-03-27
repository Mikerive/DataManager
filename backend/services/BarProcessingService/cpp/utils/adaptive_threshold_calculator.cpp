#include "adaptive_threshold_calculator.h"
#include <sstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

// Initialize global instance
AdaptiveThresholdCalculator& AdaptiveThresholdCalculator::getInstance() {
    static AdaptiveThresholdCalculator instance;
    return instance;
}

// Calculate adaptive volume threshold with proper initialization
double AdaptiveThresholdCalculator::calculate_volume_threshold(
    const std::vector<double>& values, 
    size_t current_idx, 
    size_t lookback_window, 
    double ratio,
    const std::string& ticker_id) {
    
    // Ensure we have enough data for adaptive calculation
    if (current_idx < lookback_window) {
        // During initialization period, use a simple average of available data
        if (current_idx == 0) {
            return values[0] * ratio; // Use first value with ratio for first bar
        }
        
        // For initial bars, use average of available values
        double sum = 0.0;
        for (size_t i = 0; i < current_idx; ++i) {
            sum += values[i];
        }
        return (sum / current_idx) * ratio;
    }
    
    // Use the internal calculation with caching
    return calculate_adaptive_threshold_internal(
        "volume", values, current_idx, lookback_window, ratio, ticker_id);
}

// Calculate adaptive tick threshold with proper initialization
double AdaptiveThresholdCalculator::calculate_tick_threshold(
    const std::vector<double>& values, 
    size_t current_idx, 
    size_t lookback_window, 
    double ratio,
    const std::string& ticker_id) {
    
    // Ensure we have enough data for adaptive calculation
    if (current_idx < lookback_window) {
        // During initialization period, use a simple average of available data
        if (current_idx == 0) {
            return values[0] * ratio; // Use first value with ratio for first bar
        }
        
        // For initial bars, use average of available values
        double sum = 0.0;
        for (size_t i = 0; i < current_idx; ++i) {
            sum += values[i];
        }
        return (sum / current_idx) * ratio;
    }
    
    // Use the internal calculation with caching
    return calculate_adaptive_threshold_internal(
        "tick", values, current_idx, lookback_window, ratio, ticker_id);
}

// Calculate adaptive entropy threshold with proper initialization
double AdaptiveThresholdCalculator::calculate_entropy_threshold(
    const std::vector<double>& values, 
    size_t current_idx, 
    size_t lookback_window, 
    double ratio,
    const std::string& ticker_id) {
    
    // Ensure we have enough data for adaptive calculation
    if (current_idx < lookback_window) {
        // For entropy bars, require more initial data before starting
        // Return a negative value to indicate we should wait for more data
        return -1.0;
    }
    
    // Use the internal calculation with caching
    return calculate_adaptive_threshold_internal(
        "entropy", values, current_idx, lookback_window, ratio, ticker_id);
}

// Clear the entire cache
void AdaptiveThresholdCalculator::clear_cache() {
    threshold_cache_.clear();
}

// Clear cache for a specific ticker
void AdaptiveThresholdCalculator::clear_cache_for_ticker(const std::string& ticker_id) {
    if (ticker_id.empty()) return;
    
    // Create an iterator to traverse the map and erase entries for the specified ticker
    auto it = threshold_cache_.begin();
    while (it != threshold_cache_.end()) {
        if (it->first.find(ticker_id) == 0) {
            // Key starts with ticker_id, so remove it
            it = threshold_cache_.erase(it);
        } else {
            ++it;
        }
    }
}

// Internal calculation with caching
double AdaptiveThresholdCalculator::calculate_adaptive_threshold_internal(
    const std::string& bar_type,
    const std::vector<double>& values, 
    size_t current_idx, 
    size_t lookback_window, 
    double ratio,
    const std::string& ticker_id) {
    
    // Check if we can use a cached value
    if (!ticker_id.empty()) {
        std::string cache_key = create_cache_key(ticker_id, bar_type, current_idx, lookback_window, ratio);
        auto cache_it = threshold_cache_.find(cache_key);
        
        if (cache_it != threshold_cache_.end()) {
            return cache_it->second;
        }
    }
    
    // Calculate the average over the lookback window
    double sum = 0.0;
    size_t start_idx = current_idx >= lookback_window ? current_idx - lookback_window : 0;
    size_t count = current_idx - start_idx;
    
    if (count == 0) {
        throw std::runtime_error("Invalid lookback window or current index.");
    }
    
    for (size_t i = start_idx; i < current_idx; ++i) {
        sum += values[i];
    }
    
    double avg = sum / count;
    double threshold = avg * ratio;
    
    // Cache the result if a ticker ID was provided
    if (!ticker_id.empty()) {
        std::string cache_key = create_cache_key(ticker_id, bar_type, current_idx, lookback_window, ratio);
        threshold_cache_[cache_key] = threshold;
    }
    
    return threshold;
}

// Create a unique key for caching
std::string AdaptiveThresholdCalculator::create_cache_key(
    const std::string& ticker_id,
    const std::string& bar_type,
    size_t current_idx,
    size_t lookback_window,
    double ratio) {
    
    std::stringstream ss;
    ss << ticker_id << "_" << bar_type << "_" << current_idx << "_" << lookback_window << "_";
    ss << std::fixed << std::setprecision(4) << ratio;
    return ss.str();
} 