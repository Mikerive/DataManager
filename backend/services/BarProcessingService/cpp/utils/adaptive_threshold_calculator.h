#ifndef ADAPTIVE_THRESHOLD_CALCULATOR_H
#define ADAPTIVE_THRESHOLD_CALCULATOR_H

#include <vector>
#include <string>
#include <unordered_map>

/**
 * Class responsible for calculating adaptive thresholds with caching capability.
 * This is designed as a singleton to ensure consistent caching across the application.
 */
class AdaptiveThresholdCalculator {
public:
    /**
     * Get the singleton instance
     */
    static AdaptiveThresholdCalculator& getInstance();

    /**
     * Calculate adaptive threshold for volume bars
     * 
     * @param values The volume values
     * @param current_idx Current position in the data
     * @param lookback_window Window size for lookback 
     * @param ratio Multiplier for the threshold
     * @param ticker_id Optional ticker ID for caching
     * @return The calculated threshold
     */
    double calculate_volume_threshold(
        const std::vector<double>& values, 
        size_t current_idx, 
        size_t lookback_window, 
        double ratio,
        const std::string& ticker_id = ""
    );

    /**
     * Calculate adaptive threshold for tick bars
     * 
     * @param values The tick values
     * @param current_idx Current position in the data
     * @param lookback_window Window size for lookback 
     * @param ratio Multiplier for the threshold
     * @param ticker_id Optional ticker ID for caching
     * @return The calculated threshold
     */
    double calculate_tick_threshold(
        const std::vector<double>& values, 
        size_t current_idx, 
        size_t lookback_window, 
        double ratio,
        const std::string& ticker_id = ""
    );

    /**
     * Calculate adaptive threshold for entropy bars
     * For entropy bars, returns -1.0 during the initialization period
     * to indicate that more data is needed
     * 
     * @param values The entropy values
     * @param current_idx Current position in the data
     * @param lookback_window Window size for lookback 
     * @param ratio Multiplier for the threshold
     * @param ticker_id Optional ticker ID for caching
     * @return The calculated threshold or -1.0 if in initialization
     */
    double calculate_entropy_threshold(
        const std::vector<double>& values, 
        size_t current_idx, 
        size_t lookback_window, 
        double ratio,
        const std::string& ticker_id = ""
    );

    /**
     * Clear the entire cache
     */
    void clear_cache();

    /**
     * Clear cached values for a specific ticker
     */
    void clear_cache_for_ticker(const std::string& ticker_id);

private:
    /**
     * Private constructor for singleton pattern
     */
    AdaptiveThresholdCalculator() = default;
    
    /**
     * Private copy constructor and assignment operator to prevent copying
     */
    AdaptiveThresholdCalculator(const AdaptiveThresholdCalculator&) = delete;
    AdaptiveThresholdCalculator& operator=(const AdaptiveThresholdCalculator&) = delete;

    /**
     * Internal calculation with caching support
     */
    double calculate_adaptive_threshold_internal(
        const std::string& bar_type,
        const std::vector<double>& values, 
        size_t current_idx, 
        size_t lookback_window, 
        double ratio,
        const std::string& ticker_id
    );

    /**
     * Create a unique cache key for storing threshold values
     */
    std::string create_cache_key(
        const std::string& ticker_id,
        const std::string& bar_type,
        size_t current_idx,
        size_t lookback_window,
        double ratio
    );

    /**
     * Cache for storing calculated thresholds
     */
    std::unordered_map<std::string, double> threshold_cache_;
};

#endif // ADAPTIVE_THRESHOLD_CALCULATOR_H 