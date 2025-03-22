#include "entropy_utils.h"
#include <cmath>
#include <algorithm>

/**
 * Calculate Shannon entropy for a window of price changes.
 * 
 * @param price_changes Array of price changes
 * @param start_idx End index of the window
 * @param window_size Size of the window
 * @return Shannon entropy value
 */
double calculate_shannon_entropy(
    const std::vector<double>& price_changes, 
    size_t start_idx,
    size_t window_size
) {
    if (start_idx < window_size || start_idx >= price_changes.size()) {
        return 0.0;
    }
    
    // Use the last window_size elements
    std::vector<double> window;
    for (size_t i = start_idx - window_size + 1; i <= start_idx; i++) {
        window.push_back(std::abs(price_changes[i]));
    }
    
    // Calculate sum of absolute values
    double sum_abs = 0.0;
    for (double val : window) {
        sum_abs += val;
    }
    
    if (sum_abs == 0.0) {
        return 0.0;
    }
    
    // Calculate probabilities and entropy
    double entropy = 0.0;
    for (double val : window) {
        if (val > 0.0) {
            double prob = val / sum_abs;
            entropy -= prob * std::log2(prob);
        }
    }
    
    return entropy;
}

/**
 * Calculate Tsallis entropy for a window of price changes.
 * 
 * @param price_changes Array of price changes
 * @param start_idx End index of the window
 * @param window_size Size of the window
 * @param q Tsallis q-parameter (default: 1.5)
 * @return Tsallis entropy value
 */
double calculate_tsallis_entropy(
    const std::vector<double>& price_changes, 
    size_t start_idx,
    size_t window_size,
    double q
) {
    if (start_idx < window_size || start_idx >= price_changes.size()) {
        return 0.0;
    }
    
    // Use the last window_size elements
    std::vector<double> window;
    for (size_t i = start_idx - window_size + 1; i <= start_idx; i++) {
        window.push_back(std::abs(price_changes[i]));
    }
    
    // Calculate sum of absolute values
    double sum_abs = 0.0;
    for (double val : window) {
        sum_abs += val;
    }
    
    if (sum_abs == 0.0) {
        return 0.0;
    }
    
    // Calculate probabilities and Tsallis entropy
    double sum_prob_q = 0.0;
    for (double val : window) {
        if (val > 0.0) {
            double prob = val / sum_abs;
            sum_prob_q += std::pow(prob, q);
        }
    }
    
    return (1.0 - sum_prob_q) / (q - 1.0);
}

/**
 * Calculate price changes from an array of prices.
 * 
 * @param prices_ptr Pointer to array of prices
 * @param data_size Size of the array
 * @return Vector of price changes
 */
std::vector<double> calculate_price_changes(
    const double* prices_ptr,
    size_t data_size
) {
    std::vector<double> changes(data_size, 0.0);
    for (size_t i = 1; i < data_size; i++) {
        changes[i] = prices_ptr[i] - prices_ptr[i-1];
    }
    return changes;
}

/**
 * Select the entropy calculation method based on a string.
 * 
 * @param method Method name ("shannon" or "tsallis")
 * @param price_changes Array of price changes
 * @param start_idx End index of the window
 * @param window_size Size of the window
 * @param q Tsallis q-parameter (used only for Tsallis entropy)
 * @return Entropy value
 */
double calculate_entropy(
    const std::string& method,
    const std::vector<double>& price_changes,
    size_t start_idx,
    size_t window_size,
    double q
) {
    if (method == "shannon") {
        return calculate_shannon_entropy(price_changes, start_idx, window_size);
    } else if (method == "tsallis") {
        return calculate_tsallis_entropy(price_changes, start_idx, window_size, q);
    } else {
        // Default to Shannon entropy
        return calculate_shannon_entropy(price_changes, start_idx, window_size);
    }
} 