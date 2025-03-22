#pragma once

#include <vector>
#include <string>

/**
 * Calculate Shannon entropy for a window of price changes.
 * 
 * Shannon entropy is a measure of information content or
 * uncertainty in a probability distribution.
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
);

/**
 * Calculate Tsallis entropy for a window of price changes.
 * 
 * Tsallis entropy is a generalization of Shannon entropy
 * that introduces a parameter q, which allows for non-extensive
 * statistics.
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
    double q = 1.5
);

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
);

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
    double q = 1.5
); 