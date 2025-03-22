#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "bar_params.h"
#include "bar_result.h"

// Forward declarations of calculator classes
class VolumeBarCalculator;
class TickBarCalculator;
class TimeBarCalculator;
class EntropyBarCalculator;

/**
 * Main class for calculating different types of bars from market data.
 * This class manages multiple bar calculation types and provides batch processing.
 */
class BarCalculator {
public:
    /**
     * Constructor
     */
    BarCalculator();
    
    /**
     * Destructor
     */
    ~BarCalculator();

    /**
     * Set the price data for calculations.
     * 
     * @param timestamps_ptr Pointer to array of timestamps
     * @param opens_ptr Pointer to array of opening prices
     * @param highs_ptr Pointer to array of high prices
     * @param lows_ptr Pointer to array of low prices
     * @param closes_ptr Pointer to array of closing prices
     * @param volumes_ptr Pointer to array of volumes
     * @param data_size Size of the data arrays
     */
    void set_data(
        const int64_t* timestamps_ptr,
        const double* opens_ptr,
        const double* highs_ptr,
        const double* lows_ptr,
        const double* closes_ptr,
        const double* volumes_ptr,
        size_t data_size
    );

    /**
     * Calculate bars for a specific bar type.
     * 
     * @param params Parameters for bar calculation
     * @return Result of the bar calculation
     */
    BarResult calculate_bars(const BarParams& params);

    /**
     * Process multiple bar types in a single pass.
     * 
     * @param params_list Vector of parameter sets for different bar types
     * @return Map of bar type to bar calculation results
     */
    std::unordered_map<std::string, BarResult> batch_process(
        const std::vector<BarParams>& params_list
    );

private:
    // Data members
    std::vector<int64_t> timestamps_;
    std::vector<double> opens_;
    std::vector<double> highs_;
    std::vector<double> lows_;
    std::vector<double> closes_;
    std::vector<double> volumes_;
    size_t data_size_;

    // Specialized calculators
    std::unique_ptr<VolumeBarCalculator> volume_calculator_;
    std::unique_ptr<TickBarCalculator> tick_calculator_;
    std::unique_ptr<TimeBarCalculator> time_calculator_;
    std::unique_ptr<EntropyBarCalculator> entropy_calculator_;
    
    // Helper method to initialize calculators
    void initialize_calculators();
}; 