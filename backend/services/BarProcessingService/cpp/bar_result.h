#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>

namespace py = pybind11;

/**
 * Class to hold the result of a bar calculation.
 * 
 * This class stores the raw data for bars (OHLCV, timestamps) and
 * provides methods for adding bars, checking the result status,
 * and converting the results to Python objects.
 */
class BarResult {
public:
    /** Indices of the timestamp values in the original data array */
    std::vector<size_t> timestamp_indices;
    
    /** Indices of the start time values in the original data array */
    std::vector<size_t> start_time_indices;
    
    /** Indices of the end time values in the original data array */
    std::vector<size_t> end_time_indices;
    
    /** Opening prices for each bar */
    std::vector<double> opens;
    
    /** Highest prices for each bar */
    std::vector<double> highs;
    
    /** Lowest prices for each bar */
    std::vector<double> lows;
    
    /** Closing prices for each bar */
    std::vector<double> closes;
    
    /** Volumes for each bar */
    std::vector<double> volumes;
    
    /** Type of bar ('volume', 'tick', 'time', 'entropy', etc.) */
    std::string bar_type;
    
    /** The parameter value used for the calculation */
    double ratio;
    
    /**
     * Constructor.
     * 
     * @param type Bar type string
     * @param r Parameter ratio/threshold value
     */
    BarResult(const std::string& type, double r);
    
    /**
     * Convert the bar results to a Python dictionary.
     * 
     * @param timestamps Python array containing the original timestamps
     * @return Python dictionary with bar data
     */
    py::dict to_dict(py::array timestamps) const;
    
    /**
     * Add a new bar to the results.
     * 
     * @param ts_idx Index of the timestamp in the original array
     * @param start_idx Index of the start time in the original array
     * @param end_idx Index of the end time in the original array
     * @param open Opening price
     * @param high Highest price
     * @param low Lowest price
     * @param close Closing price
     * @param volume Volume
     */
    void add_bar(size_t ts_idx, size_t start_idx, size_t end_idx, 
                double open, double high, double low, double close, double volume);
    
    /**
     * Check if the result is empty (no bars).
     * 
     * @return True if no bars, false otherwise
     */
    bool empty() const;
    
    /**
     * Get the number of bars in the result.
     * 
     * @return Number of bars
     */
    size_t size() const;
}; 