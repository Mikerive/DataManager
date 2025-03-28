#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include "bar_params.h"

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
    
    /** Actual timestamp values for each bar */
    std::vector<int64_t> timestamps;
    
    /** Actual start timestamp values for each bar */
    std::vector<int64_t> start_timestamps;
    
    /** Actual end timestamp values for each bar */
    std::vector<int64_t> end_timestamps;
    
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
    
    /** Type of bar (BarType enum) */
    BarType bar_type;
    
    /** Bar type as string (for backward compatibility) */
    std::string bar_type_string;
    
    /** The parameter value used for the calculation */
    double ratio;
    
    /**
     * Default constructor.
     */
    BarResult() : bar_type(BarType::Time), bar_type_string("time"), ratio(0.0) {}
    
    /**
     * Constructor with string bar type.
     * 
     * @param type Bar type string
     * @param r Parameter ratio/threshold value
     */
    BarResult(const std::string& type, double r);
    
    /**
     * Constructor with enum bar type.
     * 
     * @param type Bar type enum
     * @param r Parameter ratio/threshold value
     */
    BarResult(BarType type, double r);
    
    /**
     * Convert the bar results to a Python dictionary.
     * 
     * @param timestamps Python array containing the original timestamps
     * @return Python dictionary with bar data
     */
    py::dict to_dict(py::array timestamps_arr) const;
    
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
     * Add a new bar to the results with explicit timestamp values.
     * 
     * @param ts_idx Index of the timestamp in the original array
     * @param start_idx Index of the start time in the original array
     * @param end_idx Index of the end time in the original array
     * @param timestamp Actual timestamp value
     * @param start_time Actual start time value
     * @param end_time Actual end time value
     * @param open Opening price
     * @param high Highest price
     * @param low Lowest price
     * @param close Closing price
     * @param volume Volume
     */
    void add_bar_with_timestamps(
        size_t ts_idx, size_t start_idx, size_t end_idx,
        int64_t timestamp, int64_t start_time, int64_t end_time,
        double open, double high, double low, double close, double volume);
    
    /**
     * Verify timestamp continuity to ensure timeframe preservation.
     * 
     * @return True if timestamps are continuous and valid, false otherwise
     */
    bool verify_timestamps() const;
    
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