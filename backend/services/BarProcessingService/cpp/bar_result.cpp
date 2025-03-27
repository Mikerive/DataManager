#include "bar_result.h"
#include <pybind11/stl.h>  // Add this include for automatic STL conversions
#include <pybind11/numpy.h>
#include <algorithm>
#include <stdexcept>

/**
 * Constructor with string bar type.
 */
BarResult::BarResult(const std::string& type, double r) : ratio(r) {
    // Convert string to enum
    if (type == "volume") {
        bar_type = BarType::Volume;
    } else if (type == "tick") {
        bar_type = BarType::Tick;
    } else if (type == "time") {
        bar_type = BarType::Time;
    } else if (type == "entropy") {
        bar_type = BarType::Entropy;
    } else if (type == "dollar") {
        bar_type = BarType::Dollar;
    } else if (type == "information") {
        bar_type = BarType::Information;
    } else {
        // Default to time
        bar_type = BarType::Time;
    }
    
    bar_type_string = type;
}

/**
 * Constructor with enum bar type.
 */
BarResult::BarResult(BarType type, double r) : bar_type(type), ratio(r) {
    // Convert enum to string for backwards compatibility
    bar_type_string = to_string(type);
}

/**
 * Convert the bar results to a Python dictionary.
 * 
 * @param timestamps Python array containing the original timestamps
 * @return Python dictionary with bar data
 */
py::dict BarResult::to_dict(py::array timestamps_arr) const {
    py::dict result;
    
    // Convert data to NumPy arrays
    size_t bar_count = size();
    
    py::array_t<int64_t> ts_array(bar_count);
    py::array_t<double> opens_array(bar_count);
    py::array_t<double> highs_array(bar_count);
    py::array_t<double> lows_array(bar_count);
    py::array_t<double> closes_array(bar_count);
    py::array_t<double> volumes_array(bar_count);
    py::array_t<int64_t> start_time_array(bar_count);
    py::array_t<int64_t> end_time_array(bar_count);
    
    // Get raw pointers from the arrays
    auto timestamps_ptr = static_cast<int64_t*>(timestamps_arr.request().ptr);
    auto ts_ptr = static_cast<int64_t*>(ts_array.request().ptr);
    auto opens_ptr = static_cast<double*>(opens_array.request().ptr);
    auto highs_ptr = static_cast<double*>(highs_array.request().ptr);
    auto lows_ptr = static_cast<double*>(lows_array.request().ptr);
    auto closes_ptr = static_cast<double*>(closes_array.request().ptr);
    auto volumes_ptr = static_cast<double*>(volumes_array.request().ptr);
    auto start_time_ptr = static_cast<int64_t*>(start_time_array.request().ptr);
    auto end_time_ptr = static_cast<int64_t*>(end_time_array.request().ptr);
    
    // Fill the arrays
    for (size_t i = 0; i < size(); ++i) {
        // Use stored actual timestamps if available, otherwise use indices
        if (!this->timestamps.empty()) {
            ts_ptr[i] = this->timestamps.size() > 0 ? this->timestamps[i] : timestamps_ptr[timestamp_indices[i]];
            start_time_ptr[i] = this->start_timestamps.size() > 0 ? this->start_timestamps[i] : timestamps_ptr[start_time_indices[i]];
            end_time_ptr[i] = this->end_timestamps.size() > 0 ? this->end_timestamps[i] : timestamps_ptr[end_time_indices[i]];
        } else {
            ts_ptr[i] = timestamps_ptr[timestamp_indices[i]];
            start_time_ptr[i] = timestamps_ptr[start_time_indices[i]];
            end_time_ptr[i] = timestamps_ptr[end_time_indices[i]];
        }
        
        opens_ptr[i] = opens[i];
        highs_ptr[i] = highs[i];
        lows_ptr[i] = lows[i];
        closes_ptr[i] = closes[i];
        volumes_ptr[i] = volumes[i];
    }
    
    // Add the arrays to the result dictionary
    result["timestamps"] = ts_array;
    result["opens"] = opens_array;
    result["highs"] = highs_array;
    result["lows"] = lows_array;
    result["closes"] = closes_array;
    result["volumes"] = volumes_array;
    result["start_times"] = start_time_array;
    result["end_times"] = end_time_array;
    result["bar_type"] = bar_type_string;
    result["ratio"] = ratio;
    
    return result;
}

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
void BarResult::add_bar(size_t ts_idx, size_t start_idx, size_t end_idx, 
                     double open, double high, double low, double close, double volume) {
    timestamp_indices.push_back(ts_idx);
    start_time_indices.push_back(start_idx);
    end_time_indices.push_back(end_idx);
    opens.push_back(open);
    highs.push_back(high);
    lows.push_back(low);
    closes.push_back(close);
    volumes.push_back(volume);
}

void BarResult::add_bar_with_timestamps(
    size_t ts_idx, size_t start_idx, size_t end_idx,
    int64_t timestamp, int64_t start_time, int64_t end_time,
    double open, double high, double low, double close, double volume) {
    
    // Add indices
    timestamp_indices.push_back(ts_idx);
    start_time_indices.push_back(start_idx);
    end_time_indices.push_back(end_idx);
    
    // Add actual timestamp values
    timestamps.push_back(timestamp);
    start_timestamps.push_back(start_time);
    end_timestamps.push_back(end_time);
    
    // Add OHLCV data
    opens.push_back(open);
    highs.push_back(high);
    lows.push_back(low);
    closes.push_back(close);
    volumes.push_back(volume);
}

/**
 * Check if the result is empty (no bars).
 * 
 * @return True if no bars, false otherwise
 */
bool BarResult::empty() const {
    return timestamp_indices.empty();
}

/**
 * Get the number of bars in the result.
 * 
 * @return Number of bars
 */
size_t BarResult::size() const {
    return timestamp_indices.size();
}

bool BarResult::verify_timestamps() const {
    if (size() < 2) {
        return true;  // Not enough bars to check continuity
    }
    
    // Check if we have actual timestamps to verify
    if (!timestamps.empty() && timestamps.size() == size()) {
        // Check for timestamp ordering (should be monotonically increasing)
        for (size_t i = 1; i < timestamps.size(); ++i) {
            if (timestamps[i] < timestamps[i-1]) {
                return false;  // Timestamps are not in order
            }
        }
        
        // Check that end time of previous bar matches or precedes start time of next bar
        for (size_t i = 1; i < size(); ++i) {
            if (end_timestamps[i-1] > start_timestamps[i]) {
                return false;  // Gap between bars
            }
        }
    }
    
    return true;
} 