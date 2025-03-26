#include "bar_result.h"
#include <pybind11/stl.h>  // Add this include for automatic STL conversions

/**
 * Constructor for BarResult.
 * 
 * @param type Bar type string
 * @param r Parameter ratio/threshold value
 */
BarResult::BarResult(const std::string& type, double r) 
    : bar_type(type), ratio(r) {}

/**
 * Convert the bar results to a Python dictionary.
 * 
 * @param timestamps Python array containing the original timestamps
 * @return Python dictionary with bar data
 */
py::dict BarResult::to_dict(py::array timestamps) const {
    py::dict result;
    
    // Create new arrays for timestamps from indices
    py::list ts_out;
    py::list start_ts_out;
    py::list end_ts_out;
    
    // Get the array buffer for timestamps
    // Use py::cast to convert indices to Python objects
    for (size_t i = 0; i < timestamp_indices.size(); i++) {
        ts_out.append(timestamps[py::cast(timestamp_indices[i])]);
        start_ts_out.append(timestamps[py::cast(start_time_indices[i])]);
        end_ts_out.append(timestamps[py::cast(end_time_indices[i])]);
    }
    
    // Add arrays to the result dictionary
    result["timestamps"] = ts_out;
    result["start_times"] = start_ts_out;
    result["end_times"] = end_ts_out;
    
    // Convert C++ vectors to Python lists for numeric data
    py::list opens_list;
    py::list highs_list;
    py::list lows_list;
    py::list closes_list;
    py::list volumes_list;
    
    for (size_t i = 0; i < opens.size(); i++) {
        opens_list.append(opens[i]);
        highs_list.append(highs[i]);
        lows_list.append(lows[i]);
        closes_list.append(closes[i]);
        volumes_list.append(volumes[i]);
    }
    
    result["opens"] = opens_list;
    result["highs"] = highs_list;
    result["lows"] = lows_list;
    result["closes"] = closes_list;
    result["volumes"] = volumes_list;
    result["bar_type"] = bar_type;
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