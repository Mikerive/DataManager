#include "bar_result.h"

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
    
    // Extract actual timestamps from indices
    py::array_t<py::object> ts_out(timestamp_indices.size());
    py::array_t<py::object> start_ts_out(start_time_indices.size());
    py::array_t<py::object> end_ts_out(end_time_indices.size());
    
    // Get the array buffer for timestamps
    // Note: This requires careful handling since timestamps are Python objects
    for (size_t i = 0; i < timestamp_indices.size(); i++) {
        ts_out.mutable_at(i) = timestamps[timestamp_indices[i]];
        start_ts_out.mutable_at(i) = timestamps[start_time_indices[i]];
        end_ts_out.mutable_at(i) = timestamps[end_time_indices[i]];
    }
    
    // Add arrays to the result dictionary
    result["timestamp"] = ts_out;
    result["start_time"] = start_ts_out;
    result["end_time"] = end_ts_out;
    result["open"] = opens;
    result["high"] = highs;
    result["low"] = lows;
    result["close"] = closes;
    result["volume"] = volumes;
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