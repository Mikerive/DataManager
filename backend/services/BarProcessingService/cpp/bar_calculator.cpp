#include "bar_calculator.h"
#include "calculators/volume_bar_calculator.h"
#include "calculators/tick_bar_calculator.h"
#include "calculators/time_bar_calculator.h"
#include "calculators/entropy_bar_calculator.h"
#include <stdexcept>
#include <sstream>

BarCalculator::BarCalculator() {
    initialize_calculators();
}

BarCalculator::~BarCalculator() = default;

void BarCalculator::initialize_calculators() {
    volume_calculator_ = std::make_unique<VolumeBarCalculator>();
    tick_calculator_ = std::make_unique<TickBarCalculator>();
    time_calculator_ = std::make_unique<TimeBarCalculator>();
    entropy_calculator_ = std::make_unique<EntropyBarCalculator>();
}

void BarCalculator::set_data(
    const int64_t* timestamps_ptr,
    const double* opens_ptr,
    const double* highs_ptr,
    const double* lows_ptr,
    const double* closes_ptr,
    const double* volumes_ptr,
    size_t data_size
) {
    // Validate input
    if (!timestamps_ptr || !opens_ptr || !highs_ptr || !lows_ptr || !closes_ptr || !volumes_ptr) {
        throw std::runtime_error("Null pointer provided to set_data");
    }
    
    if (data_size == 0) {
        throw std::runtime_error("Empty data provided to set_data");
    }
    
    // Validate timestamps are monotonically increasing
    for (size_t i = 1; i < data_size; ++i) {
        if (timestamps_ptr[i] < timestamps_ptr[i-1]) {
            std::stringstream error_msg;
            error_msg << "Timestamps must be monotonically increasing. "
                      << "Found timestamp[" << i << "] = " << timestamps_ptr[i]
                      << " < timestamp[" << (i-1) << "] = " << timestamps_ptr[i-1];
            throw std::runtime_error(error_msg.str());
        }
    }
    
    // Copy data to member vectors
    timestamps_.assign(timestamps_ptr, timestamps_ptr + data_size);
    opens_.assign(opens_ptr, opens_ptr + data_size);
    highs_.assign(highs_ptr, highs_ptr + data_size);
    lows_.assign(lows_ptr, lows_ptr + data_size);
    closes_.assign(closes_ptr, closes_ptr + data_size);
    volumes_.assign(volumes_ptr, volumes_ptr + data_size);
    data_size_ = data_size;
}

BarResult BarCalculator::calculate_bars(const BarParams& params) {
    // Validate that data has been set
    if (data_size_ == 0) {
        throw std::runtime_error("No data set for calculation");
    }
    
    // Route to the appropriate calculator based on bar type
    if (params.bar_type == "volume") {
        return volume_calculator_->calculate(
            timestamps_, opens_, highs_, lows_, closes_, volumes_, params
        );
    } else if (params.bar_type == "tick") {
        return tick_calculator_->calculate(
            timestamps_, opens_, highs_, lows_, closes_, volumes_, params
        );
    } else if (params.bar_type == "time") {
        return time_calculator_->calculate(
            timestamps_, opens_, highs_, lows_, closes_, volumes_, params
        );
    } else if (params.bar_type == "entropy") {
        return entropy_calculator_->calculate(
            timestamps_, opens_, highs_, lows_, closes_, volumes_, params
        );
    } else {
        throw std::runtime_error("Unknown bar type: " + params.bar_type);
    }
}

std::unordered_map<std::string, BarResult> BarCalculator::batch_process(
    const std::vector<BarParams>& params_list
) {
    // Validate that data has been set
    if (data_size_ == 0) {
        throw std::runtime_error("No data set for calculation");
    }
    
    std::unordered_map<std::string, BarResult> results;
    
    // Process each parameter set
    for (const auto& params : params_list) {
        // Create a unique key for this parameter set
        std::string key = params.bar_type + "_" + std::to_string(params.ratio);
        
        // Calculate the bars for this parameter set
        results[key] = calculate_bars(params);
        
        // Verify timeframe preservation for each result
        if (!results[key].verify_timestamps()) {
            throw std::runtime_error("Timeframe integrity verification failed for " + key);
        }
    }
    
    return results;
} 