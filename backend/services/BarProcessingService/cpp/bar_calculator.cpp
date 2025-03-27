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
    switch (params.bar_type) {
        case BarType::Volume:
            return volume_calculator_->calculate(
                timestamps_, opens_, highs_, lows_, closes_, volumes_, params
            );
        
        case BarType::Tick:
            return tick_calculator_->calculate(
                timestamps_, opens_, highs_, lows_, closes_, volumes_, params
            );
        
        case BarType::Time:
            return time_calculator_->calculate(
                timestamps_, opens_, highs_, lows_, closes_, volumes_, params
            );
        
        case BarType::Entropy:
            return entropy_calculator_->calculate(
                timestamps_, opens_, highs_, lows_, closes_, volumes_, params
            );
        
        default:
            std::string error_msg = "Unsupported bar type: " + to_string(params.bar_type);
            throw std::runtime_error(error_msg);
    }
}

std::vector<BarResult> BarCalculator::batch_process(const std::vector<BarParams>& all_params) {
    std::vector<BarResult> results;
    for (const auto& params : all_params) {
        try {
            results.push_back(calculate_bars(params));
        } catch (const std::exception& e) {
            std::string error_msg = "Error processing bar type " + 
                to_string(params.bar_type) + ": " + e.what();
            throw std::runtime_error(error_msg);
        }
    }
    return results;
}

std::string get_bar_type_name(BarType bar_type) {
    return to_string(bar_type);
} 