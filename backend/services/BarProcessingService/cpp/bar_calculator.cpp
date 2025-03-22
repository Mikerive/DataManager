#include "bar_calculator.h"
#include "calculators/volume_bar_calculator.h"
#include "calculators/tick_bar_calculator.h"
#include "calculators/time_bar_calculator.h"
#include "calculators/entropy_bar_calculator.h"
#include <stdexcept>

BarCalculator::BarCalculator() {
    initialize_calculators();
}

BarCalculator::~BarCalculator() {
    // Smart pointers will handle cleanup
}

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
    // Clear existing data
    timestamps_.clear();
    opens_.clear();
    highs_.clear();
    lows_.clear();
    closes_.clear();
    volumes_.clear();
    
    // Copy data to vectors
    timestamps_.assign(timestamps_ptr, timestamps_ptr + data_size);
    opens_.assign(opens_ptr, opens_ptr + data_size);
    highs_.assign(highs_ptr, highs_ptr + data_size);
    lows_.assign(lows_ptr, lows_ptr + data_size);
    closes_.assign(closes_ptr, closes_ptr + data_size);
    volumes_.assign(volumes_ptr, volumes_ptr + data_size);
    
    data_size_ = data_size;
}

BarResult BarCalculator::calculate_bars(const BarParams& params) {
    if (data_size_ == 0) {
        throw std::runtime_error("No data available for bar calculation");
    }
    
    // Initialize the result with bar type and ratio
    BarResult result(params.bar_type, params.ratio);
    
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
    
    return result;
}

std::unordered_map<std::string, BarResult> BarCalculator::batch_process(
    const std::vector<BarParams>& params_list
) {
    if (data_size_ == 0) {
        throw std::runtime_error("No data available for bar calculation");
    }
    
    std::unordered_map<std::string, BarResult> results;
    
    for (const auto& params : params_list) {
        std::string key = params.bar_type + "_" + std::to_string(params.ratio);
        results[key] = calculate_bars(params);
    }
    
    return results;
} 