#pragma once

#include <string>

// Enum for bar types
enum class BarType {
    Time = 0,
    Volume = 1,
    Tick = 2,
    Dollar = 3,
    Information = 4,
    Entropy = 5
};

// Comparison operators for BarType
inline bool operator==(const BarType& lhs, const BarType& rhs) {
    return static_cast<int>(lhs) == static_cast<int>(rhs);
}

inline bool operator!=(const BarType& lhs, const BarType& rhs) {
    return static_cast<int>(lhs) != static_cast<int>(rhs);
}

// Conversion to string for error messages
inline std::string to_string(const BarType& type) {
    switch (type) {
        case BarType::Time: return "Time";
        case BarType::Volume: return "Volume";
        case BarType::Tick: return "Tick";
        case BarType::Dollar: return "Dollar";
        case BarType::Information: return "Information";
        case BarType::Entropy: return "Entropy";
        default: return "Unknown(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

/**
 * Structure to hold parameters for bar calculations.
 * 
 * This structure contains all the necessary parameters for configuring
 * different types of bar calculations such as volume bars, tick bars,
 * entropy bars, etc.
 */
struct BarParams {
    BarType bar_type;
    double ratio;
    
    // Adaptive threshold parameters
    size_t lookback_window = 20;
    
    // Additional parameters for specific bar types
    size_t window_size = 20;
    std::string method = "shannon";
    double q_param = 2.0;
    
    // Default constructor
    BarParams() : bar_type(BarType::Time), ratio(1.0) {}
    
    // Constructor with required parameters
    BarParams(BarType type, double r) : bar_type(type), ratio(r) {}
    
    // Full constructor
    BarParams(
        BarType type, 
        double r, 
        size_t lookback = 20,
        size_t window = 20,
        const std::string& method_name = "shannon",
        double q = 2.0
    ) : 
        bar_type(type), 
        ratio(r), 
        lookback_window(lookback),
        window_size(window),
        method(method_name),
        q_param(q)
    {}
}; 