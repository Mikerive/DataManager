#pragma once

#include <string>

/**
 * Structure to hold parameters for bar calculations.
 * 
 * This structure contains all the necessary parameters for configuring
 * different types of bar calculations such as volume bars, tick bars,
 * entropy bars, etc.
 */
struct BarParams {
    /** Type of bar ('volume', 'tick', 'time', 'entropy', etc.) */
    std::string bar_type;
    
    /** 
     * The primary parameter value for the bar calculation.
     * For volume bars: volume threshold as a multiple of average volume
     * For tick bars: number of ticks per bar
     * For time bars: time interval in minutes
     * For entropy bars: entropy threshold
     */
    double ratio;
    
    /** Window size for calculations that use rolling windows (e.g., entropy) */
    int window_size = 100;
    
    /** Window size for calculating moving averages (e.g., volume bars) */
    int avg_window = 200;
    
    /** Method for calculations that support different algorithms (e.g., entropy) */
    std::string method = "shannon";
    
    /** Parameter for Tsallis entropy calculation */
    double q_param = 1.5;
}; 