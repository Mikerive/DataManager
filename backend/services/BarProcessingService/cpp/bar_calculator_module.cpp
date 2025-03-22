#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bar_calculator.h"
#include "bar_params.h"
#include "bar_result.h"

namespace py = pybind11;

/**
 * Convert a numpy array to a vector
 */
template <typename T>
std::vector<T> numpy_to_vector(py::array_t<T> array) {
    py::buffer_info buf = array.request();
    auto ptr = static_cast<T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.size);
}

/**
 * Main module definition for the bar calculator
 */
PYBIND11_MODULE(bar_calculator_cpp, m) {
    m.doc() = "C++ Bar calculator module for efficient bar computation";
    
    // Define the BarParams struct
    py::class_<BarParams>(m, "BarParams")
        .def(py::init<>())
        .def_readwrite("bar_type", &BarParams::bar_type)
        .def_readwrite("ratio", &BarParams::ratio)
        .def_readwrite("window_size", &BarParams::window_size)
        .def_readwrite("avg_window", &BarParams::avg_window)
        .def_readwrite("method", &BarParams::method)
        .def_readwrite("q_param", &BarParams::q_param);
    
    // Define the BarResult class
    py::class_<BarResult>(m, "BarResult")
        .def(py::init<std::string, double>())
        .def("to_dict", &BarResult::to_dict)
        .def("add_bar", &BarResult::add_bar)
        .def("empty", &BarResult::empty)
        .def("size", &BarResult::size);
    
    // Define the BarCalculator class
    py::class_<BarCalculator>(m, "BarCalculator")
        .def(py::init<>())
        .def("set_data", [](BarCalculator& self, 
                           py::array_t<int64_t> timestamps,
                           py::array_t<double> opens,
                           py::array_t<double> highs,
                           py::array_t<double> lows,
                           py::array_t<double> closes,
                           py::array_t<double> volumes) {
            // Get buffer info for all arrays
            py::buffer_info ts_buf = timestamps.request();
            py::buffer_info opens_buf = opens.request();
            py::buffer_info highs_buf = highs.request();
            py::buffer_info lows_buf = lows.request();
            py::buffer_info closes_buf = closes.request();
            py::buffer_info volumes_buf = volumes.request();
            
            // Check array sizes
            if (ts_buf.size != opens_buf.size || ts_buf.size != highs_buf.size ||
                ts_buf.size != lows_buf.size || ts_buf.size != closes_buf.size ||
                ts_buf.size != volumes_buf.size) {
                throw std::runtime_error("Input arrays must have the same size");
            }
            
            // Call the C++ method with pointers and size
            self.set_data(
                static_cast<int64_t*>(ts_buf.ptr),
                static_cast<double*>(opens_buf.ptr),
                static_cast<double*>(highs_buf.ptr),
                static_cast<double*>(lows_buf.ptr),
                static_cast<double*>(closes_buf.ptr),
                static_cast<double*>(volumes_buf.ptr),
                ts_buf.size
            );
        })
        .def("calculate_bars", &BarCalculator::calculate_bars)
        .def("batch_process", [](BarCalculator& self, py::list params_list) {
            // Convert the Python list to a C++ vector of BarParams
            std::vector<BarParams> cpp_params;
            for (auto item : params_list) {
                cpp_params.push_back(item.cast<BarParams>());
            }
            
            // Call the C++ method
            auto result = self.batch_process(cpp_params);
            
            // Convert the result back to a Python dictionary
            py::dict py_result;
            for (const auto& pair : result) {
                py_result[py::str(pair.first)] = pair.second;
            }
            
            return py_result;
        });

    // Add version information
    m.attr("__version__") = "0.1.0";
} 