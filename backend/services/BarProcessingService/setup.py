import os
import sys
from glob import glob
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(BASE_DIR, "cpp")
INCLUDE_DIR = CPP_DIR
CALCULATORS_DIR = os.path.join(CPP_DIR, "calculators")
UTILS_DIR = os.path.join(CPP_DIR, "utils")

# Source files
CPP_SOURCES = [
    os.path.join(CPP_DIR, "bar_calculator_module.cpp"),
    os.path.join(CPP_DIR, "bar_calculator.cpp"),
    os.path.join(CPP_DIR, "bar_result.cpp"),
    os.path.join(CPP_DIR, "calculators", "volume_bar_calculator.cpp"),
    os.path.join(CPP_DIR, "calculators", "tick_bar_calculator.cpp"),
    os.path.join(CPP_DIR, "calculators", "time_bar_calculator.cpp"),
    os.path.join(CPP_DIR, "calculators", "entropy_bar_calculator.cpp"),
    os.path.join(CPP_DIR, "utils", "entropy_utils.cpp"),
    os.path.join(CPP_DIR, "utils", "adaptive_threshold_calculator.cpp")
]

# Check if we're running in a Windows environment
is_windows = (sys.platform == 'win32')

# Custom build class for C++11 or higher
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/std:c++14', '/O2'],
        'unix': ['-std=c++14', '-O3', '-Wall', '-fvisibility=hidden']
    }
    
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
            
        build_ext.build_extensions(self)

    def finalize_options(self):
        super().finalize_options()
        # Add pybind11 include directory
        import pybind11
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())

# Define the extension module
ext_modules = [
    Extension(
        '_bar_processor',
        sources=CPP_SOURCES,
        include_dirs=[
            INCLUDE_DIR,
            CALCULATORS_DIR,
            UTILS_DIR
        ],
        language='c++'
    ),
]

# Setup function
setup(
    name='bar_processor',
    version='0.1.0',
    author='AlgoTrader Team',
    author_email='info@algotrader.com',
    description='C++ implementation of bar calculation for the BarProcessingService',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.6',
    # Explicitly specify no packages to avoid auto-discovery conflicts
    packages=[],
) 