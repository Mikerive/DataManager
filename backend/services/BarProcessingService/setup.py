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
    os.path.join(CPP_DIR, "utils", "entropy_utils.cpp")
]

# Check if we're running in a Windows environment
is_windows = (sys.platform == 'win32')

# Helper to locate the pybind11 include directory
def get_pybind_include():
    import pybind11
    return pybind11.get_include()

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

# Define the extension module
ext_modules = [
    Extension(
        'cpp_ext.bar_calculator_cpp',
        sources=CPP_SOURCES,
        include_dirs=[
            get_pybind_include(),
            INCLUDE_DIR,
            CALCULATORS_DIR,
            UTILS_DIR
        ],
        language='c++'
    ),
]

# Get the pybind11 package information
def get_pybind_requires():
    return ["pybind11>=2.6.0"]

# Setup function
setup(
    name='bar_calculator_cpp',
    version='0.1.0',
    author='AlgoTrader Team',
    author_email='info@algotrader.com',
    description='C++ implementation of bar calculation for the BarProcessingService',
    long_description='',
    ext_modules=ext_modules,
    install_requires=get_pybind_requires(),
    setup_requires=get_pybind_requires(),
    cmdclass={'build_ext': BuildExt},
    packages=['cpp_ext'],
    package_dir={'cpp_ext': 'cpp_ext'},
    package_data={'cpp_ext': ['*.so', '*.pyd']},
    zip_safe=False,
    python_requires='>=3.6',
)

# Create the cpp_ext directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, 'cpp_ext'), exist_ok=True)

# Create an __init__.py file in the cpp_ext directory
with open(os.path.join(BASE_DIR, 'cpp_ext', '__init__.py'), 'w') as f:
    f.write("# This file is automatically generated by setup.py\n") 