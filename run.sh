#!/bin/bash

# ------------------------------------------------------------
# Script: compile_and_profile.sh
# Description: Compiles all *.cu files in the current directory
#              using nvcc and profiles each executable using ncu.
# Author: [Your Name]
# Date: [Date]
# ------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display messages in color
function echo_info {
    echo -e "\e[34m[INFO]\e[0m $1"
}

function echo_success {
    echo -e "\e[32m[SUCCESS]\e[0m $1"
}

function echo_error {
    echo -e "\e[31m[ERROR]\e[0m $1"
}

# Check if nvcc is installed
if ! command -v nvcc &> /dev/null
then
    echo_error "nvcc could not be found. Please install the NVIDIA CUDA Toolkit."
    exit 1
fi

# Check if ncu is installed
if ! command -v ncu &> /dev/null
then
    echo_error "ncu (NVIDIA Nsight Compute) could not be found. Please install it from https://developer.nvidia.com/nsight-compute."
    exit 1
fi

# Iterate over all *.cu files in the current directory
for file in *.cu; do
    # Check if there are no .cu files
    if [ "$file" == "*.cu" ]; then
        echo_error "No .cu files found in the current directory."
        exit 1
    fi

    # Extract the base name by removing the .cu extension
    base="${file%.cu}"

    echo_info "Compiling $file..."

    # Compile the .cu file using nvcc
    nvcc "$file" -arch=sm_80 -O3 -o "$base"

    echo_success "Compilation successful: $base"

    echo_info "Profiling $base with NVIDIA Nsight Compute..."

    # Define the output report name
    report_name="${base}_report"

    # Run ncu profiling
    ncu -o "$report_name" --set full -f "./$base"

    echo_success "Profiling completed: ${report_name}.ncu-rep"

    echo "---------------------------------------------"
done

echo_success "All files have been compiled and profiled successfully."
