#!/bin/bash
# -----------------------------------------------------------------
# Script: run.sh
# Description: Compiles all *.cu files in the current directory using
#              nvcc and profiles each executable using ncu, but with
#              an automatic lookup table to detect GPU architecture
#              (compute capability) instead of a hard-coded sm value.
#              It also continues even if an error occurs.
# Author: [Your Name]
# Date: [Date]
# -----------------------------------------------------------------

# ----------- Colorful output functions -----------
function echo_info {
    echo -e "\e[34m[INFO]\e[0m $1"
}

function echo_success {
    echo -e "\e[32m[SUCCESS]\e[0m $1"
}

function echo_error {
    echo -e "\e[31m[ERROR]\e[0m $1"
}

# -------------- Parse arguments --------------
MODE="pc"
NCU_CMD="ncu"  # default if pc
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --server)
            MODE="server"
            shift
            ;;
        --pc)
            MODE="pc"
            shift
            ;;
        *)
            # If there's an extra parameter, treat it as a path to ncu
            NCU_CMD="$1"
            shift
            ;;
    esac
done

# If mode is server, ask for ncu path if not already set
if [ "$MODE" == "server" ]; then
    if [ "$NCU_CMD" == "ncu" ]; then
        echo_info "Please input the full path to Nsight Compute (ncu) executable:"
        read -r NCU_CMD
    fi
    echo_info "Using sudo to run '$NCU_CMD'"
fi
# ncu -o "$report_name" --set full -f "./$base"
# -------------- Decide how to run ncu commands --------------
function run_ncu {
    local out_file="$1"
    local app="$2" 

    if [ "$MODE" == "server" ]; then
        sudo "$NCU_CMD" -o "$out_file" --set full -f $app
        echo "sudo "$NCU_CMD" -o "$out_file" --set full -f $app"
    else
        "$NCU_CMD" -o "$out_file" --set full -f $app
    fi
}

# ----------- Function to run commands safely -----------
# Continues even if error occurs, printing an error message but not exiting.
function run_command {
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo_error "Failed command: $*"
    fi
}

# -------------------------------------------------------------------
# Detect GPU name using nvidia-smi
# -------------------------------------------------------------------
gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
GPUNAME=${gpu// /_}
if [ -z "$GPUNAME" ]; then
    GPUNAME="UnknownGPU"
    echo_error "Could not detect GPU name. Using 'UnknownGPU'."
fi
echo_info "Detected GPU Name: $GPUNAME"

# -------------------------------------------------------------------
# Create subdirectories
# -------------------------------------------------------------------
mkdir -p "$GPUNAME/bin"
mkdir -p "$GPUNAME/ncu_reports"

# ----------- Hard-coded GPU to compute capability table -----------
# You may extend or adjust these mappings as needed.
declare -A GPU_CC_MAP=(
    ["NVIDIA H100"]="90"
    ["NVIDIA L4"]="89"
    ["NVIDIA L40"]="89"
    ["NVIDIA A100"]="80"
    ["NVIDIA A40"]="86"
    ["NVIDIA A30"]="80"
    ["NVIDIA A10"]="86"
    ["NVIDIA A16"]="86"
    ["NVIDIA A2"]="86"
    ["NVIDIA T4"]="75"
    ["NVIDIA V100"]="70"
    ["GeForce RTX 4090"]="89"
    ["GeForce RTX 4080"]="89"
    ["GeForce RTX 4070 Ti"]="89"
    ["GeForce RTX 4060 Ti"]="89"
    ["GeForce RTX 3090 Ti"]="86"
    ["GeForce RTX 3090"]="86"
    ["GeForce RTX 3080 Ti"]="86"
    ["GeForce RTX 3080"]="86"
    ["GeForce RTX 3070 Ti"]="86"
    ["GeForce RTX 3070"]="86"
    ["Geforce RTX 3060 Ti"]="86"
    ["Geforce RTX 3060"]="86"
)

# ----------- Function to detect GPU and get its CC -----------
function get_compute_capability {
    # We retrieve the GPU name from nvidia-smi
    local gpu_name
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
    
    if [ -z "$gpu_name" ]; then
        echo_error "Could not detect GPU name from nvidia-smi. Defaulting to sm_70."
        echo "70"  # default compute capability
        return
    fi

    # Trim leading/trailing spaces
    gpu_name="$(echo "$gpu_name" | xargs)"

    local cc="${GPU_CC_MAP[$gpu_name]}"
    if [ -z "$cc" ]; then
        # echo_error "GPU '$gpu_name' not in lookup table. Defaulting to sm_70."
        echo "70"
    else
        # echo_info "Detected GPU: '$gpu_name' => Compute Capability sm_$cc"
        echo "$cc"
    fi
}

# ----------- Main script logic -----------

# This script no longer uses 'set -e' so it won’t exit on first error

# Attempt to detect the GPU and retrieve the corresponding compute capability
COMPUTE_CAPABILITY=$(get_compute_capability)

echo_info "Using compute capability: sm_${COMPUTE_CAPABILITY}"

# Check if nvcc is installed
if ! command -v nvcc &> /dev/null; then
    echo_error "nvcc could not be found. Please install the NVIDIA CUDA Toolkit."
fi

# Check if ncu is installed
if ! command -v ncu &> /dev/null; then
    echo_error "ncu (NVIDIA Nsight Compute) could not be found in PATH."
    echo_info "You can still compile, but profiling might fail."
fi

# Iterate over all *.cu files in the current directory
shopt -s nullglob
CU_FILES=( *.cu )

if [ ${#CU_FILES[@]} -eq 0 ]; then
    echo_error "No .cu files found in the current directory."
    exit 0
fi

for file in "${CU_FILES[@]}"; do
    base="${file%.cu}"
    exe="$GPUNAME/bin/$base"

    echo_info "Compiling $file..."
    run_command nvcc "$file" -arch="sm_${COMPUTE_CAPABILITY}" -O3 -o "$exe"

    # Only attempt to profile if compilation succeeded (executable exists)
    if [ -x "$exe" ]; then
        echo_success "Compilation successful: $exe"

        echo_info "Profiling $exe with NVIDIA Nsight Compute..."
        # Define the output report name
    
        report_name="$GPUNAME/ncu_reports/${base}_report"

        # Run ncu profiling
        # run_command ncu -o "$report_name" --set full -f "./$base"
        run_ncu "$report_name" "./$exe"
        if [ -f "${report_name}.ncu-rep" ]; then
            echo_success "Profiling completed: ${report_name}.ncu-rep"
        else
            echo_error "Profiling step failed; no report was generated."
        fi
    else
        echo_error "Skipping profiling because $base does not exist."
    fi

    echo "---------------------------------------------"
done

echo_success "Done. All possible .cu files have been compiled and (if successful) profiled."

