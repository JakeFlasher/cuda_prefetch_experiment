Certainly! Below is an enhanced and streamlined version of your `README.md`. It removes redundant instructions, clarifies steps, and ensures all necessary information is included for a smooth installation and execution process.

---

# CUDA Prefetch Experiment (3090 Ti)

Welcome to the **CUDA Prefetch Experiment (3090 Ti)**! This project contains scripts and tools to explore and benchmark CUDA performance on NVIDIA GPUs like the RTX 3090 Ti using the CUDA Toolkit and Nsight Compute.

## 🚀 Getting Started

### 🔍 System Requirements

Ensure your system meets the following:

- **Operating System**: Ubuntu 22.04 (64-bit)
- **GPU**: NVIDIA GPU with Compute Capability 8.0 or higher (e.g., RTX 3090 Ti)
- **CUDA Toolkit**: Version 11.x or higher

### 🛠️ Installation Guide

Follow these steps to set up your environment:

#### 1. Download and Install the CUDA Keyring

Download the CUDA keyring package to ensure proper repository signing:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

#### 2. Update Package Lists

After adding the keyring, update your package database:

```bash
sudo apt-get update
```

#### 3. Install the CUDA Toolkit

Install the CUDA Toolkit, which includes the NVIDIA Compiler (`nvcc`) and other essential tools:

```bash
sudo apt-get install -y cuda-toolkit
sudo apt-get -y install cudnn9-cuda-12 (for CUTLASS)
```

*If the above command doesn't work, try:*

```bash
sudo apt install -y nvidia-cuda-toolkit
```

#### 4. Verify CUDA Installation

Check if `nvcc` (NVIDIA CUDA Compiler) is installed correctly:

```bash
nvcc --version
```

You should see output similar to:

```plaintext
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Feb_14_19:10:22_PST_2023
Cuda compilation tools, release 11.x, V11.x.xx
Build cuda_11.x.r11.x/compiler.32414927_0
```

#### 5. Install Nsight Compute (Optional but Recommended)

Nsight Compute is NVIDIA's interactive kernel profiler for CUDA applications.

1. **Download Nsight Compute**

   Visit the [Nsight Compute download page](https://developer.nvidia.com/nsight-compute) and download the latest `.run` installer for Linux.

2. **Make the Installer Executable**

   ```bash
   chmod +x nsight-compute*.run
   ```

3. **Run the Installer**

   ```bash
   sudo ./nsight-compute*.run
   ```

4. **Verify Installation**

   ```bash
   ncu --version
   ```

   You should see output similar to:

   ```plaintext
   Nsight Compute Host         2024.x.x
   Build id: Release_2024.x.x-xxxxxxx-xxxxx
   ```

> **Note**: Avoid running `ncu` with `sudo` to prevent permission issues related to deploying section files. Instead, run `ncu` as a regular user.

---

## 🧪 Running the Experiment

Once all dependencies are installed, follow these steps to run the experiment:

1. **Navigate to the Project Directory**

   ```bash
   cd cuda_prefetch_experiment_3090ti/
   ```

2. **Make Scripts Executable**

   ```bash
   chmod +x *.sh
   ```

3. **Run the Experiment Script**

   ```bash
   ./run.sh
   ```

---

## 📂 Project Structure

```
cuda_prefetch_experiment_3090ti/
├── run.sh                # Experiment script
├── README.md             # Project documentation
└── <other files>         # Additional scripts/tools
```

---

## 💡 Additional Notes

- **Remote Profiling**: If running on a server, ensure you have SSH access to manage GPU resources remotely.
- **GPU Profiling**: Use `ncu` (Nsight Compute) to analyze performance bottlenecks in your CUDA applications.
- **Driver Updates**: Keep your GPU drivers up-to-date for maximum compatibility and performance.

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute as per the terms of the license.

---

## 🤝 Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

---

## 📧 Support

For any questions or assistance, please reach out via the [Issues](https://github.com/your-repo/issues) section or contact the repository maintainer.

---

Enjoy exploring CUDA with this project! 🎉

---

### 🔗 Useful Links

- [NVIDIA Nsight Compute Documentation](https://developer.nvidia.com/nsight-compute)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Nsight Compute Profiling Guide FAQ](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq)

---

### 📝 Change Log

**v1.0.0**  
- Initial release with installation and usage instructions.

---

By following this improved `README.md`, users should find it easier to set up and run the CUDA Prefetch Experiment with clear, concise instructions and necessary information.
