Here's a beautiful and detailed GitHub `README.md` that documents the installation process based on the commands you provided. I've structured it to be clear, visually appealing, and beginner-friendly:

---

# CUDA Prefetch Experiment (3090 Ti)

Welcome to the **CUDA Prefetch Experiment (3090 Ti)** project! This repository contains scripts and tools to explore and benchmark CUDA performance on NVIDIA GPUs like the 3090 Ti using CUDA Toolkit and Nsight Compute.

This guide will walk you through installing the required dependencies to run this project effectively.

---

## 🚀 Getting Started

Before diving into the experiment, make sure your system meets the following **requirements**:

### ✅ System Requirements

- **Operating System**: Ubuntu 22.04 (64-bit)
- **GPU**: NVIDIA GPU with Compute Capability 8.0 or higher (e.g., RTX 3090 Ti)
- **CUDA Version**: CUDA Toolkit 11.x or higher

---

## 🛠️ Installation Guide

Follow these step-by-step instructions to install the necessary dependencies for this project.

### 1️⃣ Download the CUDA Keyring

First, download the CUDA keyring package to ensure proper repository signing:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```

### 2️⃣ Add the CUDA Keyring to Your System

Install the keyring package using `dpkg`:

```bash
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

### 3️⃣ Update the Package List

After adding the keyring, update your package database:

```bash
sudo apt-get update
```

### 4️⃣ Install the CUDA Toolkit

Install the CUDA Toolkit, which includes the **NVIDIA Compiler (`nvcc`)** and other essential tools:

```bash
sudo apt-get install cuda-toolkit
```

> **Note**: If the above command doesn't work, you can also try:
>
> ```bash
> sudo apt install nvidia-cuda-toolkit
> ```

### 5️⃣ Verify CUDA Installation

Check if `nvcc` (NVIDIA CUDA Compiler) is installed correctly:

```bash
nvcc --version
```

If installed properly, you should see the CUDA version in the output.

### 6️⃣ Install Nsight Compute (Optional but Recommended)

To analyze and profile your CUDA applications, you will need **Nsight Compute**. Follow these steps:

1. Download the Nsight Compute `.run` file manually from NVIDIA's [Nsight Compute download page](https://developer.nvidia.com/nsight-compute).
2. Make the installer executable:

   ```bash
   chmod a+x nsight-compute*.run
   ```

3. Run the installer with superuser privileges:

   ```bash
   sudo ./nsight-compute*.run
   ```

4. Test it by running:

   ```bash
   ncu --version
   ```

---

## 🧪 Running the Experiment

Once all dependencies are installed, follow the steps below to start the experiment:

1. Navigate to the project directory:

   ```bash
   cd cuda_prefetch_experiment_3090ti/
   ```

2. Make the scripts executable:

   ```bash
   chmod a+x *.sh
   ```

3. Run the experiment script:

   ```bash
   ./run.sh
   ```

---

## 📂 Project Structure

This is the general structure of the repository:

```plaintext
cuda_prefetch_experiment_3090ti/
├── run.sh                # Experiment script
├── README.md             # Project documentation
├── <other files>         # Additional scripts/tools
```

---

## 💡 Additional Notes

- If you're running this on a server, make sure you have SSH access to manage GPU resources remotely.
- For advanced GPU profiling, use `ncu` (Nsight Compute) to analyze performance bottlenecks.
- Always ensure your GPU drivers are up-to-date for maximum compatibility.

---

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute as per the terms of the license.

---

## 🤝 Contributing

Contributions are welcome! If you encounter any issues or have suggestions to improve the project, feel free to open an issue or submit a pull request.

---

## 📧 Support

If you have any questions or need help, feel free to reach out via the [Issues](https://github.com/your-repo/issues) section or contact the repository maintainer.

---

Enjoy exploring CUDA with this project! 🎉

--- 

Let me know if you'd like to modify or add anything specific!
