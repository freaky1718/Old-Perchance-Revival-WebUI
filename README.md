# Perchance Revival - Easy Local SD 1.5 Image Generation

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to **Perchance Revival**! This is a user-friendly Gradio web application designed to bring back the experience of the old Perchance image generator by featuring the specific Stable Diffusion 1.5 models and common generation parameters/presets it used.

Generate images effortlessly and for free, directly on your own PC. This version is designed to **prioritize and utilize your NVIDIA GPU** for significantly faster generation if you have one, while still offering a CPU option for broader compatibility. The goal is to make local SD 1.5 generation as simple as possible, especially for those familiar with the old Perchance site.

Welcome to **Perchance Revival**! This is a user-friendly Gradio web application designed to bring back the experience of the old Perchance image generator by featuring the specific Stable Diffusion 1.5 models and common generation parameters/presets it used.

Generate images effortlessly and for free, directly on your own PC. This version is designed to **prioritize and utilize your NVIDIA GPU** for significantly faster generation if you have one, while still offering a CPU option for broader compatibility. The goal is to make local SD 1.5 generation as simple as possible, especially for those familiar with the old Perchance site.

01-06-2025:
> ⚠️ **Early Development Notice**  
> This app is still in **very early development** – as in, *started today* early. It can already generate images successfully, but expect bugs, missing polish, and future changes (including some big ones). It’s functional but **not production-ready yet**. Stability is good so far, but this is more of a preview than a final release.

This project is primarily designed for **Windows** users, offering a straightforward setup with easy-to-use batch files. Manual setup options are also provided for other platforms or advanced users.


This project is primarily designed for **Windows** users, offering a straightforward setup with easy-to-use batch files. Manual setup options are also provided for other platforms or advanced users.

## Application Screenshot:

![Screenshot of the Perchance Revival Web UI](images/ciphercore01.png)
*(Note: This screenshot shows the core layout. The 'Perchance Revival' version features specific model options and is focused on recreating that experience.)*

## ✨ Features

*   **Focused on Perchance SD 1.5 Models & Presets:** Access the models and common generation parameters/presets popular on the old Perchance site directly within the app for a nostalgic experience. These models are downloaded and cached automatically from the Hugging Face Hub on first use.
*   **Add Your Own Models:** Easily load *additional* Stable Diffusion 1.5 models (in `diffusers` format) from a local `./checkpoints` folder, alongside the featured Perchance models.
*   **GPU Accelerated (Recommended) or CPU:**
    *   Default setup **attempts GPU installation** for faster generation (requires compatible NVIDIA GPU and drivers).
    *   Automatically falls back to CPU if GPU setup fails, or allows manual CPU-only installation (generation is much slower on CPU).
*   **Comprehensive Generation Controls:**
    *   **Positive & Negative Prompts:** Tell the AI what to include and what to avoid.
    *   **Inference Steps:** Control the detail level (more steps = often more detail, but slower).
    *   **CFG Scale:** Adjust how closely the image follows your prompt (higher = stricter adherence).
    *   **Schedulers:** Experiment with different sampling methods (Euler, DPM++ 2M, DDPM, LMS).
    *   **Image Sizes:** Standard SD1.5 resolutions, plus a "hire.fix" option (interpreted as 1024x1024).
    *   **Seed Control:** Get reproducible results with a specific seed, or use -1 for random.
*   **User-Friendly Interface:**
    *   Clean and intuitive Gradio UI.
    *   Organized controls with advanced settings in an accordion.
    *   Direct image display with convenient download and share options.
*   **Safety First (Note):** The built-in safety checker is **disabled** in this version to match the flexibility of the old Perchance generator. Please be mindful of the content you generate.

## 🚀 Prerequisites

*   **Windows Operating System:** The provided batch files (`.bat`) are for Windows. For other operating systems, follow the manual setup steps below.
*   **Python:** 3.8 or higher. Ensure Python is installed and added to your system's PATH. Download from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
*   **Git:** (Required for manual setup and updating) For cloning the repository.
*   **Hardware:**
    *   A modern CPU is required.
    *   For the default GPU setup to succeed: A compatible **NVIDIA GPU** with up-to-date drivers installed. At least 6-8GB VRAM is recommended for 512x512, more for larger sizes or multiple images.
*   **Internet Connection:** Required for downloading models from Hugging Face Hub and for updates.

## 📦 Easy Setup (Windows - Download & Run)

This is the recommended and easiest method for most Windows users.

1.  **Download the project:**
    *   Go to the GitHub repository page: `https://github.com/Raxephion/Old-Perchance-Revival-WebUI`
    *   Click the green "<> Code" button.
    *   Click "Download ZIP".
2.  **Extract the ZIP:** Extract the downloaded ZIP file to a location on your computer (e.g., your Documents folder or Desktop). This will create a folder like `Old-Perchance-Revival-WebUI-main` (or similar). You can rename it if you prefer, for example, to `PerchanceRevival`.
3.  **Run the Setup Script:**
    *   Navigate into the extracted folder (e.g., `PerchanceRevival`).
    *   Find the file named `setup.bat`.
    *   **Double-click `setup.bat`** to run it.
    *   A command prompt window will open. Follow the instructions in the window. This script will create a Python virtual environment (`venv`), install all necessary core dependencies from `requirements.txt`, and **attempt to install the GPU-accelerated version of PyTorch (CUDA) by default**.
    *   **❗ IMPORTANT:** Read the output in the command prompt carefully during and after the script finishes.
        *   If the GPU installation **succeeds**, the output will confirm this. You are ready to run the app.
        *   If the GPU installation **fails** (e.g., no compatible NVIDIA GPU, incorrect drivers, or internet issues), the script will print specific error messages and **instructions on how to manually install the CPU-only version** of PyTorch as an alternative. Follow those instructions if the GPU install fails and you cannot resolve the underlying GPU/driver issue.
4.  **Prepare Additional Local Models (Optional):**
    *   Inside the extracted project folder (e.g., `PerchanceRevival`), create a directory named `checkpoints` (if `setup.bat` didn't create it).
    *   Place any *additional* Stable Diffusion 1.5 models (in `diffusers` format – a folder containing files like `model_index.json`, `unet/`, `vae/`, etc.) that you want to use *beyond* the featured Perchance models inside the `checkpoints` directory. The app will detect and list these alongside the default options.
        Example structure:
        ```
        PerchanceRevival/
        ├── checkpoints/          <-- For YOUR additional models
        │   ├── my-custom-model/
        │   │   ├── model_index.json
        │   │   ├── unet/
        │   │   └── ...
        │   └── another-local-model/
        │       └── ...
        ├── main.py              <-- Main application script
        ├── requirements.txt     <-- Dependency list
        ├── setup.bat            <-- Easy setup script
        ├── run.bat              <-- Easy run script
        ├── update.bat           <-- Easy update script
        ├── images/              <-- Folder containing example image
        │   └── ciphercore01.png <-- Example screenshot image file
        └── ...                  <-- Other app files
        ```

## 🔄 Updating the Application (Windows - Easy Method)

To get the latest code and dependency updates from this repository after using the easy setup:

*   Navigate to the project folder (e.g., `PerchanceRevival`).
*   Find the file named `update.bat`.
*   **Double-click `update.bat`** to run it.
*   A command prompt window will open and pull the latest changes from the GitHub repository and upgrade the Python packages in your virtual environment according to `requirements.txt`. It will **not** attempt to change your PyTorch installation (GPU vs CPU) unless specifically updated in `requirements.txt`.
*   **Important:** This assumes you have not made local changes that conflict with the repository updates. If `git pull` fails, you may need to handle merge conflicts manually or discard local changes.


## ▶️ Running the Application (Windows - Easy Method)

Once the setup is complete (including successful PyTorch installation, whether GPU or CPU), launch the Gradio web UI by double-clicking the `run.bat` file in your project folder (e.g., `PerchanceRevival`).

*   A command prompt window will open, activate the environment, and start the application.
*   A browser window should automatically open to the application (or a local URL will be provided in the console, usually `http://127.0.0.1:7860`).


---

## ⚙️ Manual Setup (Windows - Git Clone)

This method is for Windows users who are comfortable with Git.

1.  **Clone the Repository:** Open Command Prompt or PowerShell, navigate to where you want to download the project, and run:
    ```bash
    git clone https://github.com/Raxephion/Old-Perchance-Revival-WebUI.git
    cd Old-Perchance-Revival-WebUI
    ```
    *(Note: You can rename the `Old-Perchance-Revival-WebUI` directory after cloning if you prefer.)*
2.  **Proceed with Batch Files:** Continue by following **Step 2 (Run the Setup Script)**, **Step 4 (Prepare Additional Local Models)** (for your *own* checkpoints), **Running**, and **Updating** instructions from the **📦 Easy Setup (Windows - Download & Run)** section above. Make sure the `images` folder exists and contains `ciphercore01.png` if you use this method and they aren't already in the cloned repo.

## 🛠️ Manual Setup, Running & Updating (For Linux/macOS or Advanced Users)

If you are not on Windows or prefer a manual command-line approach:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Raxephion/Old-Perchance-Revival-WebUI.git
    cd Old-Perchance-Revival-WebUI
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies (including PyTorch):**
    *   Install core dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   Install PyTorch: **This step is crucial and depends on your hardware.**
        *   **For NVIDIA GPU with CUDA (Recommended for speed):** Find the appropriate command for your CUDA version on the PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). Example for CUDA 12.1:
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ```
        *   **For CPU ONLY (if no NVIDIA GPU or CUDA fails):**
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ```
4.  **Prepare Additional Local Models (Optional):** Follow Step 4 from the **📦 Easy Setup (Windows - Download & Run)** section above (the part about the `checkpoints` folder for *your own* models).
5.  **Run the Application:**
    ```bash
    python main.py
    ```
    Ensure your virtual environment is activated (`source venv/bin/activate`) before running this command.
6.  **Updating Manually:**
    *   Navigate to the project directory in your terminal (`cd Old-Perchance-Revival-WebUI`).
    *   Ensure your virtual environment is activated (`source venv/bin/activate`).
    *   Pull the latest code: `git pull`
    *   Update dependencies (excluding PyTorch unless you change the `requirements.txt` or manually upgrade): `pip install -r requirements.txt --upgrade`
    *   Deactivate the environment: `deactivate`

## ⚙️ Uninstall:

1.  **Delete the main directory (folder) - this app is completely portable.**


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.
