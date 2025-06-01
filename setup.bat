@echo off
echo Setting up the environment for Perchance Revival...

:: Define the virtual environment directory name
set VENV_DIR=venv

:: Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.8+ and make sure it's added to your system's PATH.
    echo You can download Python from https://www.python.org/downloads/windows/
    goto end
)
echo Found Python.

:: Create a virtual environment if it doesn't exist
if not exist %VENV_DIR% (
    echo Creating virtual environment "%VENV_DIR%"...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment. Check Python installation or permissions.
        goto end
    )
    echo Virtual environment created.
) else (
    echo Virtual environment "%VENV_DIR%" already exists. Skipping creation.
)

:: Activate the virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment. Check virtual environment setup.
    goto end
)
echo Virtual environment activated.

:: Install core dependencies from requirements.txt
echo Installing core dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install core dependencies from requirements.txt.
    echo Ensure requirements.txt exists and is correct. See error above.
    goto deactivate_and_end
)
echo Core dependencies installed successfully.

:: --- Attempt to Install PyTorch CUDA version by default ---
:: This assumes a compatible NVIDIA GPU and drivers are present.
:: If this fails, the user might not have a GPU, compatible drivers, or the specified CUDA version.
echo.
echo Attempting to install PyTorch with CUDA support (recommended for NVIDIA GPUs)...
echo This will install the PyTorch version compatible with CUDA 12.1.
echo If you have a different CUDA version or no NVIDIA GPU, this might fail.
echo (Checking for compatible CUDA versions: https://pytorch.org/get-started/locally/)

:: --- !!! IMPORTANT !!! ---
:: Specify the CUDA version here. cu121 is currently common. Adjust if needed.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
:: -------------------------

if %errorlevel% neq 0 (
    echo.
    echo !!! IMPORTANT: Failed to install PyTorch CUDA version (cu121) !!!
    echo This usually means:
    echo 1. You do not have a compatible NVIDIA GPU.
    echo 2. Your NVIDIA drivers are not installed or are too old for CUDA 12.1.
    echo 3. You need a different CUDA version of PyTorch.
    echo 4. An internet or other installation issue occurred.
    echo.
    echo Error details:
    echo %errorlevel%
    echo.
    echo To proceed, you have two options:
    echo Option A: Troubleshoot your NVIDIA GPU/drivers/CUDA version (see https://pytorch.org/get-started/locally/)
    echo Option B: Install the CPU-only version of PyTorch (much slower generation)
    echo.
    echo To install the CPU version manually (while in the activated venv):
    echo pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo.
    echo After addressing the issue or installing the CPU version, you can try running the app with run.bat.
    goto deactivate_and_end # Still exit setup script on failure
) else (
    echo.
    echo PyTorch CUDA version installed successfully.
)

echo.
echo --- SETUP COMPLETE ---
echo Environment "%VENV_DIR%" created/activated and dependencies installed.
echo.
echo PyTorch with CUDA support was successfully installed.
echo You can now run the application using run.bat.
echo.

goto end

:deactivate_and_end
:: Deactivate the virtual environment before exiting on error
echo Deactivating virtual environment...
deactivate
echo.

:end
echo Press any key to exit...
pause
