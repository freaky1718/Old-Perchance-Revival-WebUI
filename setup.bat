@echo off
echo Setting up the environment for Perchance Revival...

:: Define the virtual environment directory name
set VENV_DIR=venv

:: Change directory to the script's location (Ensures script runs from its folder)
cd /d "%~dp0"

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
echo This will install the latest recommended PyTorch version compatible with CUDA 12.1.
echo If you have a different CUDA version or no NVIDIA GPU, this might fail.
echo (Checking for compatible CUDA versions and specific commands: https://pytorch.org/get-started/locally/)

:: --- !!! IMPORTANT !!! ---
:: This command installs the latest stable torch/torchvision/torchaudio compatible with CUDA 12.1
:: using the official PyTorch index. Check the PyTorch website link above for other options.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
:: -------------------------

if %errorlevel% neq 0 (
    echo.
    echo !!! IMPORTANT: Failed to install PyTorch CUDA version (using cu121 index) !!!
    echo.
    echo This usually means:
    echo 1. You do not have a compatible NVIDIA GPU or your drivers are too old for CUDA 12.1.
    echo 2. You need a different CUDA version of PyTorch.
    echo 3. An internet or other installation issue occurred.
    echo 4. **Your PyTorch version is too old for some models.** Some models use file formats that require PyTorch v2.6+.
    echo    Even if this install succeeds, if you get model loading errors later related to v2.6,
    echo    it means the PyTorch version installed here wasn't new enough for that specific model/library version.
    echo    Check the PyTorch website for the *absolute latest* compatible version for your system.
    echo.
    echo Error details:
    echo %errorlevel%
    echo.
    echo To proceed, you have two options:
    echo Option A: Troubleshoot your NVIDIA GPU/drivers/CUDA version by checking https://pytorch.org/get-started/locally/ and manually running the correct install command for your system.
    echo Option B: Install the CPU-only version of PyTorch (much slower generation). While in the activated venv (you are now), run:
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
echo PyTorch with CUDA support was successfully installed using the recommended version for CUDA 12.1.
echo If you encounter model loading errors after running run.bat, especially those mentioning
echo "torch.load" or needing "version 2.6+", it means the installed PyTorch version
echo wasn't new enough for that specific model/library combination.
echo You may need to manually install a newer PyTorch version (if available) following instructions at https://pytorch.org/get-started/locally/
echo while the virtual environment is active.
echo.
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
