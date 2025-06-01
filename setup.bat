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
echo This command installs the latest stable PyTorch version compatible with CUDA 12.1
echo from the official source (https://download.pytorch.org/whl/cu121).
echo Note that PyTorch v2.6 or higher might be required for some models due to recent library updates.
echo If the version installed here is too old, you may see model loading errors later.
echo (See https://pytorch.org/get-started/locally/ for specific commands for your system)

:: --- !!! IMPORTANT !!! ---
:: This command installs the latest stable torch/torchvision/torchaudio compatible with CUDA 12.1.
:: If you know you need PyTorch 2.6 specifically and this command doesn't provide it,
:: you might need a command like: pip install torch==2.6.0 torchvision==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
:: (Confirm actual version numbers and index on the PyTorch website!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
:: -------------------------

if %errorlevel% neq 0 (
    echo.
    echo !!! IMPORTANT: Failed to install PyTorch CUDA version (using cu121 index) during setup !!!
    echo.
    echo This installation step failed. This usually means one of the following:
    echo 1. You do not have a compatible NVIDIA GPU or your drivers are too old for CUDA 12.1.
    echo 2. You need a different CUDA version of PyTorch.
    echo 3. An internet or other package conflict issue occurred during installation.
    echo.
    echo Error details from pip:
    echo %errorlevel%
    echo.
    echo To resolve the setup installation failure:
    echo - Check your internet connection.
    echo - Ensure you have a compatible NVIDIA GPU and the correct drivers installed for CUDA 12.1.
    echo - Visit https://pytorch.org/get-started/locally/ to find the *exact* command for your specific system (OS, Package Manager, CUDA version). You may need a different CUDA index or an older version of PyTorch if your hardware/drivers are older.
    echo.
    echo If you proceed by manually installing PyTorch (either CUDA or CPU) while in the activated venv,
    echo use a command similar to those on the PyTorch website. For CPU-only (slower), use:
    echo pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo.
    echo After successfully installing PyTorch manually or resolving the setup issue, you can try running the app with run.bat.
    goto deactivate_and_end # Still exit setup script on failure
) else (
    echo.
    echo PyTorch CUDA version installed successfully using the recommended latest stable for CUDA 12.1.
    echo.
    echo NOTE: Some newer models or library versions require PyTorch v2.6+. If you encounter model
    echo loading errors after running run.bat, especially those mentioning
    echo "torch.load" or needing "version 2.6+", it means the PyTorch
    echo version installed here was not new enough. In that case, you will need to manually
    echo install PyTorch 2.6+ (if available for your system/CUDA version) by following
    echo instructions at https://pytorch.org/get-started/locally/ while the virtual environment is active.
    echo.
)

echo.
echo --- SETUP COMPLETE ---
echo Environment "%VENV_DIR%" created/activated and dependencies installed.
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
