@echo off
echo Setting up the environment for Perchance Revival...

:: Define the virtual environment directory name
set VENV_DIR=venv

:: Change directory to the script's location
cd /d "%~dp0"

:: Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.8+ and ensure it's in your PATH.
    echo Download: https://www.python.org/downloads/windows/
    goto final_pause
)
echo Found Python.

:: Create virtual environment if it doesn't exist
if not exist %VENV_DIR% (
    echo Creating virtual environment "%VENV_DIR%"...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        goto final_pause
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists. Skipping creation.
)

:: Activate the virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment.
    goto final_pause
)
echo Virtual environment activated.

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Set variables
set PYTORCH_INSTALL_STATUS=NOT_ATTEMPTED
set REQ_INSTALL_STATUS=NOT_ATTEMPTED

:: Install PyTorch 2.6.0 with CUDA 12.4
echo.
echo Installing PyTorch 2.6.0 with CUDA 12.4...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

if %errorlevel% neq 0 (
    set PYTORCH_INSTALL_STATUS=FAILED
    echo PyTorch installation failed. Check CUDA compatibility or visit https://pytorch.org/get-started/locally/
    echo Optional CPU-only command: pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
) else (
    set PYTORCH_INSTALL_STATUS=SUCCESS
    echo PyTorch installed successfully.
)

:: Install requirements
echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    set REQ_INSTALL_STATUS=FAILED
    echo Error: Failed to install core dependencies.
) else (
    set REQ_INSTALL_STATUS=SUCCESS
    echo Core dependencies installed successfully.
)

:: Summary
echo.
echo ------- SETUP SUMMARY -------
echo Virtual Environment: %VENV_DIR%
echo PyTorch Install: %PYTORCH_INSTALL_STATUS%
echo Requirements Install: %REQ_INSTALL_STATUS%
echo -----------------------------

:: Deactivate the environment
echo Deactivating virtual environment...
call deactivate

:: Final prompt
:final_pause
echo.
echo Setup complete. Press any key to exit...
pause >nul
