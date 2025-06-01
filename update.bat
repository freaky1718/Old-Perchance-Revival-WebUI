@echo off
echo Updating the Perchance Revival application...

:: Define the virtual environment directory name (must match setup.bat and run.bat)
set VENV_DIR=venv

:: Change directory to the script's location
cd /d "%~dp0"

:: Check if Git is available
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Git not found. Please install Git and make sure it's added to your system's PATH.
    echo You can download Git from https://git-scm.com/
    goto end
)
echo Found Git.

:: Check if the virtual environment exists
if not exist %VENV_DIR%\Scripts\activate.bat (
    echo Error: Virtual environment "%VENV_DIR%" not found.
    echo Please run setup.bat first to create the environment.
    goto end
)

:: Activate the virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment. Check virtual environment setup.
    goto end
)
echo Virtual environment activated.

:: Pull latest code from the repository
echo Pulling latest code from GitHub...
git pull
if %errorlevel% neq 0 (
    echo Error: Failed to pull latest code. This could be due to local changes, network issues, or Git configuration.
    echo Please resolve the Git issue manually or discard local changes if necessary.
    goto deactivate_and_end
)
echo Code updated successfully.

:: Install/Upgrade dependencies from requirements.txt
echo Installing/Upgrading dependencies from requirements.txt...
pip install -r requirements.txt --upgrade
if %errorlevel% neq 0 (
    echo Error: Failed to install/upgrade dependencies. See the output above for details.
    echo This could be due to network issues or conflicts between packages.
    goto deactivate_and_end
)
echo Dependencies updated successfully.

echo.
echo --- UPDATE COMPLETE ---
echo The application code and dependencies (from requirements.txt) are now up-to-date.
echo.
echo Important Note on PyTorch (GPU/CPU):
echo This update does NOT change your current PyTorch installation (whether CPU or CUDA).
echo If you manually installed the CUDA version and it was successful, it should still work.
echo If you encounter issues with GPU acceleration after updating, or if you initially
echo installed the CPU version and now want to try the GPU version, you will need to
echo manually run the appropriate PyTorch installation command again while inside the
echo activated virtual environment. Refer to the setup.bat instructions or the README
echo for the specific commands.
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
