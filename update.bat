@echo off
echo Updating the Perchance Revival application...

:: Define variables
set VENV_DIR=venv
set REPO_URL=https://github.com/Raxephion/perchance-revival-webui
set ZIP_FILE=repo.zip
set TEMP_DIR=temp_extraction

:: Change directory to the script's location
cd /d "%~dp0"

:: Cleanup old temp files if they exist
if exist %ZIP_FILE% del %ZIP_FILE%
if exist %TEMP_DIR% rmdir /s /q %TEMP_DIR%

:: ==================================================================
:: GIT-BASED UPDATE (Preferred Method)
:: ==================================================================
where git >nul 2>nul
if %errorlevel% equ 0 (
    echo Found Git. Using Git to update...
    git pull
    if %errorlevel% neq 0 (
        echo.
        echo ####################################################################
        echo #  GIT PULL FAILED!                                                #
        echo #  This is likely because you have made local changes to files.    #
        echo #  To fix this, you can open a command prompt and run:             #
        echo #  > git stash                                                     #
        echo #  > git pull                                                      #
        echo #  Or, you can delete the folder and re-download.                  #
        echo ####################################################################
        echo.
        goto deactivate_and_end
    )
    echo Code updated successfully via Git.
    goto update_dependencies
)

:: ==================================================================
:: GIT-LESS UPDATE (Fallback Method)
:: ==================================================================
echo.
echo Git not found. Attempting to download latest code directly...
echo.

:: --- DOWNLOAD ATTEMPT 1: Using curl (More Reliable) ---
where curl >nul 2>nul
if %errorlevel% equ 0 (
    echo --- Attempting download with curl...
    curl -L -o %ZIP_FILE% "%REPO_URL%/archive/refs/heads/main.zip"
    if %errorlevel% equ 0 (
        echo Download successful.
        goto extract_zip
    )
    echo curl failed. It might be blocked. Trying PowerShell next...
)

:: --- DOWNLOAD ATTEMPT 2: Using PowerShell (with all compatibility fixes) ---
echo --- Attempting download with PowerShell...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = 'Tls12', 'Tls11', 'Tls'; $webClient = New-Object System.Net.WebClient; $webClient.Proxy = [System.Net.WebRequest]::GetSystemWebProxy(); $webClient.DownloadFile('%REPO_URL%/archive/refs/heads/main.zip', '%ZIP_FILE%')"
if %errorlevel% neq 0 (
    echo.
    echo ####################################################################
    echo #  DOWNLOAD FAILED.                                                #
    echo #  Both curl and PowerShell failed to download the update.         #
    echo #  This is almost certainly caused by a FIREWALL or ANTIVIRUS.     #
    echo #                                                                  #
    echo #  ACTION REQUIRED:                                                #
    echo #  1. Check your Antivirus (Avast, Norton, etc.) for quarantines.  #
    echo #  2. Check Windows Defender Firewall to ensure it isn't blocking  #
    echo #     cmd.exe, powershell.exe, or curl.exe.                         #
    echo #  3. Try the MANUAL download method in the README.                #
    echo ####################################################################
    echo.
    goto end
)
echo Download successful.

:extract_zip
echo Extracting files...
powershell -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%TEMP_DIR%' -Force"
if %errorlevel% neq 0 (
    echo Error: Failed to extract the ZIP file.
    goto cleanup_and_end
)

echo Moving files to the correct directory...
for /d %%d in (%TEMP_DIR%\*) do (
    move /y "%%d\*" "." >nul
)

echo Cleaning up temporary files...
del %ZIP_FILE% 2>nul
rmdir /s /q %TEMP_DIR% 2>nul
echo Successfully downloaded and extracted the latest code.

:update_dependencies
if not exist %VENV_DIR%\Scripts\activate.bat (
    echo Error: Virtual environment not found. Please run setup.bat first.
    goto end
)

if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call %VENV_DIR%\Scripts\activate.bat
)

echo Installing/Upgrading Python packages...
pip install -r requirements.txt --upgrade
if %errorlevel% neq 0 (
    echo Error: Failed to install/upgrade dependencies.
    goto deactivate_and_end
)
echo Dependencies updated successfully.

echo.
echo --- UPDATE COMPLETE ---
echo The application is now up-to-date.
echo.

goto end

:cleanup_and_end
del %ZIP_FILE% 2>nul
rmdir /s /q %TEMP_DIR% 2>nul
goto end

:deactivate_and_end
if defined VIRTUAL_ENV (
    deactivate
)
echo.

:end
echo Press any key to exit...
pause
