@echo off
setlocal enabledelayedexpansion

:: Set Python version and download URL
set PYTHON_VERSION=3.12.2
set PYTHON_INSTALLER=python-!PYTHON_VERSION!-amd64.exe
set DOWNLOAD_URL=https://www.python.org/ftp/python/!PYTHON_VERSION!/!PYTHON_INSTALLER!

:: Set install path (change as needed)
set INSTALL_PATH=C:\Python!PYTHON_VERSION!

:: Download Python installer
echo Downloading Python !PYTHON_VERSION!...
powershell -Command "(New-Object System.Net.WebClient).DownloadFile('!DOWNLOAD_URL!', '!PYTHON_INSTALLER!')"

:: Install Python silently
echo Installing Python...
start /wait !PYTHON_INSTALLER! /quiet InstallAllUsers=1 PrependPath=1 TargetDir=!INSTALL_PATH!

:: Verify installation
echo Checking Python installation...
!INSTALL_PATH!\python.exe --version

:: Clean up installer
del !PYTHON_INSTALLER!

echo Python installation complete.
pause
