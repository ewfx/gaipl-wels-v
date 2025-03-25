@echo off
setlocal

:: Set the Python script filename (change as needed)
set SCRIPT_NAME=code\src\app.py

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    pause
    exit /b
)

:: Run the Python script
echo Running %SCRIPT_NAME%...
python %SCRIPT_NAME%
