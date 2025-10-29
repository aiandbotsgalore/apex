@echo off
echo.
echo ===========================================
echo    APEX DIRECTOR - Music Video Generator
echo ===========================================
echo.

REM Install dependencies
echo Installing required packages...
pip install fastapi uvicorn python-multipart jinja2

if errorlevel 1 (
    echo.
    echo ❌ Failed to install dependencies
    echo Please make sure you have Python and pip installed
    pause
    exit /b 1
)

REM Create necessary directories
echo Creating directories...
if not exist "uploads" mkdir uploads
if not exist "output" mkdir output

echo.
echo 🚀 Starting APEX DIRECTOR web interface...
echo.
echo 📱 Open your browser and go to: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo ===========================================
echo.

REM Start the web interface
python apex_director_web_interface.py

pause