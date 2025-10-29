@echo off
echo ================================================
echo  APEX DIRECTOR - REAL Video Generation System
echo ================================================
echo.
echo Starting web interface...
echo.
echo IMPORTANT: After starting, visit this URL in your browser:
echo    http://localhost:9000
echo.
echo Your videos will be saved to:
echo    /workspace/outputs/
echo.
echo ================================================
echo.

cd /d "%~dp0"
python simple_apex_director.py

pause