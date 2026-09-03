@echo off
REM Double-click this to set up Jarvis.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.13 install.py 2>nul || py -3 install.py
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python install.py
    ) else (
        echo No Python found. Install it from python.org ^(pick 3.13^),
        echo making sure "Add python.exe to PATH" is ticked, then run this again.
    )
)
echo.
pause
