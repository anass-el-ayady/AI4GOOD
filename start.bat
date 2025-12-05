@echo off
echo Starting AI Learning Lab...
echo.
cd /d "%~dp0"
call venv\Scripts\activate.bat
echo Virtual environment activated!
echo.
echo Starting Flask server...
echo The app will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py
pause
