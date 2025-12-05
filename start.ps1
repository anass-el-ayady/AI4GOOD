# PowerShell startup script
Write-Host "Starting AI Learning Lab..." -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Yellow
Write-Host "The app will be available at: http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

python app.py
