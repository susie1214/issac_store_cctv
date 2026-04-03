@echo off
chcp 65001 >nul
title Smart Store

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo venv not found
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist "static\icons\icon-192.png" (
    python create_icons.py
)

python web_app.py

pause
