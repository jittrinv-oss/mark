@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 web_app.py 2>nul || python web_app.py
pause
