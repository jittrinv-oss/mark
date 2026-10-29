@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 web_sharp.py 2>nul || python web_sharp.py
pause
