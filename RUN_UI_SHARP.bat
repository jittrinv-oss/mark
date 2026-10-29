@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 mark_ui_sharp.py 2>nul || python mark_ui_sharp.py
if errorlevel 1 pause
