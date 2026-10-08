@echo off
chcp 65001 >nul
cd /d "%~dp0"
python mark_ui_plus.py
if errorlevel 1 pause
