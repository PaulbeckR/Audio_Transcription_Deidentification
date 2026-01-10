@echo off
REM GPU Baseline Test Runner
cd /d "%~dp0"
.venv\Scripts\python.exe run_baseline_test.py
pause
