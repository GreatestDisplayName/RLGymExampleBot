@echo off
setlocal

:: Load common utilities
call "%~dp0common_utils.bat"

:: --- Configuration ---
set SCRIPT_NAME=complete_workflow.py

:: --- Script Body ---
echo ================================================================================
echo 🚀 RLGym Complete Workflow Manager 🚀
echo ================================================================================
echo.
echo This script runs the complete pipeline: Load Map → Train → Export → Load → Play
echo.

:: Environment Check
call :CheckEnvironment

:: Argument Handling
call :HandleArguments "%SCRIPT_NAME%" "" %*

:: Command Execution
python "%SCRIPT_DIR%\%SCRIPT_NAME%" %FINAL_ARGS%

:: Exit Handling
call :ExitHandler

echo.
echo ✅ Workflow completed successfully!
pause
exit /b 0