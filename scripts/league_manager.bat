@echo off
setlocal

:: Load common utilities
call "%~dp0common_utils.bat"

:: --- Configuration ---
set SCRIPT_NAME=league_manager.py

:: --- Script Body ---
echo ================================================================================
echo 🚀 RLGym Enhanced Self-Play League Manager 🚀
echo ================================================================================
echo.

:: Environment Check
call :CheckEnvironment

:: Argument Handling
call :HandleArguments "%SCRIPT_NAME%" "" %*

:: Command Execution
python -u "%SCRIPT_DIR%\%SCRIPT_NAME%" %FINAL_ARGS%

:: Exit Handling
call :ExitHandler

echo.
echo ✅ Command finished successfully.
pause
exit /b 0