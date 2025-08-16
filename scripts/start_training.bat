@echo off
setlocal

:: Load common utilities
call "%~dp0common_utils.bat"

:: --- Configuration ---
set SCRIPT_NAME=launch_training.py
set DEFAULT_ARGS=--algorithm PPO --timesteps 100000 --test --convert

:: --- Script Body ---
echo ================================================================================
echo 🚀 RLGym Training Launcher 🚀
echo ================================================================================
echo.

:: Environment Check
call :CheckEnvironment

:: Argument Handling
call :HandleArguments "%SCRIPT_NAME%" "%DEFAULT_ARGS%" %*

:: Command Execution
python "%SCRIPT_DIR%\%SCRIPT_NAME%" %FINAL_ARGS%

:: Exit Handling
call :ExitHandler

echo.
echo ✅ Training completed successfully!
pause
exit /b 0