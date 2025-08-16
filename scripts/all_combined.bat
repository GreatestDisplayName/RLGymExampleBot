@echo off
setlocal

echo ================================================================================
echo 🚀 RLGym All Combined Workflows 🚀
echo ================================================================================
echo.

call "%~dp0complete_workflow.bat" %*
if errorlevel 1 goto :error_exit

call "%~dp0gemini.bat" %*
if errorlevel 1 goto :error_exit

call "%~dp0league_manager.bat" %*
if errorlevel 1 goto :error_exit

call "%~dp0run_gemini.bat" %*
if errorlevel 1 goto :error_exit

call "%~dp0start_training.bat" %*
if errorlevel 1 goto :error_exit

echo.
echo ✅ All combined workflows completed successfully!
pause
exit /b 0

:error_exit
echo.
echo ❌ An error occurred during one of the workflows.
pause
exit /b %errorlevel%