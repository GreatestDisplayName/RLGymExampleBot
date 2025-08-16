@echo off
setlocal

rem --- Configuration ---
set VENV_DIR=.venv

rem --- Activate virtual environment if it exists ---
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    echo Virtual environment not found. Assuming Python is in PATH.
)

rem --- Run the GUI application ---
start "" "%VENV_DIR%\Scripts\python.exe" gui.py

endlocal