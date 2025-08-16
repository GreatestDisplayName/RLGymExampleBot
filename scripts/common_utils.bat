@echo off
setlocal

:: --- Common Configuration ---
set VENV_DIR=.venv
set SCRIPT_DIR=..\src

:: --- Function to activate virtual environment and check Python ---
:CheckEnvironment
    if exist "%VENV_DIR%\Scripts\activate.bat" (
        echo Found virtual environment, attempting to activate...
        call "%VENV_DIR%\Scripts\activate.bat"
        echo Virtual environment activated.
    )

    python --version >nul 2>&1
    if errorlevel 1 (
        echo [91m❌ Python is not available in your PATH.[0m
        echo Please ensure Python is installed and accessible, or activate your virtual environment.
        echo For example: call .venv\Scripts\activate
        pause
        exit /b 1
    )
    goto :EOF

:: --- Function to handle arguments and display help ---
:HandleArguments
    set "SCRIPT_TO_RUN=%~1"
    set "DEFAULT_ARGS=%~2"
    shift
    shift
    set "PASSED_ARGS=%*"

    if "%PASSED_ARGS%"=="" (
        echo No arguments provided. Displaying help:
        echo.
        python "%SCRIPT_DIR%\%SCRIPT_TO_RUN%" --help
        echo.
        if not "%DEFAULT_ARGS%"=="" (
            echo Press any key to start with default settings, or close this window to cancel.
            echo (Default: %DEFAULT_ARGS%)
            pause >nul
            echo.
            echo Starting with default parameters...
            set "FINAL_ARGS=%DEFAULT_ARGS%"
        ) else (
            echo Press any key to continue, or close this window to cancel.
            pause >nul
            echo.
            echo Continuing without parameters...
            set "FINAL_ARGS="
        )
    ) else (
        echo Starting with provided parameters: %PASSED_ARGS%
        set "FINAL_ARGS=%PASSED_ARGS%"
    )
    goto :EOF

:: --- Function for Exit Handling ---
:ExitHandler
    if errorlevel 1 (
        echo.
        echo [91m❌ Script failed with error code %errorlevel%.[0m
        pause
        exit /b %errorlevel%
    )
    goto :EOF

endlocal
