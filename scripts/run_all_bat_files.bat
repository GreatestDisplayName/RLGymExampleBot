@echo off

call complete_workflow.bat %*
call gemini.bat %*
call league_manager.bat %*
call run_gemini.bat %*
call start_training.bat %*

echo All .bat files executed.
pause
