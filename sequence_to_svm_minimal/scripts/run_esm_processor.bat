@echo off
REM Easy wrapper for ESM sequence processing
REM Usage: run_esm_processor.bat <input_file> <output_path> <mode>

echo.
echo Activating esm_env...
call %USERPROFILE%\miniconda3\condabin\conda.bat activate esm_env

if "%1"=="" (
    echo.
    echo Usage: run_esm_processor.bat ^<input_file^> ^<output_path^> ^<mode^>
    echo.
    echo Modes:
    echo   embeddings - Extract ESM-2 embeddings only
    echo   fold       - Predict structures with ESMFold only
    echo   both       - Do both
    echo.
    echo Example:
    echo   run_esm_processor.bat experiments\exp1\raw.txt results\esm_output embeddings
    echo.
    pause
    exit /b 1
)

set INPUT=%1
set OUTPUT=%2
set MODE=%3

if "%MODE%"=="" set MODE=embeddings

echo.
echo Running ESM processor...
echo   Input:  %INPUT%
echo   Output: %OUTPUT%
echo   Mode:   %MODE%
echo.

python models\esm_sequence_processor.py --input "%INPUT%" --output "%OUTPUT%" --mode %MODE%

echo.
echo Done!
pause
