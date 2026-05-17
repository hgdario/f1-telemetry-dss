@echo off

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

cd src


where streamlit >nul 2>nul
if %errorlevel%==0 (
    streamlit run app.py %*
) else (
    py -m streamlit run app.py %*
)

endlocal
