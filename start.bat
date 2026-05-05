@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  TALOS · Lanzador local Streamlit
REM  - Activa el venv si existe (.venv\ o venv\).
REM  - Cambia a src\ para que Streamlit cargue .streamlit\config.toml
REM    y los imports planos (session_loader, ui_assets, Circuit2d, ...)
REM    resuelvan sin trampas de PYTHONPATH.
REM
REM  Equivalente Docker (CMD del contenedor):
REM     WORKDIR /app/src
REM     CMD ["streamlit","run","appResearch.py", ^
REM          "--server.port=8501","--server.address=0.0.0.0"]
REM ─────────────────────────────────────────────────────────────────────────

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

cd src

REM Preferimos `streamlit` directo (lo da el venv).
REM Si no está, caemos al lanzador `py` de Windows.
where streamlit >nul 2>nul
if %errorlevel%==0 (
    streamlit run appResearch.py %*
) else (
    py -m streamlit run appResearch.py %*
)

endlocal
