@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo  TALOS - Instalacion de dependencias
echo ============================================
echo.

REM --- Detectar Python (py launcher o python) ---
set "PYTHON="
where py >nul 2>nul && set "PYTHON=py"
if not defined PYTHON ( where python >nul 2>nul && set "PYTHON=python" )
if not defined PYTHON (
    echo ERROR: No se ha encontrado Python.
    echo Instala Python 3.10 o superior desde https://www.python.org/downloads/
    echo y marca la opcion "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)
echo Python detectado: %PYTHON%
echo.

REM --- Crear el entorno virtual (.venv) si no existe ---
if not exist ".venv\Scripts\activate.bat" (
    echo Creando entorno virtual aislado ".venv" ...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
) else (
    echo Entorno virtual ".venv" ya existe, se reutiliza.
)
echo.

REM --- Activar el entorno virtual ---
call ".venv\Scripts\activate.bat"

REM --- Instalar dependencias dentro del entorno ---
echo Actualizando pip e instalando dependencias principales...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Fallo al instalar dependencias principales.
    pause
    exit /b 1
)

echo.
echo Dependencias principales instaladas correctamente.
echo.
set /p research="Instalar dependencias de investigacion (re-entrenamiento de clasificadores)? [s/n]: "
if /i "%research%"=="s" (
    echo.
    echo Instalando dependencias de investigacion...
    python -m pip install -r requirements-research.txt
    if errorlevel 1 (
        echo ERROR: Fallo al instalar dependencias de investigacion.
        pause
        exit /b 1
    )
    echo Dependencias de investigacion instaladas correctamente.
)

echo.
echo ============================================
echo  Listo. Ejecuta: start.bat
echo ============================================
pause
