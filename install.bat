@echo off
echo ============================================
echo  TALOS - Instalacion de dependencias
echo ============================================
echo.

echo Instalando dependencias principales...
pip install -r requirements.txt
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
    pip install -r requirements-research.txt
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
