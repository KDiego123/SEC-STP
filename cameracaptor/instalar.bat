@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv .venv
) else (
    where python >nul 2>nul
    if not %errorlevel%==0 (
        echo No se encontro Python. Instala Python 3 y vuelve a ejecutar este archivo.
        pause
        exit /b 1
    )
    python -m venv .venv
)

if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Instalacion terminada. Ejecuta ejecutar_panel.bat para abrir el panel.
pause
