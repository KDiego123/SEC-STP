@echo off
setlocal
cd /d "%~dp0"

rem IP de respaldo: el panel busca primero la MAC por ONVIF en todas las redes.
set "CAMERA_HOST=192.168.1.51"
if not "%~1"=="" set "CAMERA_HOST=%~1"

if not exist ".venv\Scripts\python.exe" (
    echo Falta el entorno de Python. Ejecuta instalar.bat primero.
    pause
    exit /b 1
)

rem Evita que los hilos de inferencia consuman CPU mientras esperan otro frame.
if not defined KMP_BLOCKTIME set "KMP_BLOCKTIME=0"
if not defined OMP_WAIT_POLICY set "OMP_WAIT_POLICY=PASSIVE"

".venv\Scripts\python.exe" ".\control_panel.py" ^
    --credentials-file ".\claves.txt" ^
    --host "%CAMERA_HOST%" ^
    --faces-dir ".\face_gallery"
