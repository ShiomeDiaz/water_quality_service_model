@echo off
REM Liberar el puerto 3035 si está ocupado
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3035') do (
    echo Matando proceso en puerto 3035 con PID %%a
    taskkill /PID %%a /F
)

REM Crear entorno virtual si no existe
IF NOT EXIST venv (
    python -m venv venv
)

REM Activar el entorno virtual
call venv\Scripts\activate

REM Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

REM Ejecutar el script Python
python main.py

REM Dejar la ventana abierta para ver errores
pause
