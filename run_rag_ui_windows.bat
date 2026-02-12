@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =========================================================
REM RAG Studio Pro - Lanzador Windows (doble clic)
REM =========================================================

cd /d "%~dp0"

echo.
echo [RAG Studio Pro] Iniciando en carpeta:
echo %CD%
echo.

REM 1) Resolver Python (py o python)
set "PY_CMD="
where py >nul 2>&1
if %errorlevel%==0 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>&1
  if %errorlevel%==0 (
    set "PY_CMD=python"
  )
)

if "%PY_CMD%"=="" (
  echo [ERROR] No se encontro Python en PATH.
  echo Instala Python 3.10+ desde https://www.python.org/downloads/
  pause
  exit /b 1
)

echo [INFO] Python detectado: %PY_CMD%

REM 2) Activar entorno virtual si existe
if exist ".venv\Scripts\activate.bat" (
  echo [INFO] Activando entorno virtual .venv...
  call ".venv\Scripts\activate.bat"
)

REM 3) Verificar archivos principales
if not exist "web_ui.py" (
  echo [ERROR] No se encontro web_ui.py en esta carpeta.
  pause
  exit /b 1
)

if not exist "requirements.txt" (
  echo [WARNING] No se encontro requirements.txt. Continuando sin instalar dependencias.
) else (
  REM 4) Asegurar pip actualizado e instalar requirements
  echo [INFO] Actualizando pip...
  %PY_CMD% -m pip install --upgrade pip
  if not %errorlevel%==0 (
    echo [ERROR] No se pudo actualizar pip.
    pause
    exit /b 1
  )

  echo [INFO] Instalando dependencias desde requirements.txt...
  %PY_CMD% -m pip install -r requirements.txt
  if not %errorlevel%==0 (
    echo [ERROR] Fallo instalando dependencias.
    echo Revisa tu conexion, permisos o version de Python.
    pause
    exit /b 1
  )
)

REM 5) Preflight opcional
if exist "preflight_check.py" (
  echo [INFO] Ejecutando preflight...
  %PY_CMD% preflight_check.py
  if not %errorlevel%==0 (
    echo [WARNING] Preflight detecto advertencias. Se intentara iniciar igualmente.
  )
)

REM 6) Lanzar aplicación local

echo [INFO] Iniciando servidor local...
echo [INFO] URL: http://127.0.0.1:7860
echo [INFO] Presiona Ctrl + C para detener.
echo.

%PY_CMD% web_ui.py
set "EXIT_CODE=%errorlevel%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo [ERROR] El servidor termino con codigo %EXIT_CODE%.
  echo Revisa dependencias ^(bs4/langchain/chromadb/ollama^) y que Ollama este activo.
) else (
  echo [OK] Servidor finalizado correctamente.
)

echo.
pause
exit /b %EXIT_CODE%
