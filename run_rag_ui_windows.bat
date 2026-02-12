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

REM 2) Activar entorno virtual si existe
if exist ".venv\Scripts\activate.bat" (
  echo [INFO] Activando entorno virtual .venv...
  call ".venv\Scripts\activate.bat"
)

REM 3) Verificar archivo principal
if not exist "web_ui.py" (
  echo [ERROR] No se encontro web_ui.py en esta carpeta.
  pause
  exit /b 1
)

REM 4) Recordatorio de dependencias (opcional)
echo [INFO] Si es la primera vez, instala dependencias con:
echo        pip install -U beautifulsoup4 langchain-community langchain-ollama langchain-text-splitters chromadb pypdf

echo [INFO] Iniciando servidor local...
echo [INFO] URL: http://127.0.0.1:7860
echo [INFO] Presiona Ctrl + C para detener.
echo.

%PY_CMD% web_ui.py
set "EXIT_CODE=%errorlevel%"

echo.
if not "%EXIT_CODE%"=="0" (
  echo [ERROR] El servidor termino con codigo %EXIT_CODE%.
  echo Revisa dependencias (bs4/langchain/chromadb/ollama) y que Ollama este activo.
) else (
  echo [OK] Servidor finalizado correctamente.
)

echo.
pause
exit /b %EXIT_CODE%
