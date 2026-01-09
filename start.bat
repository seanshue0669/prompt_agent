@echo off
setlocal enabledelayedexpansion

set PYTHONUNBUFFERED=1
set LOG_DIR=logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set RUN_TS=%%i
set LOG_FILE=%LOG_DIR%\run_%RUN_TS%.log

echo [%date% %time%] Starting Prompt Agent > "%LOG_FILE%"

echo Prompt Agent Launcher
echo Input prompt:  user\input_prompt.txt
echo Output prompt: user\output_prompt.txt
echo.
echo Logging to %LOG_FILE%
echo.

echo Input prompt:  user\input_prompt.txt>>"%LOG_FILE%"
echo Output prompt: user\output_prompt.txt>>"%LOG_FILE%"

where ollama >nul 2>nul
if errorlevel 1 (
  echo [WARN] Ollama not found in PATH. Start your LLM server manually.
  echo [WARN] Ollama not found in PATH. Start your LLM server manually.>>"%LOG_FILE%"
) else (
  powershell -NoProfile -Command "try {Invoke-RestMethod http://127.0.0.1:11434/api/tags > $null; exit 0} catch { exit 1 }"
  if errorlevel 1 (
    echo Starting Ollama server...
    echo Starting Ollama server...>>"%LOG_FILE%"
    start "" /min cmd /c "set OLLAMA_HOST=0.0.0.0:11434 && ollama serve"
    timeout /t 3 /nobreak >nul
  )
)

for /f %%i in ('powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike '169.254.*' -and $_.InterfaceAlias -notlike '*Loopback*'} | Select-Object -First 1 -ExpandProperty IPAddress)"') do set HOST_IP=%%i

if "%HOST_IP%"=="" (
  echo [WARN] Unable to detect host IPv4. Using localhost.
  echo [WARN] Unable to detect host IPv4. Using localhost.>>"%LOG_FILE%"
  set LLM_BASE_URL=http://127.0.0.1:11434/v1
) else (
  set LLM_BASE_URL=http://%HOST_IP%:11434/v1
)

echo Using LLM_BASE_URL=%LLM_BASE_URL%
echo Using LLM_BASE_URL=%LLM_BASE_URL%>>"%LOG_FILE%"
echo.

powershell -NoProfile -Command "Start-Transcript -Path '%LOG_FILE%' -Append | Out-Null; python main.py; $exitCode=$LASTEXITCODE; Stop-Transcript | Out-Null; exit $exitCode"
if errorlevel 1 (
  echo.
  echo [ERROR] Run failed. See log: %LOG_FILE%
)

endlocal
