@echo off
chcp 949 >nul
setlocal enabledelayedexpansion
title AI 하수관로 CCTV 자동분석 시스템

echo.
echo  ============================================================
echo    AI 기반 하수관로 CCTV 영상 자동분석 시스템
echo  ============================================================
echo.

cd /d "%~dp0"

REM ---- 앱 위치 확인 --------------------------------------------------
set "APPDIR=%~dp0app"
if not exist "%APPDIR%\serve_demo.py" (
  echo  [오류] 앱 폴더를 찾을 수 없습니다.
  echo         찾은 경로: %APPDIR%
  echo.
  echo  압축을 풀 때 [app] 폴더가 함께 풀렸는지 확인하세요.
  echo  압축 파일 안에서 바로 실행하면 이 오류가 납니다.
  echo  반드시 폴더에 풀어놓고 실행하세요.
  echo.
  pause
  exit /b 1
)

REM ---- 파이썬 찾기 ---------------------------------------------------
call :FIND_PY

if not defined PY (
  echo  [안내] 이 PC에 파이썬이 없습니다.
  echo         이 프로그램을 돌리려면 파이썬이 필요합니다.
  echo.
  choice /c YN /m "  지금 자동으로 설치할까요? (Y=예 / N=아니오)"
  if errorlevel 2 goto NOPY
  echo.
  echo  파이썬을 설치합니다. 몇 분 걸립니다...
  echo.
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  echo.
  REM 설치 직후에는 이 창의 PATH가 아직 갱신되지 않았다. 표준 설치 경로까지
  REM 함께 뒤진다.
  call :FIND_PY
  if not defined PY (
    echo  파이썬 설치는 끝났지만 이 창에서는 아직 인식되지 않습니다.
    echo  이 창을 닫고 [실행하기.bat]을 다시 실행해 주세요.
    echo.
    pause
    exit /b 1
  )
)

echo  [1/3] 파이썬 확인 완료
echo.

REM ---- 모델 파일 확인 ------------------------------------------------
if not exist "%APPDIR%\assets\best.pt" (
  echo  [경고] 판독기 모델 assets\best.pt 가 없습니다.
  echo         분석 실행 시 오류가 납니다.
  echo.
)
if not exist "%APPDIR%\assets\classifier.onnx" (
  echo  [경고] 분류기 모델 assets\classifier.onnx 가 없습니다.
  echo         분석 실행 시 오류가 납니다.
  echo.
)

cd /d "%APPDIR%"

REM ---- 필요한 패키지 준비 (처음 한 번만) -----------------------------
echo  [2/3] 필요한 프로그램을 준비합니다.
echo.
echo        ** 처음 실행할 때만 ** 인터넷에서 약 1GB를 내려받습니다.
echo        회선에 따라 5~15분 걸리며, 다음 실행부터는 바로 시작합니다.
echo        중간에 창을 닫지 마세요.
echo.

set "PYTHONIOENCODING=cp949"
%PY% setup_env.py
if errorlevel 1 (
  echo.
  echo  ------------------------------------------------------------
  echo   준비에 실패했습니다. 인터넷 연결을 확인한 뒤 다시 실행하세요.
  echo   회사망이라면 방화벽이 pypi.org 를 막고 있을 수 있습니다.
  echo  ------------------------------------------------------------
  echo.
  pause
  exit /b 1
)

REM 준비된 전용 환경의 파이썬으로 바꿔 실행한다.
set "RUNPY=%APPDIR%\.venv\Scripts\python.exe"
if not exist "%RUNPY%" set "RUNPY=%PY%"

REM ---- 실행 ----------------------------------------------------------
REM 외부 통신 관련 값은 이 PC에서만 쓰도록 비운다.
set "REMOTE_YOLO_URL="
set "AUTH_ENABLED=0"

echo  [3/3] 브라우저를 자동으로 엽니다.
echo.
echo        주소: http://localhost:8080
echo.
echo  ------------------------------------------------------------
echo   종료하려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요.
echo  ------------------------------------------------------------
echo.

"%RUNPY%" serve_demo.py --port 8080

echo.
echo  프로그램이 종료되었습니다.
pause
exit /b 0


REM ====================================================================
REM  파이썬 실행 파일을 찾아 PY 에 넣는다. 못 찾으면 PY 는 비어 있다.
REM  주의: 이 아래는 서브루틴이다. 위에서 exit /b 로 끊어야 흘러들지 않는다.
REM ====================================================================
:FIND_PY
set "PY="
for %%P in (
  "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe"
  "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
  "C:\Program Files\Python313\python.exe"
  "C:\Program Files\Python312\python.exe"
  "C:\Python313\python.exe"
  "C:\Python312\python.exe"
) do (
  if exist %%P if not defined PY set "PY=%%~P"
)
if defined PY goto :eof

REM PATH 에 있는 python. 스토어 안내용 껍데기(python.exe)는 실제로는 실행이
REM 안 되므로, 찾았다고 끝내지 말고 진짜 도는지 확인한다.
python -c "import sys" >nul 2>nul
if not errorlevel 1 (
  set "PY=python"
  goto :eof
)

REM 마지막으로 런처(py).
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
goto :eof


:NOPY
echo.
echo  아래 주소에서 Python 3.12 이상을 직접 설치한 뒤 다시 실행하세요.
echo.
echo         https://www.python.org/downloads/
echo.
echo  설치 화면 첫 페이지에서 "Add Python to PATH" 를 반드시 체크하세요.
echo.
pause
exit /b 1
