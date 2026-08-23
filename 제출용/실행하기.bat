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
  echo.
  pause
  exit /b 1
)

REM ---- 파이썬 찾기 ---------------------------------------------------
set "PY="
for %%P in (
  "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe"
  "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
  "C:\Python313\python.exe"
  "C:\Python312\python.exe"
) do (
  if exist %%P set "PY=%%~P"
)
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  py -3 --version >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
  echo  [오류] 파이썬을 찾을 수 없습니다.
  echo         python.org 에서 Python 3.12 이상을 설치한 뒤 다시 실행하세요.
  echo         설치할 때 "Add Python to PATH" 를 반드시 체크하세요.
  echo.
  pause
  exit /b 1
)

echo  [1/3] 파이썬 확인 완료
echo.

REM ---- 모델 파일 확인 ------------------------------------------------
if not exist "%APPDIR%\assets\best.pt" (
  echo  [경고] 판독기 모델 assets\best.pt 이 없습니다.
  echo         분석 실행 시 오류가 납니다.
  echo.
)
if not exist "%APPDIR%\assets\classifier.onnx" (
  echo  [경고] 분류기 모델 assets\classifier.onnx 이 없습니다.
  echo         분석 실행 시 오류가 납니다.
  echo.
)

echo  [2/3] 서버를 시작합니다. 잠시만 기다려 주세요...
echo.

cd /d "%APPDIR%"

REM 외부 통신 없이 이 PC에서만 추론한다
set "REMOTE_YOLO_URL="
set "AUTH_ENABLED=0"
REM 이 창은 CP949다. 파이썬 기본 출력은 UTF-8이라 한글이 깨진다.
set "PYTHONIOENCODING=cp949"

echo  [3/3] 브라우저가 자동으로 열립니다.
echo.
echo        주소: http://localhost:8080
echo.
echo  ------------------------------------------------------------
echo   종료하려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요.
echo  ------------------------------------------------------------
echo.

%PY% serve_demo.py --port 8080

echo.
echo  서버가 종료되었습니다.
pause
