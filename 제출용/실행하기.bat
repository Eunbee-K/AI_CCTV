@echo off
chcp 949 >nul
setlocal enabledelayedexpansion
title AI 하수관로 CCTV 자동분석 시스템

echo.
echo  ============================================================
echo    AI 하수관로 CCTV 결함 자동분석 시스템
echo  ============================================================
echo.

cd /d "%~dp0"

REM ---- 앱 위치 확인 --------------------------------------------------
set "APPDIR=%~dp0app"
if not exist "%APPDIR%\serve_demo.py" (
  echo  [오류] 앱 폴더를 찾을 수 없습니다.
  echo         찾은 위치: %APPDIR%
  echo.
  echo  압축을 풀 때 [app] 폴더도 함께 풀렸는지 확인하세요.
  echo  압축 파일 안에서 바로 실행하면 이 오류가 납니다.
  echo  반드시 압축을 풀어놓고 실행하세요.
  echo.
  pause
  exit /b 1
)

REM ---- 파이썬 찾기 ---------------------------------------------------
call :FIND_PY

if not defined PY (
  echo  [안내] 이 PC엔 파이썬이 없습니다.
  echo         이 프로그램을 실행하려면 파이썬이 필요합니다.
  echo.

  where winget >nul 2>nul
  if errorlevel 1 (
    echo  [안내] 이 PC에서 winget - Windows 설치 관리자을 찾을 수 없습니다.
    echo         자동 설치를 할 수 없으므로 수동으로 설치하세요.
    echo.
    goto NOPY
  )

  choice /c YN /m "  지금 자동으로 설치할까요? Y=예 / N=아니오"
  if errorlevel 2 goto NOPY
  echo.
  echo  파이썬을 설치합니다. 잠시 걸립니다...
  echo.
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  echo.
  REM 설치 직후에는 이 창의 PATH가 아직 갱신되지 않았다. 표준 설치 경로들과
  REM 함께 다시 찾는다.
  call :FIND_PY
  if not defined PY (
    echo  파이썬 설치는 끝났지만 이 창에서는 아직 인식되지 않습니다.
    echo  이 창을 닫고 [실행하기.bat]를 다시 실행해 주세요.
    echo.
    pause
    exit /b 1
  )
)

echo  [1/3] 파이썬 확인 완료
echo.

REM ---- 모델 파일 확인 ------------------------------------------------
if not exist "%APPDIR%\assets\best.pt" (
  echo  [경고] 판독용 모델 assets\best.pt 가 없습니다.
  echo         분석 실행 시 오류가 납니다.
  echo.
)
if not exist "%APPDIR%\assets\classifier.onnx" (
  echo  [경고] 필터용 모델 assets\classifier.onnx 가 없습니다.
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
REM PY가 "C:\Users\Eunbee Kye\...\python.exe"처럼 공백 포함된 경로일 수도
REM 있으므로 따옴표로 감싸서 호출한다.
"%PY%" %PYARG% setup_env.py
if errorlevel 1 (
  echo.
  echo  ------------------------------------------------------------
  echo   준비가 실패했습니다. 위쪽 [ERROR] 로 시작하는 줄에 원인이 있습니다.
  echo   위쪽에서 [1/3]~[3/3] 중 어느 단계에서 멈췄는지 확인하세요.
  echo   대부분은 인터넷 연결을 확인하고 다시 실행하면 됩니다.
  echo   회사망이라면 방화벽이 pypi.org 를 막고 있을 수 있습니다.
  echo  ------------------------------------------------------------
  echo.
  pause
  exit /b 1
)

REM 준비된 전용 환경의 파이썬으로 바꿔서 실행한다.
set "RUNPY=%APPDIR%\.venv\Scripts\python.exe"
set "RUNARG="
if not exist "%RUNPY%" (
  set "RUNPY=%PY%"
  set "RUNARG=%PYARG%"
)

REM ---- 실행 ----------------------------------------------------------
REM 외부 원격 서버 없이 이 PC에서만 돌아간다.
set "REMOTE_YOLO_URL="
set "AUTH_ENABLED=0"

echo  [3/3] 서버를 시작합니다.
echo.
echo        주소: http://localhost:8080
echo.
echo  ------------------------------------------------------------
echo   종료하려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요.
echo  ------------------------------------------------------------
echo.

"%RUNPY%" %RUNARG% serve_demo.py --port 8080

echo.
echo  프로그램이 종료되었습니다.
pause
exit /b 0


REM ====================================================================
REM  파이썬 실행 파일을 찾아 PY 에 넣는다. 못 찾으면 PY 는 비워 둔다.
REM  주의: 이 아래는 서브루틴이다. 실수로 exit /b 를 쓰면 흐름이 끊긴다.
REM ====================================================================
:FIND_PY
REM PATH 에 등록된 파이썬만 믿는다. 설치 위치를 추측해서 찾던 예전 방식은
REM 사용자마다 설치 경로가 달라 계속 헛짚었다 - Python 설치 마법사 첫 화면의
REM "Add python.exe to PATH"가 기본 체크이므로 거의 모든 PC에서 이것으로 충분하다.
set "PY="
set "PYARG="

REM 실행이 안내만 뜨는 스텁(python.exe)일 수도 있으므로 진짜 인터프리터가
REM 맞는지 확인한다.
python -c "import sys" >nul 2>nul
if not errorlevel 1 (
  set "PY=python"
  goto :eof
)

REM 마지막으로 launcher(py). PY는 실행파일명 "py"만 담고, "-3" 인자는 PYARG에
REM 따로 담는다 - PY는 항상 순수 파일명만 유지해서, 호출하는 쪽이 "%PY%" %PYARG%
REM 형태로 처리해야 공백이 포함된 경로가 섞여서 인수분해되지 않는다.
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  set "PY=py"
  set "PYARG=-3"
)
goto :eof


:NOPY
echo.
echo  아래 주소에서 Python 3.12 이상을 직접 설치한 뒤 다시 실행하세요.
echo.
echo         https://www.python.org/downloads/
echo.
echo  설치 화면 첫 페이지에서 "Add Python to PATH"를 반드시 체크하세요.
echo.
pause
exit /b 1
