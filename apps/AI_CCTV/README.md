# AI_CCTV

하수관로 CCTV 영상에서 YOLO(`best.pt`)로 결함을 자동 탐지하고, 편집 가능한 결과표와 엑셀 보고서를 생성하는 데스크톱 앱.

기존 tkinter 버전(`C:\Users\SAMSUNG\Desktop\CCTV\ver3.1_AI_CCTV_YOLO`)을 pywebview 데스크톱 창 + HTML/JS 프론트엔드 + FastAPI 백엔드 구조로 이식한 버전. GPT/Gemini 멀티모달 분석은 실제 파이프라인에서 쓰이지 않던 기능이라 제외했다.

## 구조

- `main.py` : uvicorn 서버를 백그라운드 스레드로 띄우고 pywebview 창을 연다.
- `backend/` : FastAPI 라우트 + 분석 파이프라인(PaddleOCR 구간인식/메타데이터, 프레임 추출, YOLO 추론, 결과 그룹핑, 엑셀 출력).
- `ui/` : vanilla JS(ES 모듈) 프론트엔드. 빌드 스텝 없음.
- `assets/best.pt` : YOLO 가중치 (git에는 포함되지 않음, `.gitignore`의 `*.pt` 규칙).

## 실행

```
pip install -r requirements.txt
python main.py
```

### 제출/시연용 (설치까지 자동)

`setup_env.py`가 전용 가상환경(`.venv`)을 만들고 필요한 패키지를 알아서 깐다.
받는 PC의 파이썬 환경을 건드리지 않는 것이 목적이라, 개발 중에는 위의
`pip install`을 그대로 쓰면 된다.

```
python setup_env.py     # 처음 한 번만 (약 1GB, 5~15분)
python serve_demo.py    # 로그인 없는 로컬 데모
```

torch는 `--index-url .../whl/cpu`로 **CPU 판**을 받는다. 그냥 받으면 CUDA가
딸린 2.5GB짜리가 와서, CPU로만 도는 이 앱에는 순수 낭비다.
설치 묶음을 고칠 때는 `setup_env.py`의 `SETUP_VERSION`을 올려야 이미 깔린
PC에서도 다시 점검한다.

## 주요 기능

- 영상 큐 추가/삭제, 미리보기(재생/일시정지/탐색, 1x/1.5x/2x/2.5x 배속) — OpenCV로 프레임을 읽어 JPEG/MJPEG로 서빙 (avi/mkv 등 브라우저가 직접 재생 못 하는 코덱도 동일하게 지원).
- "분석 실행": PaddleOCR로 조사 시작 구간 자동 인식 → 2초 간격 프레임 추출 → YOLO 탐지(로컬 CPU 또는 Colab 원격, 기본은 ngrok 고정 도메인) → 프레임 OCR로 거리/관경/현장명 보정 → 동일 거리 중복 시 "확인필요" 표시.
- 탐지 시각화: 재생 중에도 해당 시점의 bbox+결함명+신뢰도%가 실시간 오버레이로 표시. 미리보기 휠 확대/드래그 이동/더블클릭 초기화.
- 결과표 탭: 영상별 구분선, 거리 중복 그룹 접기/펼치기, 더블클릭 인라인 편집, 행 클릭 또는 키보드 ↑↓로 탐지 프레임 이동, 신뢰도 임계값 슬라이더·결함 종류(클래스) 필터, "오탐" 체크(보고서/통계에서 제외).
- 통계 탭: 영상별 × 결함 종류별 검출 수 집계 (오탐 제외).
- 수동 행 추가/삭제, "보고서 출력"으로 bbox가 그려진 프레임 이미지가 포함된 xlsx 생성.

## 참고

- 영상/보고서 파일 선택은 pywebview 네이티브 다이얼로그(`window.pywebview.api.open_video_dialog / save_excel_dialog`)를 사용한다. 브라우저에서 pywebview 없이 열면 경로를 직접 입력하는 폴백(prompt)으로 동작한다 (개발/디버깅용).
- YOLO 설정값(`YOLO_CONF`, `YOLO_IMGSZ`, `YOLO_MODEL_PATH` 등)은 환경변수로 덮어쓸 수 있다 (`backend/config.py` 참고).
