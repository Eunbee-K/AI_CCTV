# 웹 데모 핸드오프 (2026-07-30 세션)

다른 PC(노트북)에서 이어서 작업하기 위한 인수인계 문서. 이 세션에서 **OCR 버그 3종
수정 + 오버레이 자막 오탐 필터 + exe→브라우저 웹 데모(로그인)**를 붙였다.

---

## 1. 이 세션에서 바뀐 것

### (A) OCR 버그 3종 — `backend/ocr.py`
1. **한글 인식 복구.** `PaddleOCR(...)`에 `text_recognition_model_name="korean_PP-OCRv5_mobile_rec"`
   를 **명시**하도록 변경.
   - 원인: PaddleOCR 3.x는 `text_detection_model_name`을 지정하면 `lang="korean"`을
     **무시**하고 기본(비한글) 인식모델을 로드한다. 그래서 현장명/`조사시작` 등 한글이
     전부 빈 문자열로 나왔다. (앱이 조용히 깨져 있던 상태)
2. **거리(距離) 판독.** 거리 영역을 화면 중앙의 속도표시 전(`x: 0→0.30`)까지 좁히고,
   `_parse_distance_from_text`에서 속도(`x.x m/min`) 토큰 제거 + 첫 매치 사용.
   - 증상: 카메라가 멈춰 있으면 속도 0.0이 거리로 찍혀 `0.0m`, 이동 중엔 속도값이 거리로 찍힘.
   - 결과: 실제 프레임 37장 기준 정확도 **35% → 94%**.
3. **관경(직경) 숫자.** `ocr_overlay_metadata`에 우측 재질라인 영역(`right_text`)을 추가하고,
   `_parse_diameter_from_text`에서 날짜/시간 제거 후 재질(PVC 등) 앞 3자리를 뽑도록 변경.
   - 증상: 코너마커 `JD 54%`의 "54"를 관경으로 오인(Φ54).
   - 결과: 37장에서 관경 `200` 추출 **100%**. (Φ 기호 자체는 안 읽어도 숫자만 정확히)

### (B) 오버레이 자막 오탐 필터 — `backend/overlay_filter.py` (신규)
- 화면에 박힌 자막(특히 우하단 **위도/경도** 숫자)이 파손(BK)으로 오탐되던 문제.
- 탐지 박스가 고정된 자막영역에 **70% 이상** 들어가면 결함에서 제외.
- `yolo_infer.call_yolo`(로컬)·`yolo_remote.call_yolo_remote`(Colab) 두 경로 모두 적용.
- 검증: 실제 `call_yolo`로 위경도 오탐 6개 전부 제거, 실제 결함은 유지.

### (C) exe → 브라우저 웹 데모 (로그인)
- **`backend/auth.py`** (신규): 로그인 게이트. `AUTH_ENABLED=1`일 때만 동작(웹 전용).
  승인 계정은 환경변수 `AUTH_USERS="아이디:비번,아이디2:비번2"`로 관리.
  ※ 데모용 — 비번 평문 비교. 상용화 시 bcrypt+DB로 교체할 것.
- **`ui/login.html`** (신규): 로그인 화면.
- **`serve_web.py`** (신규): 헤드리스 웹 실행 진입점(`0.0.0.0`, 로그인 ON).
- **`ui/index.html` + `ui/js/app.js` + `api.js` + `css/styles.css`**: "추론 서버(Colab)"
  연결 패널(노트북 링크 + ngrok 주소 입력) + 로그인 사용자 바 + 로그아웃.
- **`backend/server.py`**: SessionMiddleware + 인증 게이트 등록.
- **`main.py`(기존 exe/pywebview)는 한 줄도 안 바뀜.** 로컬 데스크톱용으로 그대로 유지.
  로그인은 `serve_web.py`(웹)에서만 켜진다.

---

## 2. 실행 방법

의존성(한 venv에 설치). **추론을 Colab으로 돌리면 torch/ultralytics 불필요** → 웹서버 가벼움:
```bash
pip install -r apps/AI_CCTV/requirements.txt
# 최소: fastapi uvicorn paddleocr paddlepaddle opencv-python pillow openpyxl websockets itsdangerous requests
```

```bash
cd apps/AI_CCTV

# 1) 기존 데스크톱(exe) — 로그인 없음
python main.py

# 2) 웹 버전 — 로그인 있음
AUTH_USERS="admin:원하는비번" python serve_web.py --port 8000
#  접속(같은 네트워크):  http://<이 PC의 IP>:8000     (IP 확인: hostname -I / ipconfig)
#  접속(이 PC):          http://localhost:8000
#  기본 계정(미지정 시):  admin / admin1234
```
- **추론 연결**: 앱 화면의 "추론 서버(Colab)" 패널에 Colab에서 나온 ngrok 주소를 넣고 [연결].
- **외부(사무실 밖) 접속**: 터널 필요 →
  `cloudflared tunnel --url http://localhost:8000` (실행 시 공개 https 주소 자동 생성).

### 채워야 할 곳
- `ui/index.html` 의 `<a id="colabOpen" href="">` — 본인 Colab 노트북 공유 링크 (지금 공란, 주석으로 위치 표시).

---

## 3. 결정 사항 (배경)
- **OCR 엔진**: PaddleOCR 유지. RapidOCR와 비교했으나 같은 PP-OCRv5 모델이라 정확도 동급,
  차이는 배포 무게뿐. 지금은 Paddle 유지(결정 변경 시 재검토).
- **추론 위치**: 데모는 **Colab 무료**. 제약 — 이 워크스테이션은 **학습 전용(추론 불가)**,
  실사용 클라이언트는 **저사양** → 추론 오프로드 필수. 상용화 때 정식 GPU로.
- **범위**: 지금은 **윗선 시연용 데모**(된다는 걸 보여주기). 상용화 아님.

## 4. 다음 할 일 (TODO)
- [ ] `index.html` Colab `href` 링크 채우기
- [ ] 외부 접속용 터널(cloudflared/ngrok) 설정 — 워크스테이션엔 미설치였음
- [ ] 영상 **업로드** 엔드포인트 (현재 "파일추가"는 서버-로컬 경로 prompt 방식)
- [ ] 엑셀 **다운로드** (현재 저장경로 지정 방식)
- [ ] (상용) **사용자별 상태 격리** — `state.py`가 싱글톤이라 동시 접속 시 결과 섞임
- [ ] (상용) 구글 OAuth(Authlib) + 이메일 화이트리스트

## 5. 이 워크스테이션에만 있던 것 (git 아님 → 노트북에서 재생성)
- 격리 venv (fastapi+paddleocr 등): `requirements.txt`로 재설치.
- **시연용 프리로드 데모**(결과 미리 채운 화면): `CP-C-17.mp4`로 워크스테이션(GPU)에서 생성.
  노트북은 GPU가 없어 재생성 불가 → 노트북에선 `serve_web.py` + Colab 라이브로 시연하거나,
  워크스테이션에서 만든 결과물을 복사해서 사용.
- 변환 ONNX 모델(korean_rec/v5_det): RapidOCR 비교용이었고 **앱은 사용 안 함**(무시).
