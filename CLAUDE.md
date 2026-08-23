# CLAUDE.md — AI_CCTV_PROJECT 컨텍스트

이 파일은 Claude Code가 이 프로젝트 폴더에서 작업을 시작할 때 자동으로 읽는 컨텍스트 파일입니다.
이 저장소는 두 곳에 존재합니다:

1. **외장 SSD/HDD(이동식 드라이브)**: 드라이브 문자는 연결한 PC마다 다를 수 있습니다(원래 PC에서는
   `E:\`) — 작업 전에 실제 드라이브 문자를 먼저 확인하세요. 다른 PC로 옮겨 다닐 때 쓰는 원본/사본.
2. **워크스테이션 로컬 고정 경로** (`/home/workstation/ai_cctv/`): 2026-07-29 세션에서 이 워크스테이션
   전용으로 통째로 복사해둔 사본. `AI_CCTV_DATASET`(원본 데이터셋)·`AI_CCTV_RESULTS`(옛 수동 실험
   결과)도 같은 위치에 로컬 복사해뒀고, `configs/sweeps/*.yaml`은 전부 이 로컬 경로를 가리키도록
   맞춰져 있어서 외장하드를 뽑아도 스윕 학습이 그대로 돈다. 자세한 폴더 구성은 아래 "최근 작업
   로그" 참고.

## 드라이브 안 폴더 구조 (외장하드 루트 기준)

- `AI_CCTV_PROJECT\` — 이 저장소 (git 관리). 앱 코드, 학습 설정/실험 메타데이터, 문서.
- `AI_CCTV_DATASET\` — 실제 데이터셋 (git 미관리, 용량 큼).
- `AI_CCTV\` — 독립 실행되는 라벨 변환 GUI 도구 (`aihub_yolo_converter_gui.py`, `convert_aihub_to_yolo.py`). AIHub JSON/JSONL 라벨을 YOLO txt로 변환할 때 사용.

## 워크스테이션 로컬 폴더 구조 (`/home/workstation/ai_cctv/` 기준)

- `AI_CCTV_PROJECT/` — 이 저장소의 로컬 사본 (실제 작업 위치).
- `AI_CCTV_DATASET/original/` — 외장하드 `AI_CCTV_DATASET/original` 전체 로컬 복사 (~424GB).
- `AI_CCTV_RESULTS/` — 외장하드 `AI_CCTV_RESULTS`(옛 수동 실험 결과) 로컬 복사.
- `pretrained_models/` — `yolo11l.pt`/`yolo11m.pt`/`yolo11n.pt`/`yolo26n.pt`/`yolo11l-seg.pt`.
- `_sweep_workspace/` — `run_sweep.py`가 실행 중 자동 생성하는 임시 데이터셋/결과 폴더.
- `legacy/` — 스윕 자동화 이전 수동 실험 흔적 (`260629_test1` 등). 참고용으로만 보존.

## 프로젝트가 무엇인지

하수관로 CCTV 영상에서 YOLO로 결함을 자동 탐지하고 엑셀 조사표를 만드는 데스크톱 앱
(`apps/AI_CCTV`)과, 이를 위한 학습 데이터/모델 파이프라인. 전체 요구사항/현황/로드맵은
**`docs/PRD.md`를 반드시 먼저 읽을 것** — 여기 요약은 최소한만 남깁니다.

핵심 요약:
- **2026-08-15, 모델 개발 마감 — 필터(분류기 C) + YOLO(test3) 병렬 구조로 확정.**
  `apps/AI_CCTV/assets/classifier.onnx`(필터+1차 이름) + `best.pt`(test3, 검출+2차
  이름)가 배포본. 상세 성능·모집단·기준은
  `docs/reports/2026-08-15-필터C-test3-병렬-확정.md` 반드시 참고 — 결함코드별
  실측 표가 다 있다. **YOLO 성능은 반드시 야장 기준으로만 인용할 것** — S20/AIHub
  판(testset_bycode)은 학습 유출 의심이 있어 test5에서 90%→33%로 착시가 났었다.
  이 세션 이후 신규 모델 학습은 진행하지 않는다. 다음은 현장 베타테스트.
- 목표는 18종 탐지(`configs/sweeps/test3_18class.yaml`) / 8종 세그멘테이션
  (`configs/sweeps/test4_seg.yaml`) / 31종 마스터 결함 코드 체계(`docs/메타데이터(총괄).xlsx`).
- **Stage-1 필터는 학습 완료 상태**(EfficientNet-B0, 브랜치 `feat/stage1-classifier`)지만
  아직 앱에 연동되지 않았고 실영상 검증도 없다.
- 기본 추론 경로가 Google Colab 무료 GPU + ngrok 고정 도메인에 의존 — 세션 만료 시 탐지 기능 전체가 멈추는
  구조적 리스크가 있음 (PRD §7 참고). `serve_web_cpu.py`로 CPU 로컬 추론 경로가 생겨 완화됨.

## 결함 클래스 코드표 (전체 31종, 자주 참조됨)

```
0  CC  균열-원주        11 LP  연결관-돌출     22 RT  뿌리침입
1  CL  균열-길이        12 LS  이음부-접합부이상 23 IF  침입수
2  CM  균열-복합        13 JS  이음부-이탈     24 PB  막힘
3  SD  표면손상         14 JF  이음부-손상     25 PJ  이음부(정상)
4  BC  좌굴            15 JD  이음부-단차     26 ETC 기타
5  LD  라이닝 결함      16 NS  역경사          27 IN  내부(정상/맥락)
6  DF  변형            17 SG  침하            28 OUT_MH    외부-맨홀
7  BK  파손            18 DE  내피생성        29 OUT_INVERT 외부-인버트
8  CX  붕괴            19 DS  토사퇴적         30 OUT_CAR   외부-자동차
9  PO  영구장애물       20 DG  폐유부착
10 HL  천공            21 TO  임시장애물
```
(출처: `docs/메타데이터(총괄).xlsx` 클래스 코드 시트, `scripts/sweep/label_formats.py::GLOBAL_CLASS_ID`)

## 작업 3줄기

요청이 들어오면 어느 줄기인지 먼저 확인할 것. 2와 3은 둘 다 "모델 학습"이지만 **보는
숫자가 다르다** — 섞어서 보고하지 말 것.

| 줄기 | 대상 | 평가 기준 |
|---|---|---|
| **1. app** | `apps/AI_CCTV` — UI·UX·작업흐름, 사용자가 체감하는 것 | 작업흐름이 매끄러운가 |
| **2. YOLO 학습** | `configs/sweeps`, `scripts/sweep` — 메인 엔진, 결함 **위치** | mAP |
| **3. 필터 학습** | `scripts/classifier` — Stage-1 필터, 결함 **유무** | 놓침(recall) / 오탐(FPR) |

## 최근 작업 로그

### 2026-08-12 세션 — lead 구조 + LLM 병렬

**다른 PC에서 이어서 하려면 [`docs/HANDOFF.md`](docs/HANDOFF.md)를 먼저 읽을 것.**
브랜치는 `feat/stage1-classifier`(main 미병합), 작업본은 `<드라이브>:\AI_CCTV`다
(`AI_CCTV_PROJECT`는 옛 사본이니 쓰지 말 것).

- **앱 구조를 뒤집었다(줄기 1)** — `FILTER_MODE=lead`. 필터가 고른 구간이 표의 행이
  되고 YOLO는 이름·박스만 붙인다. YOLO가 못 붙인 구간은 "확인필요"로 남는다.
- **YOLO를 test6(12종) → test5(17종)로 교체.** test6에 균열 클래스가 아예 없던 게
  "너무 못 잡는다"의 실체였다. mAP50은 0.791→0.698로 내려가지만 잡는 종류가 는다.
- **LLM 판독(Gemini+GPT)을 세 번째 의견으로 붙였다.** ver_2.2 앱에서 옮겨왔고
  `ver.2.0_학습데이터` 예시 사진도 같이 보낸다. **회사망에서 못 쓰므로 기본 꺼짐**,
  앱 화면에서 켠다. 키는 디스크에 저장하지 않는다.
- **`FRAME_INTERVAL` 2초 → 1초.** 2초면 한 프레임에 0.6m씩 건너뛰어 결함이 사이로
  빠진다. 1초로 줄이니 JB1-2-0413의 2:31 결함이 살아났다.
- **신설/노후는 지금 갈라지지 않는다.** 둘 다 같은 필터·YOLO로 돈다.
- 외장하드만 들고 다닐 수 있게 정리: 바탕화면에 있던 `상면 CCTV`(7.55GB)를
  `AI_CCTV_DATASET/video/상면 CCTV/`로 옮기고, 스크립트의 하드코딩 경로를
  `scripts/paths.py`(드라이브 자동 인식)로 통일했다.

### 2026-08-14 세션 — 흑백 기각, 데이터 재고, OLD_v4

**상세: [`docs/reports/2026-08-14-필터-데이터-재고와-실험계획.md`](docs/reports/2026-08-14-필터-데이터-재고와-실험계획.md)**

- **흑백 학습은 기각(줄기 3).** 학습 데이터가 실영상보다 확연히 푸르고 진해서
  (R-B −16~−27 vs −4~−8) 색이 지름길일 수 있다고 봤는데, 지웠더니 야장 val이
  그대로였다(0.9503→0.9517). 기각 판단은 야장 val 기준이다.
- **학습 데이터 원천은 여섯 갈래**(AIHub / S20 / 야장PDF / 상면영상 / fieldset /
  한글조사표). **연속 영상의 평범한 구간을 가진 것은 상면영상뿐**이고, 그것과
  야장을 합친 조합은 지금까지 없었다.
- **현장 간 검증을 한 번도 안 해봤다.** val이 train과 같은 3개 현장이다
  (관로만 나눔). 실제 위험으로 남아 있다.
- **`clsdata_old_v4` 구성**(야장 + 영상 정상 890장×3, train 정상의 17.8%).
  val은 old_v3와 파일명까지 동일 — 비교 가능. 관로 누수 방지가 핵심이었다
  (영상 `SM1-131-001/6058.mp4` = 야장 관로 6058).
- 새 현장 자료가 오면 **학습보다 val로 먼저 쓴다.** 기존 모델을 그냥 돌려보는
  것이 최우선 실험(E1) — 학습 없이 30분이면 현장 간 일반화가 나온다.

### 2026-08-11 세션 — Stage-1 분류 필터 + 앱 관로별 분리
- **필터(줄기 3, 신규)**: YOLO 앞단에 정상/결함 이진 분류기를 두는 2단계 구조.
  EfficientNet-B0, 학습 78,388장(정상:결함 1:1, 결함 25종), Colab T4 24 epoch.
  가장 엄격한 조건에서 재현율 99.4% / 정상 오탐 0.03%, 권장 임계값 0.9.
  **한계 — 결함 25종 중 9종만 정직하게 검증됐다**(정상이 전부 AIHub 출신이라 S20 전용
  16종은 출처만으로 분리 가능). 실제 노후관로 영상 검증도 없고, 앱 미연동.
  코드·보고서는 브랜치 `feat/stage1-classifier` (main 미병합, 의도적).
  데이터 `AI_CCTV_DATASET/clsdata_v1`, 모델 `AI_CCTV_RESULTS/filter/best.pt`.
- **앱(줄기 1)**: 관로별 결과 표시·내보내기, 결함항목 콤보박스, 커서/선택 분리,
  `serve_web_cpu.py`(Colab 없이 CPU 추론), `YOLO_CHUNK_SIZE`로 로컬 추론 OOM 수정,
  `YOLO_CONF` 0.35→0.10.
- **fieldset 라벨 정리**: `fieldset_v1_label`에 31클래스 글로벌 id와 Roboflow 9클래스
  알파벳순 id가 섞여 있던 것을 글로벌 id로 통일(444장 변환, 박스 702개 보존).
  `data.yaml` `nc: 7`→`31`. **이 데이터는 신설관로용**이라 위 필터 학습에 넣지 않았다.

### 2026-07-30 세션 — OCR 버그 수정 + 오탐 필터 + 웹 데모(로그인)
**상세: [`apps/AI_CCTV/docs/WEB_DEMO.md`](apps/AI_CCTV/docs/WEB_DEMO.md) 먼저 읽을 것.** 요약:
- `backend/ocr.py`: 한글 인식 복구(`text_recognition_model_name="korean_PP-OCRv5_mobile_rec"`
  명시 — model_name 지정 시 `lang`이 무시되던 버그), 거리 판독(영역 0.30w + 속도 제외 파서,
  37장 35%→94%), 관경 숫자 안정화(우측 재질라인 + 날짜/시간 제외).
- `backend/overlay_filter.py`(신규) + `yolo_infer`/`yolo_remote`: 위도/경도 등 자막 오탐 제거.
- 웹 데모: `serve_web.py`(신규, 로그인 ON) + `backend/auth.py`(신규) + `ui/login.html`(신규)
  + "추론 서버(Colab)" 연결 패널. **`main.py`(exe)는 그대로, 로그인 없음.**
- 실행: `AUTH_USERS="admin:비번" python serve_web.py --port 8000` → 브라우저 접속(로그인).
- 미완(TODO): Colab 링크 채우기, 터널(cloudflared), 영상 업로드, 사용자별 상태격리, 구글 OAuth.

### 2026-07-29 ~ 2026-07-24 세션에서 한 일

1. **Segmentation 라벨 변환 상태 점검** (`AI_CCTV_DATASET/original/aihub_data_seg/labels`)
   — 소스: `93. ver.2.1_학습데이터(aihub)/Segmentation/라벨링데이터`의 JSON 9개(CL/CC/SD/BK/LP/JF/JD/DS/ETC).
   확인 결과 **이미 전부 변환 완료 상태**였음 (이미지 수=txt 수 정확히 일치, class id도 위 코드표와 일치).
   폴리곤(세그멘테이션) 포맷 YOLO txt로 되어 있음.

2. **`AI_CCTV_DATASET/original/aihub_data_bbox/Train_image`, `Train_labels` 폴더 재구성** (완료됨)
   - 원래 폴더명이 길고 안에 동일 이름 하위폴더가 중복돼 있었음(예: `1-1-1.균열-길이(...)\1-1-1.균열-길이(...)\`) →
     내부 파일을 바깥으로 꺼내고 빈 중복 폴더 삭제.
   - `Train_labels`는 각 클래스 폴더 안에 `labels/train`, `labels/val`, `labels/test`, `labels/잔여`
     등으로 더 나뉘어 있었음 → 전부 flatten해서 클래스 폴더 바로 아래 txt가 오도록 정리.
   - `_1`/`_2`/`_3`로 분할되어 있던 클래스(JD, PJ, IN)는 파일명 번호대가 겹치지 않는 것을 확인 후 하나로 병합.
   - 최종적으로 폴더명을 숫자 접두어 없는 코드명으로 통일: `image/`, `labels/` 아래 각각
     `CL, CC, SD, BK, LP, JF, JD, DS, ETC, PJ, IN, OUT_MH, OUT_INVERT, OUT_CAR` 14개 클래스 폴더.
   - 검증: 클래스별 image 파일 수와 labels 파일 수가 정확히 일치하고, 하위 중첩 폴더가 남아있지 않음을 확인 완료
     (예: BK 36,000 / IN 90,000 / OUT_MH 9,000 등, image·labels 양쪽 동일).
   - 원본 백업: `AI_CCTV_DATASET/original/aihub_data_bbox/labels_original_backup` 존재 (라벨 원본 보존용으로 보임 — 이 세션에서 만든 것은 아님, 기존에 있던 것으로 추정).

3. **PRD 작성**: `docs/PRD.md` — 앱의 실제 기능, 파이프라인, 모델 커버리지 갭(배포 2종 vs 목표
   18/8/31종), 리스크(Colab+ngrok 의존, 세그멘테이션 미연동, 상태 영속성 없음), 로드맵 제안 포함.
   비즈니스 지표/일정처럼 코드에서 확인 안 되는 항목은 TBD로 표시해둠 — 확정 필요.

## 다음에 이어서 할 만한 것 (PRD §8 로드맵 근거)

- **(줄기 2) v6 최종 모델 구성** — 08-10 예정이었으나 지연 중. v5(18종 mAP50 0.703)에서
  정확도가 나오는 항목만 추려 학습한다. 클래스를 늘릴수록 mAP가 떨어지는 게 확인됐다
  (2종 0.821 → 5종 0.759 → 18종 0.703).
- **(줄기 3) 신설/노후 모델 분리 — 파인튜닝이 필요하다.** 지금은 신설·노후가
  같은 모델로 돈다. 신설관로의 미세한 결함(시공품질 관점)은 현재 세팅으로 잡기
  어렵다. 관종별로 파인튜닝해 모델을 나눠야 한다.
- **(줄기 2·3) 변형(DF)이 약하다.** 검증셋에서 이 유형의 성능이 가장 낮다.
- **(줄기 1) 거리 기준 프레임 추출** — 1초 간격이라 카메라 정지 구간에서 47%가 중복된다.
- `configs/detect_v2.yaml`의 빈 `names: {}` 필드, `experiments/detect_v2_bk_ds/metadata.yaml`의
  TBD 항목 채워넣기.
- Colab+ngrok 의존을 벗어난 추론 인프라 검토 (`serve_web_cpu.py`가 폴백은 되지만 정식 경로는 아님).
