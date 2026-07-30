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
- 현재 배포 모델(`apps/AI_CCTV/assets/best.pt`)은 실사용 가능한 클래스가 **BK(파손)·DS(토사퇴적) 2종뿐**.
- 목표는 18종 탐지(`configs/sweeps/test3_18class.yaml`) / 8종 세그멘테이션
  (`configs/sweeps/test4_seg.yaml`) / 31종 마스터 결함 코드 체계(`docs/메타데이터(총괄).xlsx`).
  오늘(2026-07-29) 안에 끝내는 20시간 예산판 `test3_18class_20h.yaml` /
  `test4_seg_20h.yaml`도 있음 (자세한 내용은 세션 로그 참고).
- 기본 추론 경로가 Google Colab 무료 GPU + ngrok 고정 도메인에 의존 — 세션 만료 시 탐지 기능 전체가 멈추는
  구조적 리스크가 있음 (PRD §7 참고).

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

## 최근 작업 로그

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

- `configs/sweeps/test3_18class.yaml` 스윕 실행 → 18종 탐지 모델 학습 (데이터는 이제
  `aihub_data_bbox/image`, `labels` 아래 클래스별 폴더로 정리되어 있어 바로 활용 가능).
- `configs/detect_v2.yaml`의 빈 `names: {}` 필드, `experiments/detect_v2_bk_ds/metadata.yaml`의
  TBD 항목 채워넣기.
- Colab+ngrok 의존을 벗어난 추론 인프라 검토.
