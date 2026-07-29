# AI CCTV Project

코드, 설정, 실험 메타데이터만 Git으로 관리

이 저장소는 두 곳에 존재한다:
- **외장 SSD/HDD** (이동식, 드라이브 문자는 연결한 PC마다 다름 — 원래 PC에서는 `E:\`): 원본이자
  다른 PC로 옮겨 다닐 때 쓰는 사본.
- **워크스테이션 로컬** (`/home/workstation/ai_cctv/AI_CCTV_PROJECT`): 2026-07-29부터 실제
  스윕 학습을 돌리는 사본. 데이터셋(`AI_CCTV_DATASET`)과 결과(`AI_CCTV_RESULTS`)도 같은 위치에
  통째로 복사해둬서 외장하드를 뽑아도 학습이 끊기지 않는다.

- 실제 데이터셋: 외장하드 `E:/AI_CCTV_DATASET` / 워크스테이션 로컬 `/home/workstation/ai_cctv/AI_CCTV_DATASET`
- 학습 결과와 가중치: 외장하드 `E:/AI_CCTV_RESULTS` / 워크스테이션 로컬 `/home/workstation/ai_cctv/AI_CCTV_RESULTS`
- 원본 데이터와 대용량 모델 파일은 Git에 넣지 않음

## 폴더

- `configs/`: 학습 데이터 YAML
- `experiments/`: 실험별 조건과 성능 기록
- `scripts/`: 실행 및 보조 스크립트
- `notebooks/`: 분석 노트북
- `docs/`: 공통 문서

