# legacy — 지금은 쓰지 않는 스크립트

2026-08-16에 `scripts/classifier/`에서 옮겼다. **지우지 않은 이유**는 "그때 그
데이터셋을 어떻게 만들었지"를 되짚을 일이 생기기 때문이다. 배포 모델
(`classifier.onnx` + `best.pt`)을 만드는 데는 하나도 쓰이지 않는다.

지금 쓰는 것은 상위 폴더에 있다 — 목록은 [`../README.md`](../README.md) 참고.

## 실행할 때 주의

여기 있는 것 중 10개가 상위 폴더 모듈(`train_binary`, `build_v5_and_valset` 등)을
import한다. 옮기면서 경로가 한 단계 깊어졌으므로 그대로 실행하면 ImportError가 난다.
되짚어볼 일이 생기면 상위 폴더에서 실행하거나 `PYTHONPATH`를 잡아준다:

```bash
cd scripts/classifier
python -m legacy.<스크립트이름>          # 또는
PYTHONPATH=scripts/classifier python scripts/classifier/legacy/<이름>.py
```

## 폐기된 데이터셋 빌더

필터 v1~v5 계보다. 최종본은 `build_old_v6.py`(상위 폴더)이고, 그 결과물
`clsdata_old_v6`로 배포 모델을 학습했다.

| 파일 | 무엇이었나 | 왜 안 쓰나 |
|---|---|---|
| `build_binary_dataset.py` | 최초 이진 필터(v1) 학습셋 | AIHub 단일 출처라 현장에서 AUC 0.516 |
| `build_old_filter_dataset.py` | 노후관로 필터 초기본 | v2로 대체 |
| `build_old_v2.py` | v2 — AIHub OUT_*을 뺀 구성 | 관 밖을 한 장도 안 보여줘 관밖 오탐 39% |
| `build_old_v3.py` | v3 — 야장 도입 | 오래 배포됐으나 v6가 대체 |
| `build_old_v4.py` | v4 — 영상 정상 890장 추가 | 학습까지 가지 않음 |
| `build_new_filter.py` | 신설관로 전용 필터 | 데이터 1,404장으로 AUC 0.75, 미달 |
| `build_new_v2.py` | 신설 v2 | 위와 같은 이유 |
| `build_mixed_dataset.py` | 신설+노후 혼합 | 신설/노후를 나누지 않기로 정리됨 |
| `build_field_filter_dataset.py` | fieldset(신설) 기반 | 신설 전용이라 노후 필터에 안 넣음 |
| `add_newpipe_data.py` | fieldset을 train에 덧붙임 | 위와 같음 |

## 일회성 평가·비교

당시 판단을 위해 만든 것들이다. 지금 성능 측정은 상위 폴더의
`score_common_holdout.py`(필터C vs test3 공통 판)와 `score_deployed*.py`를 쓴다.

| 파일 | 무엇이었나 |
|---|---|
| `eval_binary.py` | 이진 필터 기본 채점(AUC·임계값 표) |
| `eval_old_models.py` | 옛 후보 모델들을 야장·AIHub로 일괄 채점 |
| `eval_on_video.py` | 조사표 거리를 정답으로 펼쳐 실영상 채점 — **이 방식이 "실영상 AUC 0.52"의 원인**이었다(정답이 부풀려짐, 폐기) |
| `score_on_real_video.py` | 위와 같은 계열 |
| `eval_series_vs_parallel.py` | series/parallel 모드 비교 → lead 채택으로 종료 |
| `compare_runs.py` | 여러 학습 결과를 같은 val로 재비교 |
| `clean_eval.py` | train과 겹치는 val 이미지를 빼고 재채점 |
| `export_onnx.py` | 이진 필터 ONNX 내보내기 → `export_multiclass_onnx.py`로 대체 |

## 데이터 점검·정리 도구

| 파일 | 무엇이었나 |
|---|---|
| `check_duplicates.py` / `verify_duplicates.py` | val↔train 중복(dHash) 검사 |
| `cleanup_v2.py` | 육안 검토 결과를 clsdata_v2에 반영 |
| `resplit.py` | 이미 만든 데이터셋의 train/val 재분할 |
| `rebuild_hires.py` | 라벨은 두고 이미지만 고해상도로 재생성 |
| `export_review_frames.py` | 검수용 프레임을 원본 해상도로 추출 |
| `extract_ordinary_normals.py` | 연속 영상에서 "평범한 정상" 구간 추출 |
| `pack_for_colab.py` | 데이터셋을 Colab용 zip으로 포장 → `resize_dataset.py` + 직접 압축으로 대체 |
