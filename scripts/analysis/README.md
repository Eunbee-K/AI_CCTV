# 학습 결과 분석 스크립트

`scripts/sweep/`로 돌린 스윕이 끝난 뒤, 나온 모델이 **왜 그 성능인지** 파고들 때
쓰는 일회성 분석 스크립트 모음. 학습 파이프라인 본체가 아니라 진단 도구이므로
`run_sweep.py`에서 호출되지 않고 각각 따로 실행한다.

2026-08-07 세션에서 `test3_18class_sweep01`(18종 탐지, `test3_960_e50`)의 낮은
recall 원인을 추적하며 작성했다. 경로가 그 실험 기준으로 하드코딩되어 있으니
다른 실험에 쓸 때는 파일 상단의 `RUN` / `V` 상수를 바꿀 것.

## 실행

`ultralytics`가 설치된 venv로 실행해야 한다 (시스템 파이썬에는 없음):

```bash
/home/workstation/venvs/test/bin/python3 scripts/analysis/<스크립트>.py
```

`rebuild_val_and_curves.py`는 `scripts/sweep/`의 모듈(`common`, `collect_dataset`)을
import하므로 저장소 경로가 바뀌면 상단 `sys.path.insert` 경로도 함께 고쳐야 한다.

## 구성

| 파일 | 역할 |
|---|---|
| `rebuild_val_and_curves.py` | **먼저 실행할 것.** 학습 후 삭제된 임시 데이터셋의 val split만 원본 quota/seed 그대로 재현해 `_sweep_workspace/val_only_test3/`에 물리 복사. 아래 스크립트들이 이 폴더를 입력으로 쓴다 |
| `threshold_table.py` | confidence 임계값(0.05~0.50)별 precision/recall 표. macro 평균과 인스턴스 수 가중평균을 함께 내고, conf 0.10 vs 0.25를 클래스별로 비교 |
| `val_conf010.py` | 최종 모델을 conf=0.10(재현율 우선)으로 재검증하고 PR/F1 커브·confusion matrix를 `<RUN>/val_conf010/`에 저장 |
| `source_split_eval.py` | val을 소스별(서울시 `seoul_txt` vs AI Hub `yolo_txt`)로 갈라 같은 모델로 각각 평가. 소스 간 라벨링 기준 불일치를 클래스별 mAP50 차이로 드러냄. 심볼릭 링크라 디스크를 거의 안 씀 |
| `label_audit.py` | val 라벨을 (클래스, 소스)별로 갈라 박스 수/중앙 면적/가로세로비/이미지 전체를 덮는 거대 박스 비율/불량 라인 수를 집계. 라벨링 기준 차이를 통계로 진단 |
| `visual_audit.py` | 약한 클래스(AI Hub CC/JF)의 GT(초록)와 예측(빨강)을 나란히 붙인 격자 이미지를 `<RUN>/label_audit/`에 저장. 통계로 안 보이는 라벨 품질을 눈으로 확인 |
| `iou_sensitivity.py` | IoU 임계값(0.30~0.75)별 AP를 계산. `AP@0.3 >> AP@0.5`면 "못 찾는" 게 아니라 "찾았는데 박스 위치가 어긋나는" 문제라는 뜻 |

## 대략의 진단 흐름

1. `rebuild_val_and_curves.py`로 val 복원
2. `threshold_table.py` — 기본 conf=0.25가 recall을 깎고 있는지 확인
3. `source_split_eval.py` — 성능 저하가 특정 소스에 몰려 있는지 확인
4. `label_audit.py` + `visual_audit.py` — 몰려 있다면 그 소스의 라벨 기준을 통계·육안으로 검증
5. `iou_sensitivity.py` — 미검출 문제인지 위치정밀도 문제인지 분리
