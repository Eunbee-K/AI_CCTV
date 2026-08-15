# scripts/classifier — 필터(분류기) 데이터셋·학습·채점

배포 모델 `apps/AI_CCTV/assets/classifier.onnx`(EfficientNet-B0, 24클래스)를
만들고 재는 스크립트들이다. 지금 안 쓰는 것은 [`legacy/`](legacy/README.md)로 옮겼다.

전체 구조와 성능은
[`docs/reports/2026-08-16-결함분석기-시스템구조.md`](../../docs/reports/2026-08-16-결함분석기-시스템구조.md)
를 먼저 읽을 것.

## 배포 모델을 다시 만들려면 (순서대로)

```bash
python build_old_v6.py          # 1. 학습셋 구성 (야장 + S20/AIHub, 관 밖 정상 20%)
python build_multiclass.py      # 2. 이진 -> 24클래스 폴더 구조로 재배치
python resize_dataset.py        # 3. 384px 사본 (Colab 업로드용, 원본 보존)
                                # 4. Colab에서 train_multiclass.py 실행
python export_multiclass_onnx.py --ckpt <best_acc.pt>   # 5. 앱이 쓸 ONNX로
```

| 파일 | 역할 |
|---|---|
| `build_old_v6.py` | 최종 학습셋(`clsdata_old_v6`) 구성 |
| `build_multiclass.py` | 이진 데이터셋을 결함코드별 폴더로 재배치 |
| `resize_dataset.py` | 384px JPEG 사본 생성(`--gray` 옵션) |
| `train_multiclass.py` | EfficientNet 다중 클래스 학습 (Colab) |
| `export_multiclass_onnx.py` | ONNX 내보내기 + 클래스 이름표(.classes.json) |

`train_binary.py`는 이진 필터(옛 계보) 학습기인데, **legacy 스크립트 8개가
import하고 있어 남겨둔다.** 배포 경로에는 쓰이지 않는다.

## 성능을 재려면

| 파일 | 무엇을 재나 |
|---|---|
| **`build_common_holdout.py`** | **필터C와 test3의 학습분을 모두 제외한 공통 검증셋** |
| **`score_common_holdout.py`** | **두 모델을 같은 사진으로 나란히 채점 (`--xlsx`)** |
| `score_deployed.py` | 배포된 분류기만 — 필터 성능·이름 정확도 |
| `score_deployed_full.py` | 분류기 + YOLO 병렬 합산 |
| `score_testset_bycode.py` | 체크포인트(.pt) 직접 채점 — 학습 직후 확인용 |

**검증셋을 새로 만들 때는 반드시 학습분과 대조할 것.** 스윕 결과 폴더의
`used_files.csv`가 그 run이 실제로 쓴 파일 목록이다. 이걸 안 보고 만든
`testset_bycode`는 test5 학습분과 56.9% 겹쳐 "이름 정확도 90%"라는 착시를 만들었다
(실제 야장 기준 33%). `build_common_holdout.py`가 이 대조를 자동으로 한다.

| 파일 | 용도 |
|---|---|
| `build_test3_holdout.py` / `score_test3_holdout.py` | test3만 따로 (야장 80% 구성) |
| `build_yolo_holdout.py` / `score_yolo_holdout.py` | YOLO 계열 일반 (test3/test5 선택) |
| `build_testset_bycode.py` | 결함코드별 판 — **YOLO 평가에는 쓰지 말 것**(위 참고) |
| `build_valset.py` / `build_v5_and_valset.py` | 야장 우선 검증셋. 후자는 코드 이름표도 제공 |
| `report_dataset.py` | 데이터셋 구성 요약 출력 |
