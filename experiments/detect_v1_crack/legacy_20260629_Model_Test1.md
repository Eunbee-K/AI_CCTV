# 20260629 Model Test 1 결과 정리

## 1. 실험 개요

- 실험 그룹: `detect_v1_crack`
- 작업 유형: BBox Detection
- 모델 계열: YOLO11L
- 모델 클래스: `0: crack`
- 완료 실험: 5개
- 미완료 실행: 1개
- 데이터셋 메타데이터: `E:/AI_CCTV_DATASET/detect_v1`
- 학습 결과: `E:/AI_CCTV_RESULTS/detect_v1_crack`

## 2. 사용 데이터

실제 학습 당시 데이터 위치는 다음과 같다.

```text
E:/02. 260629_Model_Test1/ai_cctv/dataset
```

현재 `E:/AI_CCTV_DATASET/detect_v1`에는 실제 이미지와 라벨을 복사하지 않고
당시 사용한 파일 목록과 메타데이터만 보관한다.

| 구분 | 이미지 | 라벨 |
|---|---:|---:|
| train | 11,000 | 11,000 |
| val | 1,800 | 1,800 |
| test | 4,549 | 4,549 |
| 합계 | 17,349 | 17,349 |

- 라벨 형식: YOLO BBox
- 모델 학습 클래스: `crack` 1종
- 파일명으로 확인된 세부 결함 `CL`: 16,454개
- 파일명만으로 세부 유형을 확정하지 못한 crack: 895개
- 파일명 목록: `E:/AI_CCTV_DATASET/detect_v1/manifests`

## 3. 완료 실험

성능은 각 `results.csv`에서 `mAP50-95`가 가장 높은 epoch를 기준으로 정리했다.

| 실험명 | imgsz | 설정 epoch | 완료 epoch | best epoch | best mAP50-95 | best mAP50 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sewer_yolo11l_1000test` | 640 | 100 | 100 | 86 | 0.50128 | 0.78133 | 0.80455 | 0.69314 |1
| `sewer_yolo11l_1000test-2` | 640 | 200 | 136 | 106 | 0.47509 | 0.78767 | 0.78380 | 0.73291 |
| `sewer_yolo11l_1000test-3` | 1280 | 100 | 100 | 94 | 0.42044 | 0.74679 | 0.77477 | 0.70036 |
| `sewer_yolo11l_1000test-4` | 960 | 100 | 100 | 86 | 0.48815 | 0.79507 | 0.78082 | 0.74592 |
| `sewer_yolo11l_960-2` | 960 | 100 | 70 | 40 | 0.43690 | 0.67384 | 0.75758 | 0.61072 |

## 4. 미완료 실행

`sewer_yolo11l_960`은 `args.yaml`은 있지만 `results.csv`와 완료 가중치가 없어
완료 실험 수에서 제외한다.

## 5. 결과 해석

- 최고 `mAP50-95`는 `sewer_yolo11l_1000test`의 **0.50128**이다.
- 최고 `mAP50`은 `sewer_yolo11l_1000test-4`의 **0.79507**이다.
- `sewer_yolo11l_1000test-4`는 Recall도 0.74592로 상대적으로 높다.
- `sewer_yolo11l_960-2`는 더 큰 데이터 구성을 사용했지만 현재 조건에서는
  최고 `mAP50-95`가 0.43690으로 1,000장 실험보다 낮았다.

## 6. 현재 보관 위치

```text
E:/AI_CCTV_RESULTS/detect_v1_crack/
├── sewer_yolo11l_1000test
├── sewer_yolo11l_1000test-2
├── sewer_yolo11l_1000test-3
├── sewer_yolo11l_1000test-4
├── sewer_yolo11l_960
└── sewer_yolo11l_960-2
```

각 완료 실험 폴더에는 `args.yaml`, `results.csv`, 그래프 및 `weights`가
보관되어 있다.

## 7. 재현 관련 주의사항

- 당시 train/val/test 파일명은 `detect_v1/manifests`에 저장되어 있다.
- 파일별 원본 절대경로는 기록하지 않았다.
- 실제 데이터를 다시 구성할 경우 manifest의 파일명을 원본 데이터에서 찾아
  `E:/AI_CCTV_DATASET/_materialized/detect_v1`에 배치한다.
- 학습 설정은 `E:/AI_CCTV_PROJECT/configs/detect_v1.yaml`을 사용한다.
