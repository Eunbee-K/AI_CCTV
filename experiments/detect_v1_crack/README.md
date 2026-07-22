# Detect V1 — Crack

2026-06-29에 진행한 YOLO11L Detection 실험 묶음이다.

- 데이터: crack 1종
- 현재 학습 데이터 위치: `E:/02. 260629_Model_Test1/ai_cctv/dataset`
- 데이터 수: train 11,000 / val 1,800 / test 4,549
- 결과 위치: `E:/AI_CCTV_RESULTS/detect_v1_crack`
- 최고 mAP50-95: 0.50128 (`sewer_yolo11l_1000test`, best epoch 86)

당시 train/val/test 파일명은
`E:/AI_CCTV_DATASET/detect_v1/manifests`에 기록되어 있다. 파일별 원본 경로는
저장하지 않았으므로 복원 시 원본 폴더에서 이름으로 검색해야 한다.
