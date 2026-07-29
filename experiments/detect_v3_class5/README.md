# Detect V3 — BK / LP / JF / JD / DS (5 classes)

2026-07-10~29에 워크스테이션에서 진행한 YOLO11L Detection 실험이다 (26 epoch까지 학습 후 세션 단절로
중단, 이후 legacy 경로로 재개하여 42 epoch까지 진행한 뒤 수동 중지).

- 실험명: `test3_yolo11L_class5_16000`
- 워크스테이션 데이터 경로(현재): `/home/workstation/ai_cctv/legacy/260710_TEST3/dataset`
- 결과 위치: `E:/AI_CCTV_RESULTS/detect_v3_class5/test3_yolo11L_class5_16000`
- 클래스: BK(파손), LP(연결관-돌출), JF(이음부-손상), JD(이음부-단차), DS(토사퇴적)
- imgsz: 960
- 설정 epoch: 50 / 완료 epoch: 42 (수동 중지, patience 미도달)
- 최고 epoch(best.pt): 35 — precision 0.7229, recall 0.7280, mAP50 0.7595, mAP50-95 0.5702
- best.pt 기준 val 재실행 결과: precision 0.7205, recall 0.7309, mAP50 0.7595, mAP50-95 0.5701
- 클래스별 mAP50 (val 재실행): BK 0.600 / LP 0.939 / JF 0.495 / JD 0.822 / DS 0.942
  → BK, JF가 상대적으로 약함

이어서 학습하려면 `weights/last.pt`(epoch 42 시점)에서 `resume`하면 된다. 단, ultralytics
resume은 체크포인트에 저장된 `project`/`name`을 그대로 재사용하므로, 저장 위치를 바꾸고 싶다면
체크포인트 자체를 원하는 경로로 복사한 뒤 그 경로로 resume해야 한다.
