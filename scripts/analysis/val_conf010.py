"""test3 최종 모델을 conf=0.1로 검증 (기본 0.25 대비 recall 우선 설정)."""
from ultralytics import YOLO

RUN = "/home/workstation/ai_cctv/AI_CCTV_RESULTS/test3_18class_sweep01/test3_960_e50"
model = YOLO(f"{RUN}/weights/best.pt")

res = model.val(
    data="/home/workstation/ai_cctv/_sweep_workspace/val_only_test3/data.yaml",
    imgsz=960,
    batch=8,
    device=0,
    split="val",
    conf=0.10,        # 재현율 우선
    plots=True,       # PR/P/R/F1 커브 + confusion matrix
    project=RUN,
    name="val_conf010",
    exist_ok=True,
    verbose=True,
)

print("\n===== 전체 (conf=0.10) =====")
print(f"mAP50={res.box.map50:.4f}  mAP50-95={res.box.map:.4f}  "
      f"P={res.box.mp:.4f}  R={res.box.mr:.4f}")
print(f"save_dir={res.save_dir}")
