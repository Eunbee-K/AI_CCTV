from ultralytics import YOLO
import numpy as np

RUN = "/home/workstation/ai_cctv/AI_CCTV_RESULTS/test3_18class_sweep01/test3_960_e50"
model = YOLO(f"{RUN}/weights/best.pt")
res = model.val(
    data="/home/workstation/ai_cctv/_sweep_workspace/val_only_test3/data.yaml",
    imgsz=960, batch=8, device=0, split="val",
    conf=0.001, plots=False, verbose=False,
)
box = res.box
px = box.px                      # confidence 축 (보통 0~1, 1000포인트)
p_curve = box.p_curve            # shape (nc, len(px))
r_curve = box.r_curve            # shape (nc, len(px))
f1_curve = box.f1_curve
names = res.names
ap_idx = box.ap_class_index      # 실제 평가된 클래스 인덱스 (px/curve의 row 순서와 대응)

# 인스턴스 수(가중치)로 클래스별 recall/precision을 가중평균 -> "실제 결함 기준" recall
import json
per_class = json.load(open(f"{RUN}/per_class_metrics.json"))
# per_class엔 인스턴스 수가 없으므로 stats에서 직접 집계
nt_per_class = res.nt_per_class if hasattr(res, "nt_per_class") else None
if nt_per_class is None:
    nt_per_class = getattr(box, "nt_per_class", None)

print(f"px 길이={len(px)}, 평가된 클래스 수={len(ap_idx)}")
print(f"nt_per_class 사용 가능: {nt_per_class is not None}\n")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
print(f"{'conf':>6} | {'macro P':>8} {'macro R':>8} | {'inst-weighted R':>16} {'inst-weighted P':>16}")
for th in thresholds:
    i = int(np.argmin(np.abs(px - th)))
    p_at = p_curve[:, i]
    r_at = r_curve[:, i]
    macro_p, macro_r = p_at.mean(), r_at.mean()
    if nt_per_class is not None:
        w = np.array([nt_per_class[c] for c in ap_idx], dtype=float)
        w = w / w.sum()
        wp = float((p_at * w).sum())
        wr = float((r_at * w).sum())
    else:
        wp = wr = float("nan")
    print(f"{th:6.2f} | {macro_p:8.3f} {macro_r:8.3f} | {wr:16.3f} {wp:16.3f}")

# 기본 conf=0.25 대비 recall 우선 후보 conf=0.10/0.15 요약
print("\n=== 클래스별 (conf=0.10 vs 0.25) ===")
i25 = int(np.argmin(np.abs(px - 0.25)))
i10 = int(np.argmin(np.abs(px - 0.10)))
for row, c in enumerate(ap_idx):
    name = names[c]
    print(f"  {name:4s}  conf0.25: P={p_curve[row,i25]:.2f} R={r_curve[row,i25]:.2f}   "
          f"conf0.10: P={p_curve[row,i10]:.2f} R={r_curve[row,i10]:.2f}")
