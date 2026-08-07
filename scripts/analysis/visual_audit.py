"""약한 클래스(AI Hub CC/JF)의 GT 라벨 + 모델 예측을 나란히 그려 라벨 품질을 눈으로 확인."""
import random
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

V = Path("/home/workstation/ai_cctv/_sweep_workspace/val_only_test3")
RUN = "/home/workstation/ai_cctv/AI_CCTV_RESULTS/test3_18class_sweep01/test3_960_e50"
# 감사 이미지는 결과 폴더 안에 남긴다 (git 미관리 — 필요할 때 다시 생성).
OUT = Path(RUN) / "label_audit"
OUT.mkdir(parents=True, exist_ok=True)
model = YOLO(f"{RUN}/weights/best.pt")
names = model.names

CELL = 420
def make_grid(cls_code, prefix, n=8, seed=7):
    files = sorted((V / "images" / "val").glob(f"{prefix}_{cls_code}__*"))
    random.Random(seed).shuffle(files)
    files = files[:n]
    cells = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        # GT (초록)
        lbl = V / "labels" / "val" / (f.stem + ".txt")
        gt = img.copy()
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(p[0]); cx, cy, bw, bh = map(float, p[1:5])
            x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
            x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
            cv2.rectangle(gt, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(gt, names[cid], (x1+4, max(y1-6,14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        # 예측 (빨강)
        pred = img.copy()
        r = model.predict(str(f), imgsz=960, conf=0.10, device=0, verbose=False)[0]
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            cid = int(b.cls[0]); cf = float(b.conf[0])
            cv2.rectangle(pred, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(pred, f"{names[cid]} {cf:.2f}", (x1+4, max(y1-6,14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        pair = np.hstack([cv2.resize(gt,(CELL,CELL)), cv2.resize(pred,(CELL,CELL))])
        cv2.putText(pair, "GT", (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(pair, "PRED", (CELL+8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cells.append(pair)
    rows = [np.hstack(cells[i:i+2]) for i in range(0, len(cells)-1, 2)]
    grid = np.vstack(rows)
    out = OUT / f"audit_{prefix}_{cls_code}.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  {out}  ({len(cells)}쌍)")

for cls in ["CC", "JF"]:
    print(f"[AI Hub {cls}]")
    make_grid(cls, "yolo_txt", n=8)
