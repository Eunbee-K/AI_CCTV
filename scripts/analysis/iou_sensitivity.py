"""IoU 임계값별 AP를 계산해 '못 찾는 것' vs '찾았는데 박스가 어긋남'을 구분한다.
AP@0.3 >> AP@0.5 이면 탐지는 되는데 위치정밀도가 문제라는 뜻.
"""
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou, ap_per_class

V = Path("/home/workstation/ai_cctv/_sweep_workspace/val_only_test3")
RUN = "/home/workstation/ai_cctv/AI_CCTV_RESULTS/test3_18class_sweep01/test3_960_e50"
model = YOLO(f"{RUN}/weights/best.pt")
names = model.names
name2id = {v: k for k, v in names.items()}

IOUV = np.array([0.30, 0.40, 0.50, 0.60, 0.75])


def match(gt_boxes, gt_cls, det_boxes, det_cls, iouv):
    correct = np.zeros((det_boxes.shape[0], len(iouv)), dtype=bool)
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return correct
    iou = box_iou(torch.tensor(gt_boxes, dtype=torch.float32),
                  torch.tensor(det_boxes, dtype=torch.float32)).numpy()
    iou = iou * (gt_cls[:, None] == det_cls[None, :])
    for i, th in enumerate(iouv):
        m = np.array(np.nonzero(iou >= th)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def evaluate(cls_code, prefix):
    cid = name2id[cls_code]
    files = sorted((V / "images" / "val").glob(f"{prefix}_{cls_code}__*"))
    tps, confs, pcls, tcls = [], [], [], []
    for f in files:
        img_shape = None
        # GT
        lbl = V / "labels" / "val" / (f.stem + ".txt")
        gb, gc = [], []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(p[0]); cx, cy, w, h = map(float, p[1:5])
            gb.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            gc.append(c)
        r = model.predict(str(f), imgsz=960, conf=0.001, device=0, verbose=False)[0]
        H, W = r.orig_shape
        db = (r.boxes.xyxy.cpu().numpy() / np.array([W, H, W, H])) if len(r.boxes) else np.zeros((0, 4))
        dc = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros(0, int)
        dconf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        gb = np.array(gb) if gb else np.zeros((0, 4))
        gc = np.array(gc, dtype=int)
        # 관심 클래스만
        keep_d = dc == cid
        db, dc, dconf = db[keep_d], dc[keep_d], dconf[keep_d]
        keep_g = gc == cid
        gb, gc = gb[keep_g], gc[keep_g]
        tps.append(match(gb, gc, db, dc, IOUV))
        confs.append(dconf); pcls.append(dc); tcls.append(gc)
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOUV)), bool)
    conf = np.concatenate(confs); pc = np.concatenate(pcls); tc = np.concatenate(tcls)
    out = ap_per_class(tp, conf, pc, tc, plot=False, names={})
    ap = out[5]  # (nc, n_iou)
    return ap[0] if len(ap) else np.zeros(len(IOUV)), len(tc)


print(f"{'클래스':8}{'소스':8}{'GT수':>7}" + "".join(f"{f'AP@{t:.2f}':>10}" for t in IOUV))
print("-" * 80)
for cls, prefix in [("CC", "yolo_txt"), ("JF", "yolo_txt"), ("CL", "yolo_txt"),
                    ("SD", "yolo_txt"), ("LP", "yolo_txt"), ("DS", "yolo_txt")]:
    ap, n = evaluate(cls, prefix)
    src = "AIHub"
    print(f"{cls:8}{src:8}{n:7}" + "".join(f"{v:10.3f}" for v in ap))
