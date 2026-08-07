"""val 라벨을 (클래스, 소스)별로 갈라서 라벨링 기준 차이를 통계로 진단."""
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

V = Path("/home/workstation/ai_cctv/_sweep_workspace/val_only_test3/labels/val")
PAT = re.compile(r"^(seoul_txt|yolo_txt|dir_pair)_([A-Z_]+)__")

stats = defaultdict(lambda: {"imgs": 0, "boxes": 0, "areas": [], "ars": [],
                             "ws": [], "hs": [], "bad": 0, "full": 0, "empty": 0})

for f in V.iterdir():
    if f.suffix != ".txt":
        continue
    m = PAT.match(f.name)
    if not m:
        continue
    kind, cls = m.group(1), m.group(2)
    key = (cls, kind)
    s = stats[key]
    s["imgs"] += 1
    lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        s["empty"] += 1
    for l in lines:
        p = l.split()
        if len(p) < 5:
            s["bad"] += 1
            continue
        try:
            _, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        except ValueError:
            s["bad"] += 1
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1) or w <= 0 or h <= 0 or w > 1.001 or h > 1.001:
            s["bad"] += 1
            continue
        s["boxes"] += 1
        s["areas"].append(w * h)
        s["ars"].append(w / h if h > 0 else 0)
        s["ws"].append(w)
        s["hs"].append(h)
        if w * h > 0.7:
            s["full"] += 1

TARGET = ["CC", "CL", "JF", "SD", "BK", "DS", "JD", "LP"]
print(f"{'cls':4} {'source':10} {'imgs':>6} {'boxes':>7} {'box/img':>8} "
      f"{'중앙면적':>9} {'중앙W':>7} {'중앙H':>7} {'가로세로비':>9} {'거대박스%':>9} {'불량':>5}")
print("-" * 100)
for cls in TARGET:
    for kind in ("seoul_txt", "yolo_txt"):
        s = stats.get((cls, kind))
        if not s or s["imgs"] == 0:
            continue
        n = s["boxes"]
        if n == 0:
            print(f"{cls:4} {kind:10} {s['imgs']:6} {n:7}  (박스 없음)")
            continue
        print(f"{cls:4} {kind:10} {s['imgs']:6} {n:7} {n/s['imgs']:8.2f} "
              f"{st.median(s['areas']):9.4f} {st.median(s['ws']):7.3f} {st.median(s['hs']):7.3f} "
              f"{st.median(s['ars']):9.2f} {100*s['full']/n:8.1f}% {s['bad']:5}")
    print()
