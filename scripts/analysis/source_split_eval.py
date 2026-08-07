"""val을 소스(서울시 vs AI Hub)별로 갈라 같은 모델로 각각 평가.
라벨 기준 불일치가 원인이면 한쪽 소스에서만 성능이 무너진다.
심볼릭 링크로 만들어 디스크/시간 절약.
"""
import shutil
from pathlib import Path
import yaml

SRC = Path("/home/workstation/ai_cctv/_sweep_workspace/val_only_test3")
BASE = Path("/home/workstation/ai_cctv/_sweep_workspace")
CFG = yaml.safe_load(open("/home/workstation/ai_cctv/AI_CCTV_PROJECT/configs/sweeps/test3_18class.yaml"))
class_codes = list(CFG["dataset"]["classes"].keys())

for tag, prefix in [("seoul", "seoul_txt_"), ("aihub", "yolo_txt_")]:
    dest = BASE / f"val_src_{tag}"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "images" / "val").mkdir(parents=True)
    (dest / "labels" / "val").mkdir(parents=True)
    n = 0
    for img in (SRC / "images" / "val").iterdir():
        if not img.name.startswith(prefix):
            continue
        lbl = SRC / "labels" / "val" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        (dest / "images" / "val" / img.name).symlink_to(img)
        (dest / "labels" / "val" / lbl.name).symlink_to(lbl)
        n += 1
    names_yaml = "\n".join(f"  {i}: {c}" for i, c in enumerate(class_codes))
    (dest / "data.yaml").write_text(
        f"path: {dest}\ntrain: images/val\nval: images/val\n"
        f"nc: {len(class_codes)}\nnames:\n{names_yaml}\n", encoding="utf-8")
    print(f"[{tag}] {n}장 -> {dest}")

# 평가
from ultralytics import YOLO
RUN = "/home/workstation/ai_cctv/AI_CCTV_RESULTS/test3_18class_sweep01/test3_960_e50"
model = YOLO(f"{RUN}/weights/best.pt")

results = {}
for tag in ("seoul", "aihub"):
    print(f"\n{'='*60}\n[{tag}] 평가 중...\n{'='*60}")
    res = model.val(
        data=str(BASE / f"val_src_{tag}" / "data.yaml"),
        imgsz=960, batch=8, device=0, split="val",
        conf=0.001, plots=False, verbose=True,
    )
    per = {}
    for i, c in enumerate(res.box.ap_class_index):
        per[res.names[c]] = float(res.box.ap50[i])
    results[tag] = per

print("\n\n########## 소스별 mAP50 비교 ##########")
print(f"{'class':6}{'서울시':>10}{'AI Hub':>10}{'차이':>10}")
print("-" * 40)
allc = sorted(set(results["seoul"]) | set(results["aihub"]))
for c in allc:
    s = results["seoul"].get(c)
    a = results["aihub"].get(c)
    if s is None or a is None:
        only = "서울시만" if a is None else "AIHub만"
        v = s if a is None else a
        print(f"{c:6}{v:10.3f}{'':>10}  ({only})")
        continue
    print(f"{c:6}{s:10.3f}{a:10.3f}{a-s:+10.3f}")
