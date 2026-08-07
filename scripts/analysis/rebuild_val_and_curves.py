"""test3 최종 모델의 val split만 정확히 복원(원본 quota/seed 그대로, val만 물리 복사)한 뒤,
confidence 임계값별 precision/recall을 실측 계산한다.
"""
import sys
sys.path.insert(0, "/home/workstation/ai_cctv/AI_CCTV_PROJECT/scripts/sweep")
from pathlib import Path
import shutil
from common import load_yaml, ensure_dir
import collect_dataset as cd

CFG_PATH = "/home/workstation/ai_cctv/AI_CCTV_PROJECT/configs/sweeps/test3_18class.yaml"
DEST = Path("/home/workstation/ai_cctv/_sweep_workspace/val_only_test3")

cfg = load_yaml(CFG_PATH)
ds = cfg["dataset"]
external_root = Path(ds["external_root"])
classes_cfg = ds["classes"]
class_codes = list(classes_cfg.keys())
class_to_id = {code: i for i, code in enumerate(class_codes)}
seed = ds.get("split_seed", 42)

if DEST.exists():
    shutil.rmtree(DEST)
ensure_dir(DEST / "images" / "val")
ensure_dir(DEST / "labels" / "val")

print(f"클래스 {len(class_codes)}개, seed={seed} 로 원본과 동일한 셔플/슬라이스 재현 (val만 물리 복사)")
n_total = 0
for class_code, spec in classes_cfg.items():
    pool = cd.build_class_pool(external_root, class_code, spec["sources"])
    picked = cd.deterministic_pick(pool, spec["quota"], seed, class_code)
    is_background = spec.get("background", False)
    for item in picked["val"]:
        img_path, kind, _label_path, _code = item
        yolo_lines = cd.convert_one(item, class_to_id)
        if not yolo_lines and not is_background:
            continue
        safe_prefix = f"{kind}_{class_code}"
        dest_img = DEST / "images" / "val" / f"{safe_prefix}__{img_path.name}"
        dest_lbl = DEST / "labels" / "val" / f"{safe_prefix}__{img_path.stem}.txt"
        shutil.copy2(img_path, dest_img)
        dest_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")
        n_total += 1
    print(f"  [{class_code}] val {len(picked['val'])}장 복사 완료 (누적 {n_total})")

# data.yaml (train은 형식상 val과 동일 경로로 채움 - 실제 val()엔 안 쓰임)
names_yaml = "\n".join(f"  {i}: {c}" for i, c in enumerate(class_codes))
(DEST / "data.yaml").write_text(
    f"path: {DEST}\ntrain: images/val\nval: images/val\nnc: {len(class_codes)}\nnames:\n{names_yaml}\n",
    encoding="utf-8",
)
print(f"\n[완료] val {n_total}장 복원 -> {DEST}")
