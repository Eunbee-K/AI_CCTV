"""신설관로 필터 학습셋.

**노후 필터에서 통한 원칙을 그대로 적용한다.**

노후 필터는 야장 데이터만으로 AUC 0.9449였는데, AIHub 정상(IN·PJ)을 보강하자
AIHub IN 정답률이 41% -> 100%로 뒤집히면서 야장 AUC도 0.9488로 올랐다.
"이음부 없이 밋밋한 관 내부도 정상"이라는 개념이 빠져 있었던 것이다.

그때 지킨 두 가지를 여기서도 지킨다.

1. **OUT_*(맨홀·인버트·자동차)은 넣지 않는다.** 관 밖 장면이라 너무 쉬워서,
   모델이 "관 밖 = 정상"만 배우고 관 내부 판별은 못 배운다. 첫 필터(AIHub 78,388장,
   현장 AUC 0.516)가 정확히 그렇게 망가졌다. 기존 clsdata_v4_newpipe는 AIHub 정상
   1,500장 중 900장이 OUT_*이라 이 함정에 다시 빠져 있었다.
2. **AIHub가 현장 데이터를 압도하지 않게 한다.** 양이 아니라 도메인이 성능을 정한다.

**val은 clsdata_v2의 것을 그대로 쓴다** — 앞선 A~E 실험(최고 AUC 0.790)과 같은
잣대로 비교해야 개선 여부를 알 수 있다.

신설 현장 데이터는 두 곳에서 온다:
  clsdata_v2        가평 상면 정비공사 (우수관)
  fieldset_v1_label 오수관 — 관로 22개, 지역 코드 6종이라 현장 다양성을 채워준다

사용:
    python build_new_filter.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from paths import DATASET  # noqa: E402

from PIL import Image

DS = DATASET
AIHUB = DS / "original/aihub_data_bbox/image"
FIELDSET = DS / "fieldset_v1_label"

# OUT_*은 의도적으로 뺐다(위 설명 참고). 결함은 균형용 최소한만.
PLAN = [
    ("IN", "normal", 1200),
    ("PJ", "normal", 900),
    ("ETC", "defect", 300),
    ("BK", "defect", 250),
    ("CL", "defect", 200),
    ("DS", "defect", 150),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DS / "clsdata_v2"))
    ap.add_argument("--out", default=str(DS / "clsdata_new_v1"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    if not base.exists():
        print(f"!! {base} 가 없습니다. clsdata_v2.zip을 먼저 풀어주세요.")
        return
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        for l in ("normal", "defect"):
            (out / s / l).mkdir(parents=True, exist_ok=True)

    rows, stat = [], Counter()

    # 1) 신설 우수관(상면) — train/val 그대로
    for split in ("train", "val"):
        for lab in ("normal", "defect"):
            for f in sorted((base / split / lab).glob("*.jpg")):
                shutil.copy2(f, out / split / lab / f"sm__{f.name}")
                pipe = f.stem.split("__")[1] if f.stem.startswith("hwp__") else f.stem.split("__")[0]
                rows.append({"split": split, "label": lab, "source": "상면(우수)",
                             "code": pipe[:24], "dst": f"{split}\\{lab}\\sm__{f.name}"})
                stat[(split, lab, "상면(우수)")] += 1

    # 2) 신설 오수관(fieldset) — train에만. val은 A~E와 비교하려면 건드리면 안 된다.
    for split in ("train", "val"):
        d = FIELDSET / "images" / split
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() != ".jpg":
                continue
            parts = f.stem.split("__")
            tag = parts[1] if len(parts) > 1 else "?"
            lab = "normal" if tag == "NORMAL" else "defect"
            lp = FIELDSET / "labels" / split / f"{f.stem}.txt"
            n_box = 0
            if lp.exists():
                n_box = sum(1 for x in lp.read_text(encoding="utf-8").splitlines()
                            if len(x.split()) >= 5)
            if lab == "normal" and n_box:
                stat["제외(정상인데 박스 있음)"] += 1
                continue
            dst = out / "train" / lab / f"fs__{f.name}"
            Image.open(f).convert("RGB").resize(
                (args.size, args.size), Image.BILINEAR).save(dst, "JPEG", quality=88)
            rows.append({"split": "train", "label": lab, "source": "fieldset(오수)",
                         "code": parts[0][:24], "dst": f"train\\{lab}\\fs__{f.name}"})
            stat[("train", lab, "fieldset(오수)")] += 1

    # 3) AIHub 보강 — IN·PJ 위주, OUT_*은 제외
    rng = random.Random(args.seed)
    for code, lab, n in PLAN:
        files = sorted((AIHUB / code).iterdir())
        rng.shuffle(files)
        for f in files[:n]:
            dst = out / "train" / lab / f"aihub_{code}_{f.stem}.jpg"
            try:
                Image.open(f).convert("RGB").resize(
                    (args.size, args.size), Image.BILINEAR).save(dst, "JPEG", quality=88)
            except Exception:
                continue
            rows.append({"split": "train", "label": lab, "source": "aihub",
                         "code": f"aihub_{code}", "dst": f"train\\{lab}\\{dst.name}"})
            stat[("train", lab, "aihub")] += 1

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("=== 구성 ===")
    for k in sorted(k for k in stat if isinstance(k, tuple)):
        print(f"  {k[0]:<6} {k[1]:<7} {k[2]:<15} {stat[k]:>5}")
    tr = sum(1 for r in rows if r["split"] == "train")
    fld = sum(1 for r in rows if r["split"] == "train" and r["source"] != "aihub")
    print(f"\n  train {tr:,}장 중 신설 현장 {fld:,}장 ({fld/tr:.0%})")
    for s in ("train", "val"):
        n = len(list((out / s / "normal").glob("*.jpg")))
        d = len(list((out / s / "defect").glob("*.jpg")))
        print(f"  {s:<6} 정상 {n:>5}  결함 {d:>5}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
