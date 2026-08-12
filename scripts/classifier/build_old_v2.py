"""노후관로 필터 v2 — 야장에 AIHub 정상을 보강한다.

**왜 보강하나**

v1(야장 9,471장, 야장 val AUC 0.9449)을 AIHub에 돌려보니 결함은 잘 잡는데
(BK 96.7% · CL 93.3% · DS 88.3% · ETC 85.7%) 정상에서 무너진다:

    PJ (정상 이음부)   67.7%
    IN (정상 관 내부)  21.3%   <- 거의 전부 결함이라고 답한다

원인은 야장 정상의 92%가 `이음부(접합부)존재`, 즉 **이음부가 보이는 프레임**이라는
것이다. 이음부 없이 밋밋한 관 내부를 정상으로 본 적이 없다. 실제 영상은 이음부
사이의 밋밋한 구간이 훨씬 길기 때문에, 이대로 두면 그 구간을 통째로 결함이라 할 수 있다.
야장 val로는 안 보이는 약점이다 — 야장 정상이 전부 이음부 프레임이라서.

**그래서 정상 위주로 보강한다.** 결함은 이미 86~97%라 균형 유지에 필요한 만큼만 넣는다.

**AIHub를 너무 많이 넣으면 안 된다.** 이 프로젝트에서 AIHub 78,388장으로 학습한
모델의 현장 AUC는 0.516(무작위)이었다. 양이 아니라 도메인이 성능을 결정했다.
야장이 학습의 절반 아래로 내려가지 않는 선을 지킨다.

**AIHub val을 따로 뗀다.** 학습 후 야장 성능이 유지되면서 AIHub 성능이 올랐는지
양쪽을 봐야, 한쪽을 얻고 다른 쪽을 잃는 건 아닌지 알 수 있다.

사용:
    python build_old_v2.py
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

# (코드, 라벨, 학습에 넣을 장수)
PLAN = [
    ("IN", "normal", 4000),    # 지금 21.3%밖에 못 맞히는 공백 — 최우선
    ("PJ", "normal", 3000),    # 67.7% -> 개선
    ("ETC", "defect", 1000),   # 결함은 이미 잘 잡으니 균형용 최소한
    ("BK", "defect", 800),
    ("CL", "defect", 700),
    ("DS", "defect", 500),
]
VAL_PER_CODE = 200   # 학습에 넣지 않고 따로 평가용으로 뗀다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DS / "clsdata_old_v1"))
    ap.add_argument("--out", default=str(DS / "clsdata_old_v2"))
    ap.add_argument("--aihub-val", default=str(DS / "aihub_holdout"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base, out, hold = Path(args.base), Path(args.out), Path(args.aihub_val)
    if out.exists():
        shutil.rmtree(out)
    print(f"야장 복사: {base.name} -> {out.name}")
    shutil.copytree(base, out)
    for lab in ("normal", "defect"):
        (hold / "val" / lab).mkdir(parents=True, exist_ok=True)

    with open(out / "manifest.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])

    rng = random.Random(args.seed)
    added, held = Counter(), Counter()
    for code, label, n_train in PLAN:
        files = sorted((AIHUB / code).iterdir())
        rng.shuffle(files)
        need = n_train + VAL_PER_CODE
        for i, f in enumerate(files[:need]):
            # 앞 VAL_PER_CODE장은 평가용으로 빼둔다(학습에 안 들어간다)
            to_val = i < VAL_PER_CODE
            dst = ((hold / "val" / label) if to_val
                   else (out / "train" / label)) / f"aihub_{code}_{f.stem}.jpg"
            try:
                Image.open(f).convert("RGB").resize(
                    (args.size, args.size), Image.BILINEAR).save(dst, "JPEG", quality=88)
            except Exception:
                continue
            if to_val:
                held[(code, label)] += 1
                continue
            r = {k: "" for k in fields}
            r.update({"split": "train", "label": label, "area": "aihub",
                      "code": f"aihub_{code}", "name": code,
                      "dst": f"train\\{label}\\{dst.name}"})
            rows.append(r)
            added[(code, label)] += 1
        print(f"  {code:<4} {label:<7} 학습 {added[(code,label)]:>5}  평가용 {held[(code,label)]:>4}")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n=== 최종 구성 ===")
    for split in ("train", "val"):
        line = []
        for lab in ("normal", "defect"):
            line.append(f"{lab} {len(list((out / split / lab).glob('*.jpg'))):>5}")
        print(f"  {split:<6} " + "  ".join(line))
    ya = sum(1 for r in rows if r["split"] == "train" and r["area"] != "aihub")
    tr = sum(1 for r in rows if r["split"] == "train")
    print(f"\n  train {tr:,}장 중 야장 {ya:,}장 ({ya/tr:.0%})")
    print(f"  AIHub 평가용(학습 미포함) {sum(held.values()):,}장 -> {hold}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
