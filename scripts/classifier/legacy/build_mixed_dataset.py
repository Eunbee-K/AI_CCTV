"""실험 C — 현장 데이터(clsdata_v2)에 AIHub/S20을 섞은 학습셋을 만든다.

**섞을 때 반드시 지켜야 하는 것**

v1이 망가진 두 번째 원인은 정상이 전부 AIHub, 결함 16종이 S20 전용이라 모델이
결함이 아니라 **출처**를 보고 갈라도 점수가 나왔던 것이다. 그래서 여기서는
**모든 출처가 정상과 결함을 둘 다 기여**하도록 맞춘다. 한쪽 출처에서 한 클래스만
가져오면 지름길이 다시 열리고, v1보다 나빠진다.

**검증은 건드리지 않는다**

val은 clsdata_v2의 것(현장 관로 5개)을 그대로 복사한다. A·B·C를 같은 잣대로
비교해야 하기 때문이다. AIHub는 train에만 들어간다.

**AIHub는 clsdata_v1의 train에서만 가져온다** — v1의 val을 섞으면 나중에 v1 지표와
비교할 때 헷갈린다.

사용:
    python build_mixed_dataset.py --per-class 3000
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("normal", "defect")


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=r"E:/AI_CCTV_DATASET/clsdata_v2")
    ap.add_argument("--aihub", default=r"E:/AI_CCTV_DATASET/clsdata_v1")
    ap.add_argument("--out", default=r"E:/AI_CCTV_DATASET/clsdata_v3_mixed")
    ap.add_argument("--per-class", type=int, default=3000,
                    help="AIHub에서 클래스당 가져올 장수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict-sources", action="store_true",
                    help="정상·결함을 둘 다 기여하는 출처만 쓴다. val에 여러 출처가 섞일 때만 필요")
    args = ap.parse_args()

    field, aihub, out = Path(args.field), Path(args.aihub), Path(args.out)
    rng = random.Random(args.seed)
    for split in ("train", "val"):
        for lab in LABELS:
            (out / split / lab).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # 1) 현장 데이터는 train·val 모두 그대로 옮긴다
    n_field = Counter()
    for split in ("train", "val"):
        for lab in LABELS:
            for f in sorted((field / split / lab).glob("*.jpg")):
                dst = out / split / lab / f"field__{f.name}"
                shutil.copy2(f, dst)
                pipe = f.stem.split("__")[1] if f.stem.startswith("hwp__") else f.stem.split("__")[0]
                rows.append({"split": split, "label": lab, "source": "field",
                             "code": pipe[:24], "dst": str(dst.relative_to(out))})
                n_field[(split, lab)] += 1

    # 2) AIHub는 train에서만, 정상·결함 **양쪽 다** 같은 양으로 가져온다
    man = [r for r in read_manifest(aihub / "manifest.csv") if r.get("split") == "train"]

    # **출처 단위로** 정상·결함을 둘 다 기여하는 곳만 남긴다.
    # S20/S22에는 정상 폴더가 아예 없어서 결함만 딸려 들어온다. 그러면 모델이
    # "S20처럼 생기면 결함"으로 갈라도 점수가 나온다 — v1을 망가뜨린 그 지름길이다.
    # 클래스 단위 검사만으로는 이걸 못 잡는다(정상·결함이 둘 다 있긴 하므로).
    labels_of: dict[str, set[str]] = defaultdict(set)
    for r in man:
        labels_of[r["source"]].add(r["binary"])
    # 다만 **val이 현장 데이터만으로 되어 있으면** 이 지름길은 val 점수를 부풀릴 수
    # 없다. 채점표에 S20이 없기 때문이다. 실제로 S20을 뺐더니 F1이 0.719 -> 0.634로
    # 가장 크게 떨어졌다 — S20은 실제 하수관 조사 영상에서 뽑은 결함이라 AIHub보다
    # 현장에 가깝고, 빼면 가장 쓸모 있는 결함 예시를 버리게 된다.
    # 그래서 기본은 포함이고, val에 다른 출처가 섞이는 구성으로 바꿀 때만 끈다.
    usable = set(labels_of)
    if args.strict_sources:
        usable = {s for s, labs in labels_of.items() if len(labs) == 2}
    dropped = set(labels_of) - usable
    for s in sorted(dropped):
        print(f"  ! 출처 '{s}' 제외 — {sorted(labels_of[s])}만 있음 (--strict-sources)")
    for s in sorted(set(labels_of) - dropped):
        if len(labels_of[s]) == 1:
            print(f"  · 출처 '{s}' 포함 — {sorted(labels_of[s])}만 있지만 val이 현장 전용이라 무해")

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in man:
        if r["source"] in usable:
            by_label[r["binary"]].append(r)

    n_ai = Counter()
    for lab in LABELS:
        pool = by_label.get(lab, [])
        if not pool:
            print(f"  ! AIHub에 {lab}이 없습니다 — 한쪽만 넣으면 출처 지름길이 열리므로 중단합니다")
            return
        # 코드가 고르게 섞이도록 코드별로 나눠 뽑는다
        per_code: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            per_code[r["code"]].append(r)
        quota = max(1, args.per_class // len(per_code))
        picked: list[dict] = []
        for code, items in sorted(per_code.items()):
            rng.shuffle(items)
            picked.extend(items[:quota])
        rng.shuffle(picked)
        for r in picked[: args.per_class]:
            src = aihub / r["dst"]
            if not src.exists():
                continue
            dst = out / "train" / lab / f"aihub__{r['code']}__{Path(r['dst']).name}"
            shutil.copy2(src, dst)
            rows.append({"split": "train", "label": lab, "source": r["source"],
                         "code": r["code"], "dst": str(dst.relative_to(out))})
            n_ai[lab] += 1

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("=== 구성 ===")
    for split in ("train", "val"):
        for lab in LABELS:
            fld = n_field[(split, lab)]
            ai = n_ai[lab] if split == "train" else 0
            print(f"  {split:<6} {lab:<7} 현장 {fld:>5}  AIHub {ai:>5}  합계 {fld + ai:>5}")
    tr_f = n_field[("train", "normal")] + n_field[("train", "defect")]
    tr_a = n_ai["normal"] + n_ai["defect"]
    print(f"\n  train에서 현장이 차지하는 비율 {tr_f / max(tr_f + tr_a, 1):.0%}")
    print("  (양쪽 출처 모두 정상·결함을 기여 → 출처만으로는 구분 불가)")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
