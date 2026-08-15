"""이미 만들어진 clsdata_v1의 train/val 분할만 다시 계산한다.

build_binary_dataset.py 초판이 (정상/결함, 코드)별로 따로 분할해서, 영상 하나가
여러 결함 코드에 걸치면 같은 관로가 train과 val 양쪽에 들어갔다(val의 5.6%가 오염).
이미지는 그대로 두고 manifest와 파일 위치만 고쳐 70분짜리 재변환을 피한다.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_binary_dataset import plan_splits, resolve_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v1")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    manifest = root / "manifest.csv"
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"manifest {len(rows):,}행")

    raw = [
        (
            r["video"] or f"__file__:{r['src']}",
            (r["binary"], r["code"]),
            f"__file__:{r['src']}",
        )
        for r in rows
    ]
    groups = resolve_groups(raw)
    plan = plan_splits(list(zip(groups, (t[1] for t in raw))), args.val_ratio, random.Random(args.seed))

    moves, stat = [], Counter()
    for r, g in zip(rows, groups):
        new = plan[g]
        stat[(new, r["binary"])] += 1
        if new != r["split"]:
            old_dst = Path(r["dst"])
            new_dst = Path(new) / r["binary"] / old_dst.name
            moves.append((root / old_dst, root / new_dst))
            r["split"], r["dst"] = new, str(new_dst)

    print(f"\n이동 대상 {len(moves):,}장")
    for split in ("train", "val"):
        n = stat[(split, "normal")], stat[(split, "defect")]
        print(f"  {split:<6} normal {n[0]:>6}  defect {n[1]:>6}  합계 {sum(n):>6}")

    tr = {r["video"] for r in rows if r["video"] and r["split"] == "train"}
    va = {r["video"] for r in rows if r["video"] and r["split"] == "val"}
    print(f"\n소스영상 train {len(tr)} / val {len(va)} / 겹침 {len(tr & va)}")

    by_code = defaultdict(Counter)
    for r in rows:
        by_code[(r["binary"], r["code"])][r["split"]] += 1
    print("\n=== 코드별 (train/val) ===")
    for (b, code), c in sorted(by_code.items()):
        flag = "  <- val 없음" if c["val"] == 0 else ""
        print(f"  {b:<7}{code:<11} train {c['train']:>6}  val {c['val']:>5}{flag}")

    if args.dry_run:
        print("\n(dry-run — 파일 이동 안 함)")
        return

    for split in ("train", "val"):
        for b in ("normal", "defect"):
            (root / split / b).mkdir(parents=True, exist_ok=True)
    for src, dst in moves:
        shutil.move(str(src), str(dst))
    print(f"\n{len(moves):,}장 이동 완료")

    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"manifest 갱신: {manifest}")


if __name__ == "__main__":
    main()
