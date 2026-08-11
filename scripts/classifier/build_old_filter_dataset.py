"""노후관로 필터 학습셋 — 야장 PDF에서 뽑은 프레임으로 만든다.

조사원이 프레임마다 `(관로)…` 또는 `(상태)…`로 직접 기록해둔 것이라, 앞서 현장 영상에서
"결함 위치에서 3m 이상 떨어졌으니 정상일 것"이라고 추정하던 것과 라벨의 질이 다르다.

**분할은 관로 단위로** 한다. 같은 관로의 프레임이 train과 val에 섞이면 검증 점수가
부풀려진다. 관로가 572개라 넉넉하게 나눌 수 있다.

**지역(면)은 섞는다.** 3개 면이 다 train에 들어가야 "정상이 어떻게 생겼는지"의 폭을
배운다. 대신 val에도 3개 면이 고루 들어가도록 면별로 나눈다.

사용:
    python build_old_filter_dataset.py --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SRC = Path(r"E:/AI_CCTV_DATASET/yajang_frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"E:/AI_CCTV_DATASET/clsdata_old_v1")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=0.2, help="검증으로 뺄 관로 비율")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    for s in ("train", "val"):
        for l in ("normal", "defect"):
            (out / s / l).mkdir(parents=True, exist_ok=True)

    with open(SRC / "yajang.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # 면별로 관로를 나눠, val에도 세 지역이 고루 들어가게 한다
    by_area: dict[str, set] = defaultdict(set)
    for r in rows:
        by_area[r["area"]].add(r["pipe"])

    rng = random.Random(args.seed)
    val_pipes: set[tuple[str, str]] = set()
    for area, pipes in sorted(by_area.items()):
        ps = sorted(pipes)
        rng.shuffle(ps)
        k = max(1, int(len(ps) * args.val_ratio))
        val_pipes |= {(area, p) for p in ps[:k]}
        print(f"  {area:<5} 관로 {len(ps):>4}개 중 val {k}개")

    man, stat = [], Counter()
    for r in rows:
        split = "val" if (r["area"], r["pipe"]) in val_pipes else "train"
        src = SRC / "images" / r["file"]
        if not src.exists():
            continue
        dst = out / split / r["label"] / r["file"]
        try:
            Image.open(src).convert("RGB").resize(
                (args.size, args.size), Image.BILINEAR).save(dst, "JPEG", quality=88)
        except Exception:
            stat["읽기 실패"] += 1
            continue
        man.append({"split": split, "label": r["label"], "area": r["area"],
                    "code": f"{r['area']}_{r['pipe']}", "name": r["name"],
                    "grade": r["grade"], "dst": f"{split}\\{r['label']}\\{r['file']}"})
        stat[(split, r["label"])] += 1

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0]))
        w.writeheader()
        w.writerows(man)

    print("\n=== 구성 ===")
    for s in ("train", "val"):
        n, d = stat[(s, "normal")], stat[(s, "defect")]
        print(f"  {s:<6} 정상 {n:>6}  결함 {d:>6}  합계 {n + d:>6}")
    print(f"\n  관로 {len({(r['area'], r['pipe']) for r in rows})}개 "
          f"(val {len(val_pipes)}개) / 총 {len(man):,}장")
    if stat["읽기 실패"]:
        print(f"  읽기 실패 {stat['읽기 실패']}장")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
