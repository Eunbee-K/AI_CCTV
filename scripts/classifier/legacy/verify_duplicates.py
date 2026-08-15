"""dHash 거리 0으로 잡힌 쌍이 진짜 중복인지, 저대비 이미지의 해시 충돌인지 가린다.

dHash는 64비트라 어두운 관 내부처럼 대비가 낮은 이미지에서는 서로 다른 장면도
같은 해시가 나올 수 있다. 실제 픽셀을 대조해야 결론이 난다.

64x64 회색조 평균절대오차(MAE)로 판정:
  MAE < 12  -> 사실상 같은 장면 (진짜 중복)
  MAE >= 12 -> 다른 장면인데 해시만 같음 (충돌)
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from check_duplicates import POPCOUNT, dhash


def small(path, size=64):
    return np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v1")
    ap.add_argument("--codes", default="IN,PJ,CL,CC,DS")
    ap.add_argument("--sample", type=int, default=25, help="코드당 검사할 거리0 쌍 수")
    ap.add_argument("--mae", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.data)
    with open(root / "manifest.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (root / r["dst"]).exists()]

    cache = root / "dhash_cache.csv"
    if cache.exists():
        print(f"해시 캐시 사용: {cache}")
        with open(cache, encoding="utf-8") as f:
            table = {r["dst"]: int(r["h"]) for r in csv.DictReader(f)}
        hashes = [table[r["dst"]] for r in rows]
    else:
        from concurrent.futures import ProcessPoolExecutor

        print(f"{len(rows):,}장 해시 계산 중...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            hashes = list(ex.map(dhash, [str(root / r["dst"]) for r in rows], chunksize=256))
        with open(cache, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dst", "h"])
            w.writerows([[r["dst"], h] for r, h in zip(rows, hashes)])
        print(f"해시 캐시 저장: {cache}")

    tr = [(r, h) for r, h in zip(rows, hashes) if r["source"] == "aihub" and r["split"] == "train" and h >= 0]
    va = [(r, h) for r, h in zip(rows, hashes) if r["source"] == "aihub" and r["split"] == "val" and h >= 0]
    tr_bytes = np.array([h for _, h in tr], dtype=np.uint64).view(np.uint8).reshape(-1, 8)

    rng = random.Random(0)
    for code in args.codes.split(","):
        cand = [(r, h) for r, h in va if r["code"] == code]
        rng.shuffle(cand)
        dup, coll = [], []
        for r, h in cand:
            d = POPCOUNT[np.bitwise_xor(np.array([h], dtype=np.uint64).view(np.uint8), tr_bytes)].sum(1)
            j = int(d.argmin())
            if d[j] != 0:
                continue
            mae = float(np.abs(small(root / r["dst"]) - small(root / tr[j][0]["dst"])).mean())
            (dup if mae < args.mae else coll).append((mae, r["dst"], tr[j][0]["dst"]))
            if len(dup) + len(coll) >= args.sample:
                break
        n = len(dup) + len(coll)
        if not n:
            print(f"=== {code}: 거리0 쌍 없음 ===")
            continue
        print(f"\n=== {code}: 거리0 쌍 {n}개 검사 ===")
        print(f"  진짜 중복(MAE<{args.mae:g}): {len(dup)} ({len(dup)/n:.0%})   해시 충돌: {len(coll)} ({len(coll)/n:.0%})")
        for m, x, y in sorted(dup)[:3]:
            print(f"    중복 MAE {m:6.1f}  {Path(x).name} <-> {Path(y).name}")
        for m, x, y in sorted(coll, reverse=True)[:2]:
            print(f"    충돌 MAE {m:6.1f}  {Path(x).name} <-> {Path(y).name}")


if __name__ == "__main__":
    main()
