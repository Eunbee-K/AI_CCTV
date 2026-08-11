"""val 점수가 중복 프레임 때문에 부풀려졌는지 검사한다.

S20은 파일명에 소스 영상이 있어 영상 단위로 분할했지만, AIHub(78,388장 중 63,762장)는
영상 정보가 없어 파일 단위로 무작위 분할했다. AIHub가 한 영상에서 여러 프레임을 뽑아
만든 데이터라면 같은 장면이 train과 val에 나뉘어 들어갔을 수 있고, 그러면 val 점수는
"기억한 것"을 맞힌 결과가 된다.

dHash(64비트 지각 해시)로 val 이미지마다 train 쪽 최근접 거리를 구해 확인한다.
거리 0 = 사실상 동일, <=5 = 같은 장면의 다른 프레임 수준.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: str) -> int:
    """가로 인접 픽셀의 대소 관계를 64비트로. 밝기/압축 차이에 둔감하다."""
    try:
        im = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    except OSError:
        return -1
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming_min(val_h: np.ndarray, train_h: np.ndarray, chunk: int = 512):
    """val 각 항목에 대해 train 전체와의 최소 해밍거리."""
    best = np.full(len(val_h), 64, dtype=np.uint8)
    tr = train_h.view(np.uint8).reshape(-1, 8)
    for i in range(0, len(val_h), chunk):
        block = val_h[i : i + chunk].view(np.uint8).reshape(-1, 1, 8)
        d = POPCOUNT[np.bitwise_xor(block, tr[None, :, :])].sum(axis=2)
        best[i : i + chunk] = d.min(axis=1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--near", type=int, default=5, help="이 거리 이하를 근접 중복으로 본다")
    args = ap.parse_args()

    root = Path(args.data)
    with open(root / "manifest.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (root / r["dst"]).exists()]
    print(f"{len(rows):,}장 해시 계산 중...")

    paths = [str(root / r["dst"]) for r in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        hashes = list(ex.map(dhash, paths, chunksize=256))

    ok = [(r, h) for r, h in zip(rows, hashes) if h >= 0]
    print(f"  해시 완료 {len(ok):,}장 (실패 {len(rows) - len(ok)})")

    # 출처별로 나눠서 본다. AIHub만 영상 단위 분할을 못 했다.
    for source_label, keep in (("AIHub", lambda r: r["source"] == "aihub"),
                               ("S20/S22", lambda r: r["source"] != "aihub")):
        tr = np.array([h for r, h in ok if keep(r) and r["split"] == "train"], dtype=np.uint64)
        va = [(r, h) for r, h in ok if keep(r) and r["split"] == "val"]
        if not len(tr) or not va:
            continue
        va_h = np.array([h for _, h in va], dtype=np.uint64)
        print(f"\n=== {source_label}: val {len(va):,} vs train {len(tr):,} ===")
        best = hamming_min(va_h, tr)
        dist = Counter(int(b) for b in best)
        exact = dist.get(0, 0)
        near = sum(v for k, v in dist.items() if k <= args.near)
        print(f"  거리 0 (사실상 동일)      : {exact:>5,}장 ({exact/len(va):.1%})")
        print(f"  거리 <={args.near} (같은 장면 수준): {near:>5,}장 ({near/len(va):.1%})")
        print("  거리 분포:", dict(sorted(dist.items())[:12]))

        by_code = defaultdict(lambda: [0, 0])
        for (r, _), b in zip(va, best):
            by_code[r["code"]][1] += 1
            by_code[r["code"]][0] += int(b) <= args.near
        worst = sorted(by_code.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:8]
        print("  근접중복 비율 높은 코드:",
              ", ".join(f"{c} {n/t:.0%}({n}/{t})" for c, (n, t) in worst))


if __name__ == "__main__":
    main()
