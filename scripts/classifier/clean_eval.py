"""train에 근접 중복이 있는 val 이미지를 제외하고 다시 채점한다.

AIHub는 소스 영상 정보가 없어 파일 단위로 무작위 분할했고, 실제로 같은 장면이
train과 val 양쪽에 들어갔다(dHash 거리 0 쌍의 52~60%가 픽셀상 동일 수준).
그 이미지들의 점수는 "학습 때 본 것을 맞힌" 결과이므로 빼고 봐야 실력이 보인다.

eval_binary.py가 남긴 val_probs.csv와 dhash_cache.csv를 재사용하므로 재추론이 없다.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from check_duplicates import POPCOUNT
from eval_binary import roc_auc


def metrics(pos, neg, thr=0.5):
    tp = sum(p >= thr for p in pos)
    fp = sum(p >= thr for p in neg)
    rec = tp / max(len(pos), 1)
    spec = (len(neg) - fp) / max(len(neg), 1)
    prec = tp / max(tp + fp, 1)
    return {
        "n_pos": len(pos), "n_neg": len(neg),
        "recall": rec, "specificity": spec, "precision": prec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "miss": len(pos) - tp, "fp": fp,
        "auc": roc_auc(pos, neg) if pos and neg else float("nan"),
    }


def show(title, m):
    print(f"\n  [{title}]  결함 {m['n_pos']:,} / 정상 {m['n_neg']:,}")
    if not m["n_pos"] or not m["n_neg"]:
        print("    표본 부족")
        return
    print(f"    재현율 {m['recall']:.4f} (놓침 {m['miss']}장)   "
          f"정상 오탐 {m['fp']}장 ({1-m['specificity']:.2%})   "
          f"F1 {m['f1']:.4f}   AUC {m['auc']:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v1")
    ap.add_argument("--probs", default=r"E:/AI_CCTV_RESULTS/filter/val_probs.csv")
    args = ap.parse_args()

    root = Path(args.data)
    with open(args.probs, encoding="utf-8") as f:
        val = list(csv.DictReader(f))
    with open(root / "dhash_cache.csv", encoding="utf-8") as f:
        hashes = {r["dst"]: int(r["h"]) for r in csv.DictReader(f)}
    with open(root / "manifest.csv", encoding="utf-8") as f:
        train = [r for r in csv.DictReader(f) if r["split"] == "train"]

    tr_h = np.array([hashes[r["dst"]] for r in train if hashes.get(r["dst"], -1) >= 0], dtype=np.uint64)
    tr_b = tr_h.view(np.uint8).reshape(-1, 8)
    print(f"train {len(tr_h):,} / val {len(val):,}")

    # dHash는 부호 없는 64비트라 int64에 -1 센티넬을 넣을 수 없다. 유효성을 따로 둔다.
    valid = np.array([hashes.get(r["dst"], -1) >= 0 for r in val])
    va_h = np.array([hashes[r["dst"]] for r, v in zip(val, valid) if v], dtype=np.uint64)
    best = np.full(len(val), 64, dtype=np.uint8)
    vb = va_h.view(np.uint8).reshape(-1, 8)
    out = np.empty(len(vb), dtype=np.uint8)
    for i in range(0, len(vb), 512):
        blk = vb[i : i + 512].reshape(-1, 1, 8)
        out[i : i + 512] = POPCOUNT[np.bitwise_xor(blk, tr_b[None, :, :])].sum(axis=2).min(axis=1)
    best[valid] = out

    for r, d in zip(val, best):
        r["dist"] = int(d)
        r["p"] = float(r["prob"])

    print("\n" + "=" * 70)
    print("근접 중복 제외 기준별 성능")
    print("=" * 70)
    for label, keep in (
        ("① 전체 (기존 보고값)", lambda r: True),
        ("② 거리 0 제외", lambda r: r["dist"] > 0),
        ("③ 거리 <=5 제외 (권장)", lambda r: r["dist"] > 5),
        ("④ 거리 <=10 제외 (가장 보수적)", lambda r: r["dist"] > 10),
    ):
        sub = [r for r in val if keep(r)]
        pos = [r["p"] for r in sub if r["binary"] == "defect"]
        neg = [r["p"] for r in sub if r["binary"] == "normal"]
        print(f"\n{label}   남은 val {len(sub):,}장 ({len(sub)/len(val):.0%})")
        show("전체", metrics(pos, neg))
        ap_ = [r["p"] for r in sub if r["binary"] == "defect" and r["source"] == "aihub"]
        inside = [r["p"] for r in sub if r["binary"] == "normal" and r["code"] in ("IN", "PJ")]
        show("AIHub 결함 vs 관 안 정상 (가장 엄격)", metrics(ap_, inside))

    print("\n" + "=" * 70)
    print("거리 >5 조건에서 결함 코드별")
    print("=" * 70)
    by_code = defaultdict(list)
    src = {}
    for r in val:
        if r["binary"] == "defect" and r["dist"] > 5:
            by_code[r["code"]].append(r["p"])
            src[r["code"]] = r["source"]
    print("  코드   출처    남은장수  재현율  놓침")
    for code, ps in sorted(by_code.items(), key=lambda kv: sum(p >= 0.5 for p in kv[1]) / len(kv[1])):
        rec = sum(p >= 0.5 for p in ps) / len(ps)
        print(f"  {code:<6} {src[code]:<6} {len(ps):>8}  {rec:>6.3f}  {len(ps)-sum(p>=0.5 for p in ps):>4}장")


if __name__ == "__main__":
    main()
