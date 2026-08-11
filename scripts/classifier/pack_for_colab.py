"""검토가 끝난 clsdata_v2를 Colab에 올릴 zip 하나로 묶는다.

사람이 폴더를 보며 잘못 들어간 프레임을 지운 뒤에 돌리는 것을 전제로 한다.
그래서 manifest를 믿지 않고 **파일시스템을 다시 훑어** 실제 남아 있는 것만 담고,
manifest도 남은 파일 기준으로 다시 쓴다. 학습 코드(train_binary.py)도 폴더를
훑는 방식이라 지운 파일은 자연히 빠진다.

같은 관로의 프레임이 train과 val에 섞이면 검증이 무의미해지므로, 포장 전에
관로가 양쪽에 걸쳐 있지 않은지 확인한다.

사용:
    python pack_for_colab.py                     # 검사 + zip
    python pack_for_colab.py --check-only        # 검사만
"""

from __future__ import annotations

import argparse
import csv
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val")
LABELS = ("normal", "defect")


def scan(root: Path):
    """실제 남아 있는 파일만 (split, label, pipe, path)로 훑는다."""
    out = []
    for split in SPLITS:
        for label in LABELS:
            d = root / split / label
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.jpg")):
                # 파일명 규칙: `<관로>__<초>s__<거리>m.jpg` 또는 `hwp__<원본이름>.jpg`
                pipe = f.stem.split("__")[1] if f.stem.startswith("hwp__") else f.stem.split("__")[0]
                out.append((split, label, pipe, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v2")
    ap.add_argument("--out", default="")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.data)
    items = scan(root)
    if not items:
        print(f"파일이 없습니다: {root}")
        return

    counts = Counter((s, l) for s, l, _, _ in items)
    print("=== 검토 후 남은 장수 ===")
    for split in SPLITS:
        n, d = counts[(split, "normal")], counts[(split, "defect")]
        print(f"  {split:<6} 정상 {n:>5}  결함 {d:>5}  합계 {n + d:>5}")
    print(f"  총 {len(items):,}장")

    # 관로가 train/val 양쪽에 걸치면 검증 점수가 부풀려진다
    pipes = defaultdict(set)
    for split, _, pipe, _ in items:
        pipes[pipe].add(split)
    straddle = [p for p, s in pipes.items() if len(s) > 1]
    print(f"\n관로 {len(pipes)}개 / train·val 양쪽에 걸친 관로 {len(straddle)}개 (0이어야 정상)")
    for p in straddle:
        print(f"  !! {p}")

    print("\n=== 관로별 (정상/결함) ===")
    per = defaultdict(Counter)
    for split, label, pipe, _ in items:
        per[(split, pipe)][label] += 1
    for (split, pipe), c in sorted(per.items()):
        print(f"  [{split:<5}] {pipe[:34]:<36} 정상 {c['normal']:>4}  결함 {c['defect']:>4}")

    # 한 관로가 정상을 독점하면 그 관로의 특징만 외울 수 있다
    norm_by_pipe = Counter()
    for split, label, pipe, _ in items:
        if split == "train" and label == "normal":
            norm_by_pipe[pipe] += 1
    total_norm = sum(norm_by_pipe.values())
    if total_norm:
        top, n = norm_by_pipe.most_common(1)[0]
        if n / total_norm > 0.35:
            print(f"\n  ! 주의: train 정상의 {n/total_norm:.0%}가 관로 하나({top[:30]})에서 나왔습니다")

    if args.check_only:
        return

    out = Path(args.out) if args.out else root.with_suffix(".zip")
    rows = [{"split": s, "label": l, "pipe": p,
             "dst": str(f.relative_to(root))} for s, l, p, f in items]
    mpath = root / "manifest.csv"
    with open(mpath, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nmanifest 갱신: {mpath} ({len(rows):,}행)")

    t0 = time.time()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:   # JPEG은 재압축 이득이 없다
        for _, _, _, f in items:
            z.write(f, f"clsdata_v2/{f.relative_to(root)}")
        z.write(mpath, "clsdata_v2/manifest.csv")
        train_py = Path(__file__).with_name("train_binary.py")
        if train_py.exists():
            z.write(train_py, "clsdata_v2/train_binary.py")
    print(f"포장 완료: {out} ({out.stat().st_size/1e6:.0f}MB, {time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
