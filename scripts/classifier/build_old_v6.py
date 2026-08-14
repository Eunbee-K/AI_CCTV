"""OLD_v6 — v3를 그대로 두고 부족한 것만 채운다. v3 가중치에서 파인튜닝할 데이터.

v5에서 야장을 학습에서 빼봤는데 결과가 엉망이었다. 야장(현장 프레임)이 빠지면
학습 데이터가 24,302 -> 5,260으로 줄고 현장 도메인이 통째로 사라진다.
**야장은 학습에 반드시 들어가야 한다.** v6는 v3를 기반으로 삼는다.

v3에서 채울 두 가지

1. **관 밖 정상이 한 장도 없다.** 정지 이미지 테스트셋에서 필터가 관 밖 40장을
   100% 결함으로 오판했다(IN 0.0158 / PJ 0.0187 vs OUT_* 0.9762). 조사 영상은
   늘 관 밖에서 시작하므로 그 구간이 상위 순위를 차지해 진짜 결함을 밀어낸다.
   AIHub OUT_MH/OUT_CAR/OUT_INVERT를 정상으로 넣는다.

2. **결함 종류별 편차가 크다.** JS 1,093 · DS 1,451인데 CC 384 · CM 381 ·
   HL 398 · JD 351 · LP 327 · LS 336 · PO 315 · SD 377이다. 원본에 여유가 있는
   종류는 MIN_PER_CODE까지 올린다. 원본 자체가 적은 것(CX 14 · DE 34 · NS 4 ·
   DF 285 · DG 303 · IF 139)은 어쩔 수 없다.

**v3에 이미 쓴 파일은 다시 넣지 않는다.** val(야장 2,580장)은 그대로 둔다 —
v3와 같은 판이어야 성능을 비교할 수 있다.

사용:
    python build_old_v6.py
    python build_old_v6.py --min-per-code 800 --out-ratio 0.20
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))              # scripts/classifier/
from paths import DATASET, FILTER_DATA  # noqa: E402
from build_v5_and_valset import interleave, rmtree_retry, scan_all  # noqa: E402

V3 = FILTER_DATA / "clsdata_old_v3"
OUT_CODES = ("OUT_MH", "OUT_CAR", "OUT_INVERT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-code", type=int, default=400,
                    help="결함 종류당 목표 장수. 많으면 줄이고 적으면 채운다. "
                         "400이면 25종 중 19종이 정확히 채워진다")
    ap.add_argument("--out-ratio", type=float, default=0.20,
                    help="정상 중 관 밖이 차지할 비율")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    out = FILTER_DATA / "clsdata_old_v6"
    rmtree_retry(out)
    for split in ("train", "val"):
        for lab in ("normal", "defect"):
            (out / split / lab).mkdir(parents=True)

    # ── 1) v3 목록을 읽는다 (아직 복사하지 않는다) ──
    # 결함은 종류당 PER_CODE로 **맞춰야** 하므로, 많은 종류는 줄이고 적은 종류는
    # 채운다. 그러려면 먼저 전체를 코드별로 모아놓고 정해야 한다.
    v3_def = defaultdict(list)          # 코드 -> [(경로, 출처)]
    v3_other = []                       # 정상 + val 전부 (그대로 옮긴다)
    used_names = set()
    with open(V3 / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            src = V3 / r["dst"].replace("\\", "/")
            if not src.exists():
                continue
            used_names.add(src.name)
            if r["split"] == "train" and r["label"] == "defect":
                v3_def[code_of_row(r)].append((src, r))
            else:
                v3_other.append((src, r))

    rows = []
    for src, r in v3_other:             # 정상·val은 손대지 않는다
        dst = out / r["split"] / r["label"] / src.name
        shutil.copy2(src, dst)
        rows.append({**r, "dst": f"{r['split']}\\{r['label']}\\{src.name}"})
    n_tr_nor = sum(1 for r in rows if r["split"] == "train" and r["label"] == "normal")
    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"v3 이식: train 정상 {n_tr_nor:,} · val {n_val:,}")
    print(f"v3 결함 {sum(len(v) for v in v3_def.values()):,}장 ({len(v3_def)}종)\n")

    pool = scan_all()

    # ── 2) 결함을 종류당 PER_CODE로 맞춘다 ──
    # **많은 쪽은 줄이고 적은 쪽은 채운다.** v3는 JS 1,093 · DS 1,451인데 CC 384 ·
    # LP 327처럼 편차가 3~4배였다. 편차가 크면 가중 샘플링으로도 완전히 상쇄되지
    # 않아 흔한 결함만 잘 잡는 모델이 된다.
    print(f"결함을 종류당 {args.per_code}장으로 맞춘다")
    print(f"  {'코드':<7}{'v3':>7}{'조정':>7}{'최종':>7}")
    n_def = 0
    for code in sorted(set(v3_def) | (set(pool) - set(OUT_CODES) - {"IN", "PJ"})):
        cur = v3_def.get(code, [])
        random.shuffle(cur)
        take_v3 = cur[:args.per_code]
        for src, r in take_v3:
            dst = out / "train" / "defect" / src.name
            shutil.copy2(src, dst)
            rows.append({**r, "dst": f"train\\defect\\{src.name}"})

        need = args.per_code - len(take_v3)
        extra = []
        if need > 0:
            cand = [x for x in interleave(pool.get(code, []), args.seed)
                    if x[0].name not in used_names]
            extra = cand[:need]
            for f, tag in extra:
                dst = out / "train" / "defect" / f"add_{tag}_{code}__{f.name}"
                shutil.copy2(f, dst)
                rows.append({"split": "train", "label": "defect", "area": tag,
                             "code": f"{tag}_{code}", "name": "", "grade": "",
                             "dst": f"train\\defect\\{dst.name}"})
        final = len(take_v3) + len(extra)
        n_def += final
        delta = final - len(cur)
        mark = "" if final == args.per_code else "  <- 원본 부족"
        print(f"  {code:<7}{len(cur):>7}{delta:>+7}{final:>7}{mark}")
    print(f"  결함 합계 {n_def:,}장\n")

    # ── 3) 관 밖 정상 추가 ──
    # 결함이 늘었으니 정상도 1:1을 유지하도록 맞추고, 그중 out_ratio만큼을 관 밖으로.
    total_def = n_def
    want_out = round(total_def * args.out_ratio)
    per_out = want_out // len(OUT_CODES)
    print(f"관 밖 정상 추가 (정상의 {args.out_ratio:.0%} 목표 = {want_out:,}장)")
    added_out = 0
    for code in OUT_CODES:
        cand = [x for x in interleave(pool.get(code, []), args.seed)
                if x[0].name not in used_names]
        take = cand[:per_out]
        for f, tag in take:
            dst = out / "train" / "normal" / f"add_{tag}_{code}__{f.name}"
            shutil.copy2(f, dst)
            rows.append({"split": "train", "label": "normal", "area": tag,
                         "code": f"{tag}_{code}", "name": "", "grade": "",
                         "dst": f"train\\normal\\{dst.name}"})
        added_out += len(take)
        print(f"  {code:<12}{len(take):>7}")

    # 결함이 정상보다 많아졌으면 관 안쪽 정상(IN/PJ)으로 균형을 맞춘다
    n_nor = n_tr_nor + added_out
    gap = total_def - n_nor
    if gap > 0:
        print(f"\n정상이 {gap:,}장 모자라 IN/PJ로 채운다")
        for code, share in (("PJ", 0.6), ("IN", 0.4)):
            cand = [x for x in interleave(pool.get(code, []), args.seed + 1)
                    if x[0].name not in used_names]
            take = cand[:round(gap * share)]
            for f, tag in take:
                dst = out / "train" / "normal" / f"add_{tag}_{code}__{f.name}"
                shutil.copy2(f, dst)
                rows.append({"split": "train", "label": "normal", "area": tag,
                             "code": f"{tag}_{code}", "name": "", "grade": "",
                             "dst": f"train\\normal\\{dst.name}"})
            n_nor += len(take)
            print(f"  {code:<12}{len(take):>7}")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["split", "label", "area", "code", "name", "grade", "dst"])
        w.writeheader()
        w.writerows(rows)

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    d = sum(1 for r in tr if r["label"] == "defect")
    n = sum(1 for r in tr if r["label"] == "normal")
    print(f"\n학습셋 -> {out}")
    print(f"  train  결함 {d:,} · 정상 {n:,}  (1:{n / max(d, 1):.2f})")
    print(f"  val    {len(va):,}  ← v3와 동일 (야장)")


def code_of_row(r: dict) -> str:
    """manifest 행에서 결함 코드를 뽑는다. 야장은 한글 이름, 원본은 `s20_XX` 형태."""
    from build_v5_and_valset import YAJANG_NAME_TO_CODE
    if r["area"] in ("북면", "상면", "설악면"):
        return YAJANG_NAME_TO_CODE.get(r["name"], "?")
    return r["code"].split("_", 1)[-1] if "_" in r["code"] else r["code"]


if __name__ == "__main__":
    _ = time
    main()
