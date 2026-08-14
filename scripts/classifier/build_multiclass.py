"""이진 데이터셋(정상/결함)을 **다중 클래스**(결함 종류별 + 정상)로 다시 묶는다.

무엇을 보려는 것인가
    지금 필터는 "결함이 있나 없나"만 답하고 종류는 YOLO가 붙인다. 그런데 YOLO는
    bbox를 학습해야 하니 라벨링 비용이 크다. **분류기가 bbox 없이 종류까지 맞힐 수
    있다면** 파이프라인이 훨씬 단순해진다.

    같은 이미지를 쓰고 폴더 구조만 바꾸므로, 이진 모델과 성능을 직접 비교할 수 있다.

클래스 구성
    결함 25종은 코드별 폴더로 나눈다(BK, CC, CL, ...).
    정상은 하나로 합칠지(`normal`) 나눌지(IN/PJ/OUT_*) 고를 수 있다.
    **기본은 합친다** — 관 밖과 관 안쪽을 나누면 클래스가 늘어 결함 쪽이 상대적으로
    묽어지고, 우리가 알고 싶은 건 "결함 종류를 가르는가"이기 때문이다.

    장수가 너무 적은 종류(NS 4장 등)는 그대로 두면 그 클래스를 아예 못 배운다.
    `--min-per-class` 미만이면 `ETC`로 합친다.

사용:
    python build_multiclass.py --src clsdata_old_v6
    python build_multiclass.py --src clsdata_old_v6 --split-normal
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))              # scripts/classifier/
from paths import FILTER_DATA  # noqa: E402
from build_v5_and_valset import YAJANG_NAME_TO_CODE, rmtree_retry  # noqa: E402


def code_of(r: dict) -> str:
    """manifest 행 -> 결함 코드. 야장은 한글 이름, 원본은 `s20_XX` 형태."""
    if r["area"] in ("북면", "상면", "설악면"):
        return YAJANG_NAME_TO_CODE.get(r.get("name", ""), "ETC")
    c = r.get("code", "")
    return c.split("_", 1)[-1] if "_" in c else (c or "ETC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="clsdata_old_v6")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-per-class", type=int, default=50,
                    help="이보다 적은 종류는 ETC로 합친다 — 못 배울 클래스를 만들지 않는다")
    ap.add_argument("--normal-ratio", type=float, default=0.30,
                    help="정상이 결함 전체 대비 차지할 비율. 다중 클래스에서는 정상이 "
                         "한 덩어리라 1:1로 두면 결함 25종을 합친 것과 맞먹어 "
                         "정상 쪽으로 쏠린다")
    ap.add_argument("--split-normal", action="store_true",
                    help="정상을 IN/PJ/OUT_*로 나눈다(기본은 normal 하나)")
    args = ap.parse_args()

    src = FILTER_DATA / args.src
    out = Path(args.out) if args.out else FILTER_DATA / f"{args.src}_multiclass"
    if not (src / "manifest.csv").exists():
        raise SystemExit(f"manifest가 없습니다: {src}")
    rmtree_retry(out)

    rows = list(csv.DictReader(open(src / "manifest.csv", encoding="utf-8-sig")))

    # 어떤 코드를 ETC로 합칠지 먼저 정한다 (train 기준)
    n_by_code = Counter(code_of(r) for r in rows
                        if r["split"] == "train" and r["label"] == "defect")
    merged = {c for c, n in n_by_code.items() if n < args.min_per_class}
    if merged:
        print(f"{args.min_per_class}장 미만이라 ETC로 합침: "
              + ", ".join(f"{c}({n_by_code[c]})" for c in sorted(merged)) + "\n")

    # **정상을 결함 전체의 normal_ratio만큼만 남긴다.**
    # 이진에서는 정상:결함이 1:1이었는데, 다중 클래스에서는 정상이 한 덩어리라
    # 결함 25종을 전부 합친 것과 맞먹는다. 그대로 두면 "일단 정상"이라고 답하는
    # 쪽으로 쏠려 종류를 가르는 힘이 줄어든다.
    # 관 밖(OUT_*) 비율은 유지하면서 줄인다 — 관 밖을 결함으로 오판하는 것이
    # v6에서 고치려는 문제라 그 몫이 사라지면 안 된다.
    drop_files = set()
    for split in ("train", "val"):
        defs = [r for r in rows if r["split"] == split and r["label"] == "defect"]
        nors = [r for r in rows if r["split"] == split and r["label"] == "normal"]
        keep_n = round(len(defs) * args.normal_ratio)
        if not nors or keep_n >= len(nors):
            continue
        # v3에서 이식된 정상 행은 code가 비어 있다. 관 밖은 파일명으로 판정한다
        # (build_old_v6가 `add_aihub_OUT_MH__...` 형태로 저장했다).
        def is_out(r):
            key = str(r.get("code", "")) + " " + str(r.get("dst", ""))
            return any(t in key for t in ("OUT_MH", "OUT_CAR", "OUT_INVERT"))
        out_n = [r for r in nors if is_out(r)]
        in_n = [r for r in nors if not is_out(r)]
        ratio_out = len(out_n) / len(nors)
        keep_out = round(keep_n * ratio_out)
        rng = random.Random(0)
        rng.shuffle(out_n)
        rng.shuffle(in_n)
        for r in out_n[keep_out:] + in_n[keep_n - keep_out:]:
            drop_files.add(r["dst"])
        print(f"[{split}] 정상 {len(nors):,} -> {keep_n:,}장 "
              f"(결함 {len(defs):,}의 {args.normal_ratio:.0%}) · "
              f"관 밖 {keep_out:,} 유지")

    plan, n_copy = defaultdict(int), 0
    man = []
    for r in rows:
        if r["dst"] in drop_files:
            continue
        if r["label"] == "defect":
            cls = code_of(r)
            if cls in merged:
                cls = "ETC"
        elif args.split_normal:
            c = r.get("code", "")
            cls = c.split("_", 1)[-1] if "_" in c else "normal"
            if cls not in ("IN", "PJ", "OUT_MH", "OUT_CAR", "OUT_INVERT"):
                cls = "normal"
        else:
            cls = "normal"

        s = src / r["dst"].replace("\\", "/")
        if not s.exists():
            continue
        d = out / r["split"] / cls / s.name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        man.append({"split": r["split"], "cls": cls, "orig_label": r["label"],
                    "area": r["area"], "file": s.name})
        plan[(r["split"], cls)] += 1
        n_copy += 1
        if n_copy % 2000 == 0:
            print(f"  {n_copy:,}장…")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["split", "cls", "orig_label", "area", "file"])
        w.writeheader()
        w.writerows(man)

    print(f"\n{out}")
    for split in ("train", "val"):
        items = sorted((c, n) for (s, c), n in plan.items() if s == split)
        if not items:
            continue
        tot = sum(n for _, n in items)
        print(f"\n[{split}] {tot:,}장 · {len(items)}클래스")
        for c, n in items:
            print(f"  {c:<12}{n:>6,}")


if __name__ == "__main__":
    main()
