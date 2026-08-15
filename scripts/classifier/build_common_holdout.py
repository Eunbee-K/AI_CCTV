"""test3(YOLO)와 필터C를 **같은 판에서** 재기 위한 공통 검증셋.

왜 필요한가
    지금까지 두 모델을 각각 다른 판에서 쟀다 — 필터C는 `testset_bycode`(S20/AIHub)와
    `valset_bycode`(야장), test3는 `test3_holdout`(야장). 학습분이 서로 달라
    한쪽에 깨끗한 판이 다른 쪽에는 유출된 판이었기 때문이다(`testset_bycode`는
    test3와 249장 겹친다).

    그래서 "필터 96% vs YOLO 36%"는 **다른 사진에서 나온 값**이라 직접 비교가
    성립하지 않았다. 이 스크립트는 **두 모델의 학습분을 동시에 제외**해 어느 쪽도
    본 적 없는 사진만 모은다.

대가
    두 학습분을 다 빼면 31종 중 12종만 사진이 남는다. 나머지는 원본이 적어
    (JS 4,124장·LS 8,796장 등) 두 모델이 전량 학습에 써버렸다. 그 19종은 이
    판으로 측정할 수 없고, 기존 판의 값을 참고치로 병기한다(비교가 아니라 각자의 값).

구성
    결함 12종 × 30장 — BK CC CL DS JD JF LP SD (둘 다 아는 8종)
                     + BC DF DG LD (필터C만 아는 4종, YOLO는 클래스가 없다)
    정상 200장 — PJ 60% · IN 20% · 관 밖 20% (실제 영상 구성에 맞춘 비율)

사용:
    python build_common_holdout.py
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
from paths import DATASET, FILTER_DATA, RESULTS  # noqa: E402

TEST3_RUN = RESULTS / "test3_18class_sweep01" / "test3_960_e50"
V6 = FILTER_DATA / "clsdata_old_v6"

S20 = DATASET / "original" / "rename_data_s20_s22_bbox"
AIHUB = DATASET / "original" / "aihub_data_bbox" / "image"

NORMAL_CODES = ("IN", "PJ", "OUT_MH", "OUT_INVERT", "OUT_CAR")
NORMAL_MIX = {"PJ": 0.60, "IN": 0.20, "OUT": 0.20}

# 이 판으로 잴 수 있는 결함(둘 다 학습분을 빼고도 30장 이상 남는 것).
# ETC(기타)는 결함 종류가 아니라 제외한다.
TARGET_CODES = ["BK", "CC", "CL", "DS", "JD", "JF", "LP", "SD",   # 둘 다 아는 8종
                "BC", "DF", "DG", "LD"]                            # 필터C만 아는 4종


def code_of(folder: str):
    """폴더명 -> 31종 코드. 다른 빌더들과 같은 규칙."""
    if folder in NORMAL_CODES:
        return folder
    m = re.match(r"^S22_([A-Z]{2})_", folder) or re.match(r"^([A-Z]{2,3})_S20_", folder)
    if m:
        return m.group(1)
    return folder if re.fullmatch(r"[A-Z]{2,3}", folder) else None


def used_by_both() -> set:
    """test3와 필터C가 학습(train/val/test 전부)에 쓴 원본 파일명 전체.

    test3는 `used_files.csv`의 source_path(원본 절대경로)에서 파일명을 뽑는다 —
    dest_filename은 `yolo_txt_BK__원본명.png`처럼 접두어가 붙어 그대로 비교하면
    겹침이 0으로 나온다(한 번 헛짚었다).

    필터C는 manifest의 dst가 `s20_DE_원본명` / `add_aihub_OUT_MH__원본명` 형태라
    접두어를 벗겨 원본명을 되살린다.
    """
    used = set()
    p = TEST3_RUN / "used_files.csv"
    if not p.exists():
        raise SystemExit(f"test3 used_files.csv가 없습니다: {p}")
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            used.add(Path(r["source_path"]).name)
    n_yolo = len(used)

    with open(V6 / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = Path(r["dst"]).name
            if "__" in n:
                n = n.split("__", 1)[1]
            parts = n.split("_", 2)
            used.add(parts[2] if len(parts) > 2 and parts[0] in ("s20", "aihub") else n)

    print(f"제외 대상 — test3 {n_yolo:,}개 + 필터C 추가분 "
          f"{len(used) - n_yolo:,}개 = 총 {len(used):,}개\n")
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defects", type=int, default=30, help="결함 종류당 장수")
    ap.add_argument("--normals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    out = FILTER_DATA / "common_holdout"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)

    used = used_by_both()

    pool = defaultdict(list)
    for root in (S20, AIHUB):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            c = code_of(d.name)
            if c is None:
                continue
            for f in d.rglob("*"):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.name not in used:
                    pool[c].append(f)

    rows = []
    print(f"{'코드':<7}{'미사용':>9}{'뽑음':>7}")
    for code in TARGET_CODES:
        fs = pool.get(code, [])
        random.shuffle(fs)
        take = fs[:args.defects]
        for f in take:
            dst = out / "images" / f"{code}__{f.name}"
            shutil.copy2(f, dst)
            rows.append({"label": "defect", "code": code,
                         "file": dst.name, "src": str(f)})
        flag = "" if len(take) == args.defects else "  <- 부족"
        print(f"  {code:<7}{len(fs):>9,}{len(take):>7}{flag}")

    print(f"\n정상 {args.normals}장 (PJ 60% · IN 20% · 관 밖 20%)")
    plan = {"PJ": round(args.normals * NORMAL_MIX["PJ"]),
            "IN": round(args.normals * NORMAL_MIX["IN"])}
    plan["OUT"] = args.normals - plan["PJ"] - plan["IN"]
    for group, n in plan.items():
        srcs = ([f for c in ("OUT_MH", "OUT_INVERT", "OUT_CAR") for f in pool.get(c, [])]
                if group == "OUT" else list(pool.get(group, [])))
        random.shuffle(srcs)
        take = srcs[:n]
        for f in take:
            code = f.parent.name if group == "OUT" else group
            dst = out / "images" / f"{code}__{f.name}"
            shutil.copy2(f, dst)
            rows.append({"label": "normal", "code": code,
                         "file": dst.name, "src": str(f)})
        print(f"  {group:<7}{len(srcs):>9,}{len(take):>7}")

    # 확인 사살 — 정말 두 학습분과 안 겹치는지 다시 센다
    overlap = used & {Path(r["src"]).name for r in rows}
    print(f"\n[유출 검증] 두 모델 학습분과 겹침 {len(overlap)}개 "
          + ("!! 문제" if overlap else "OK"))

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["label", "code", "file", "src"])
        w.writeheader()
        w.writerows(rows)

    d = sum(1 for r in rows if r["label"] == "defect")
    print(f"\n결함 {d}장 ({len({r['code'] for r in rows if r['label']=='defect'})}종) · "
          f"정상 {len(rows)-d}장")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
