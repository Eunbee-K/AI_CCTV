"""YOLO 검증셋을 **그 모델이 실제로 안 쓴 사진**으로만 만든다 (test3/test5 공용).

왜 필요한가
    `testset_bycode`는 test5 train과 259장(56.9%), test3와 249장 겹친다.
    그 판으로 잰 YOLO 성적(test5 이름정확도 90%)은 암기를 잰 값이었다.
    스윕 결과 폴더의 `used_files.csv`(그 run이 실제로 쓴 원본 파일 전체 목록)를
    빼고 새로 뽑으면 유출 없는 판을 만들 수 있다.

`build_test3_holdout.py`와 다른 점
    저쪽은 yaml+시드로 학습분을 **재현**하고 야장을 우선으로 채운다.
    여기는 `used_files.csv`로 **실제 사용분**을 빼고, S20/AIHub 원본에서만 뽑는다
    — 같은 도메인(그 모델이 학습한 분포) 안에서 "안 본 사진"을 주면
    도메인 차이가 아니라 **순수한 일반화 능력**만 볼 수 있다.

두 가지를 잰다 (score_yolo_holdout.py)
    1. 정상/결함 구분 — 이름이 틀려도 뭐라도 검출했으면 "결함이라 봤다"
    2. 결함별 이름 정확도 — 정답 코드를 실제로 맞혔나

사용:
    python build_yolo_holdout.py --model test5
    python build_yolo_holdout.py --model test3
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

# 모델별 결과 폴더 (used_files.csv가 있는 곳)
RUNS = {
    "test5": RESULTS / "test5_17class_sweep01" / "test5_960_e50",
    "test3": RESULTS / "test3_18class_sweep01" / "test3_960_e50",
}

S20 = DATASET / "original" / "rename_data_s20_s22_bbox"
AIHUB = DATASET / "original" / "aihub_data_bbox" / "image"

NORMAL_CODES = ("IN", "PJ", "OUT_MH", "OUT_INVERT", "OUT_CAR")


def code_of(folder: str):
    """폴더명 -> 31종 코드. build_testset_bycode.py와 같은 규칙."""
    if folder in NORMAL_CODES:
        return folder
    m = re.match(r"^S22_([A-Z]{2})_", folder)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Z]{2,3})_S20_", folder)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Z]{2,3}", folder):
        return folder
    return None


def used_files(run_dir: Path) -> set:
    """그 run이 실제로 쓴 원본 파일명 전체 (train+val+test).

    **source_path로 비교해야 한다** — dest_filename은 `yolo_txt_BK__원본명.png`
    처럼 접두어가 붙어서 그대로 쓰면 겹침 0개로 나온다(한 번 헛짚었다).
    """
    p = run_dir / "used_files.csv"
    if not p.exists():
        raise SystemExit(f"used_files.csv가 없습니다: {p}")
    with open(p, encoding="utf-8-sig") as f:
        return {Path(r["source_path"]).name for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RUNS), default="test5")
    ap.add_argument("--defects", type=int, default=30, help="결함 종류당 장수")
    ap.add_argument("--normals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    run_dir = RUNS[args.model]
    out = FILTER_DATA / f"{args.model}_holdout_pure"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)

    used = used_files(run_dir)
    print(f"{args.model} 학습에 실제로 쓴 파일 {len(used):,}개 제외\n")

    # 원본을 훑어 "안 쓴 것"만 모은다
    pool = defaultdict(list)
    for root in (S20, AIHUB):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            code = code_of(d.name)
            if code is None:
                continue
            for f in d.rglob("*"):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.name not in used:
                    pool[code].append(f)

    rows = []
    print(f"{'코드':<7}{'미사용 전체':>11}{'뽑음':>7}")
    for code in sorted(k for k in pool if k not in NORMAL_CODES):
        fs = pool[code]
        random.shuffle(fs)
        take = fs[:args.defects]
        for f in take:
            dst = out / "images" / f"{code}__{f.name}"
            shutil.copy2(f, dst)
            rows.append({"label": "defect", "code": code,
                         "file": dst.name, "src": str(f)})
        flag = "" if len(take) == args.defects else "  <- 부족"
        print(f"  {code:<7}{len(fs):>11,}{len(take):>7}{flag}")

    # 정상 — 실제 영상 구성에 맞춰 PJ 60 / IN 20 / 관밖 20
    print(f"\n정상 {args.normals}장 (PJ 60% · IN 20% · 관밖 20%)")
    plan = {"PJ": round(args.normals * 0.6), "IN": round(args.normals * 0.2)}
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
        print(f"  {group:<7}{len(srcs):>11,}{len(take):>7}")

    # 확인 사살 — 정말 안 겹치는지 다시 센다
    mine = {Path(r["src"]).name for r in rows}
    overlap = used & mine
    print(f"\n[유출 검증] 겹침 {len(overlap)}개 " + ("!! 문제" if overlap else "OK"))

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
