"""결함 종류별 테스트셋 — 학습에 안 쓴 사진만 모은다.

지금까지 필터 성능을 야장 val(0.95)이나 내가 만든 실영상 정답(0.52)으로 봤는데,
전자는 학습과 같은 출처라 후하고 후자는 정답 자체가 틀렸다. **결함 종류별로
무엇을 잡고 무엇을 놓치는지**를 보려면 종류마다 같은 장수로 맞춘 판이 필요하다.

구성
    결함: 종류당 DEFECT_PER_CODE장 (S20/S22 + AIHub, clsdata_old_v3 미사용분)
    정상: NORMAL_TOTAL장 — 이음부(PJ) 60% · 관내부(IN) 20% · 외부전경(OUT_*) 20%

    정상 비율을 이렇게 잡은 이유는 실제 조사 영상의 구성에 맞추기 위해서다.
    이음부는 관로마다 몇 미터씩 나오고, 외부 전경은 시작·끝에만 나온다.
    **외부를 빼면 안 된다** — v1이 "관 밖 = 정상"만 배워 현장에서 무너졌는데,
    그 실패가 재발하는지 보려면 판에 들어 있어야 한다.

사용:
    python build_testset_bycode.py
    python build_testset_bycode.py --defects 20 --normals 200
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
from paths import DATASET, FILTER_DATA  # noqa: E402

ORIG = DATASET / "original"
S20 = ORIG / "rename_data_s20_s22_bbox"
AIHUB = ORIG / "aihub_data_bbox" / "image"
TRAINED = FILTER_DATA / "clsdata_old_v3"

# 정상 구성 — 실제 영상 구성에 맞춘 비율
NORMAL_MIX = {"PJ": 0.60, "IN": 0.20, "OUT": 0.20}
OUT_DIRS = ("OUT_MH", "OUT_INVERT", "OUT_CAR")


def code_of(folder: str) -> str | None:
    """폴더명 -> 31종 코드. 같은 결함이 S20/S22/AIHub에 흩어져 있다."""
    if folder in ("IN", "PJ", *OUT_DIRS):
        return folder
    m = re.match(r"^S22_([A-Z]{2})_", folder)          # S22_CL_01균열길이(CL)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Z]{2,3})_S20_", folder)        # CL_S20_균열(길이)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Z]{2,3}", folder):            # AIHub: BK, CL, ETC …
        return folder
    return None


def used_names() -> set:
    """clsdata_old_v3에 들어간 원본 파일명."""
    out = set()
    with open(TRAINED / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["area"] not in ("s20", "aihub"):
                continue
            name = r["dst"].replace("\\", "/").split("/")[-1]
            parts = name.split("_", 2)                 # s20_DE_<원본이름>
            out.add(parts[2] if len(parts) > 2 else name)
    return out


def scan(root: Path, used: set) -> dict:
    """{코드: [미사용 경로]}"""
    pool = defaultdict(list)
    if not root.is_dir():
        return pool
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        code = code_of(d.name)
        if code is None:
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.name not in used:
                pool[code].append(f)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defects", type=int, default=20, help="결함 종류당 장수")
    ap.add_argument("--normals", type=int, default=200)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    out = Path(args.out) if args.out else FILTER_DATA / "testset_bycode"
    if out.exists():
        shutil.rmtree(out)
    (out / "defect").mkdir(parents=True)
    (out / "normal").mkdir(parents=True)

    used = used_names()
    print(f"학습에 쓴 s20/aihub 파일 {len(used):,}개 제외\n")

    pool = defaultdict(list)
    for root in (S20, AIHUB):
        for k, v in scan(root, used).items():
            pool[k] += v

    normal_codes = {"IN", "PJ", *OUT_DIRS}
    rows = []

    print(f"{'코드':<8}{'미사용 전체':>11}{'뽑음':>7}")
    for code in sorted(k for k in pool if k not in normal_codes):
        fs = pool[code]
        random.shuffle(fs)
        take = fs[:args.defects]
        for f in take:
            dst = out / "defect" / f"{code}__{f.name}"
            shutil.copy2(f, dst)
            rows.append({"label": "defect", "code": code, "src": str(f), "file": dst.name})
        flag = "" if len(take) == args.defects else f"  ← {args.defects}장 못 채움"
        print(f"  {code:<8}{len(fs):>11,}{len(take):>7}{flag}")

    print(f"\n정상 {args.normals}장 — 이음부 {NORMAL_MIX['PJ']:.0%} · "
          f"관내부 {NORMAL_MIX['IN']:.0%} · 외부전경 {NORMAL_MIX['OUT']:.0%}")
    plan = {"PJ": round(args.normals * NORMAL_MIX["PJ"]),
            "IN": round(args.normals * NORMAL_MIX["IN"]),
            "OUT": args.normals - round(args.normals * NORMAL_MIX["PJ"])
                   - round(args.normals * NORMAL_MIX["IN"])}
    for group, n in plan.items():
        srcs = ([f for d in OUT_DIRS for f in pool.get(d, [])] if group == "OUT"
                else list(pool.get(group, [])))
        random.shuffle(srcs)
        take = srcs[:n]
        for f in take:
            code = f.parent.name if group == "OUT" else group
            dst = out / "normal" / f"{code}__{f.name}"
            shutil.copy2(f, dst)
            rows.append({"label": "normal", "code": code, "src": str(f), "file": dst.name})
        print(f"  {group:<8}{len(srcs):>11,}{len(take):>7}")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["label", "code", "src", "file"])
        w.writeheader()
        w.writerows(rows)

    d = sum(1 for r in rows if r["label"] == "defect")
    n = len(rows) - d
    print(f"\n결함 {d}장 ({len({r['code'] for r in rows if r['label']=='defect'})}종) · 정상 {n}장")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
