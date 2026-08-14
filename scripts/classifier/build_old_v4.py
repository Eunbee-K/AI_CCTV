"""OLD_v4 — 야장 데이터에 **연속 영상에서 뽑은 정상**을 얹는다.

왜
    OLD_v3는 야장 val AUC 0.9503인데 실영상 0.5205다. "정상"으로 배운 것이 전부
    조사원이 뭔가 적을 만해서 적은 순간이라서다 — 현장 정상 6,744장의 89%가
    `이음부(접합부)존재`고, 아무것도 없는 밋밋한 관 내부는 한 장도 없다.
    실제 영상은 대부분이 그 구간이라 모델에게는 처음 보는 그림이다.

    그 종류가 이미 있다. `clsdata_v2`(build_field_filter_dataset.py가 상면 CCTV
    영상에서 만든 것)의 정상 1,058장이 바로 연속 영상의 평범한 구간이다.
    그런데 A~E 실험은 **그것만으로** 학습해 절대량이 부족했고(F1 0.63~0.72),
    OLD 계열은 **그것이 아예 없어서** 실영상에서 무너졌다. 둘을 합친 적이 없다.

관로 누수
    영상(clsdata_v2)과 야장(clsdata_old_v3)은 **같은 관로를 다른 방식으로 담고 있다**.
    상면 CCTV의 `SM1-131-001/6058.mp4`는 야장 관로 6058과 같은 관이다.
    old_v3에서 val인 관로(6058·6085)의 영상 프레임을 train에 넣으면 누수다.
    그래서 관로번호로 대조해 old_v3의 소속을 그대로 따라간다.

val
    **val은 old_v3와 똑같이 둔다.** 야장 2,580장 그대로. 그래야 AUC를 OLD_v3의
    0.9503과 직접 비교할 수 있다. 영상 정상을 val에 섞으면 좋은 val이 되지만
    비교가 깨진다 — 판정은 어차피 실영상(score_on_real_video.py)으로 한다.

    다만 val 관로(6058·6085)의 영상 프레임은 train에 못 넣으므로 버린다.

사용:
    python build_old_v4.py                 # 기본 배율 1
    python build_old_v4.py --repeat 3      # 영상 정상을 3배로 복제
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
from paths import DATASET, FILTER_DATA  # noqa: E402

OLD = FILTER_DATA / "clsdata_old_v3"
VID = FILTER_DATA / "clsdata_v2"

# clsdata_v2의 관로 폴더명 -> 야장 관로번호. 폴더명이 잘려 저장된 것이 있어
# 앞부분만으로 맞춘다(예: "SM1-131-001(재시공, 역구배 관처짐 연결부").
FOLDER_TO_PIPE = {
    "SM1-120-001": "5980", "SM1-121-001": "5981", "SM1-121-002": "5982",
    "SM1-121-003": "5982", "SM1-126-001": "6014", "SM1-127-001": "6070",
    "SM1-128-001": "6109", "SM1-130-001": "6118", "SM1-130-002": "6118",
    "SM1-131-001": "6058", "SM1-131-002": "6058", "SM1-132-001": "6098",
    "SM2-139-001": "6018", "SM2-141-001": "6119", "SM2-141-002": "6119",
    "SM2-144-001": "6023", "SM2-145-001": "6071", "SM2-148-001": "6085",
    "SM2-149-001": "6055", "SM2-152-001": "6024",
}


def pipe_of(folder: str) -> str | None:
    key = folder.split("(")[0].split(",")[0].strip()
    return FOLDER_TO_PIPE.get(key)


def old_v3_split() -> dict:
    """야장 관로번호 -> old_v3에서의 split."""
    out = {}
    with open(OLD / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m = re.search(r"(북면|상면|설악면)__(\d+)__", r["dst"])
            if m:
                out.setdefault(m.group(2), set()).add(r["split"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--repeat", type=int, default=1,
                    help="영상 정상을 몇 배로 넣을지. 24,302장 대비 1,058장은 4%%뿐이다")
    args = ap.parse_args()

    out = Path(args.out) if args.out else FILTER_DATA / "clsdata_old_v4"
    for split in ("train", "val"):
        for lab in ("normal", "defect"):
            (out / split / lab).mkdir(parents=True, exist_ok=True)

    rows, stat = [], Counter()

    # 1) old_v3 를 그대로 옮긴다
    with open(OLD / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            src = OLD / r["dst"].replace("\\", "/")
            if not src.exists():
                stat["old 원본 없음"] += 1
                continue
            dst = out / r["split"] / r["label"] / src.name
            shutil.copy2(src, dst)
            rows.append({**r, "source": f"yajang:{r['area']}",
                         "dst": f"{r['split']}\\{r['label']}\\{src.name}"})
            stat[f"old {r['split']}/{r['label']}"] += 1

    # 2) 영상 프레임(clsdata_v2)을 얹는다 — 관로 소속은 old_v3를 따른다
    split_of = old_v3_split()
    with open(VID / "manifest.csv", encoding="utf-8-sig") as f:
        v2 = list(csv.DictReader(f))

    unknown = defaultdict(int)
    for r in v2:
        src = VID / r["dst"].replace("\\", "/")
        if not src.exists():
            stat["영상 원본 없음"] += 1
            continue
        pipe = pipe_of(r["pipe"])
        if pipe is None:
            unknown[r["pipe"]] += 1
            continue
        owner = split_of.get(pipe, set())
        if "val" in owner:
            # old_v3 val 관로다. train에 넣으면 누수, val에 넣으면 비교가 깨진다.
            stat["val 관로라 제외"] += 1
            continue

        n = args.repeat if r["label"] == "normal" else 1
        for i in range(n):
            name = f"vid__{pipe}__{src.stem}" + (f"__r{i}" if i else "") + ".jpg"
            shutil.copy2(src, out / "train" / r["label"] / name)
            rows.append({"split": "train", "label": r["label"], "area": f"영상({pipe})",
                         "code": "", "name": "", "grade": "",
                         "source": "video", "dst": f"train\\{r['label']}\\{name}"})
            stat[f"video train/{r['label']}"] += 1

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["split", "label", "area", "code", "name", "grade",
                               "source", "dst"])
        w.writeheader()
        w.writerows(rows)

    print(f"저장: {out}")
    for k, v in sorted(stat.items()):
        print(f"  {k:<24}{v:>7,}")
    if unknown:
        print("  관로 매칭 실패(제외):")
        for k, v in unknown.items():
            print(f"    {k[:40]:<42}{v:>5}")

    print()
    c = Counter((r["split"], r["label"]) for r in rows)
    for split in ("train", "val"):
        n = c[(split, "normal")]
        d = c[(split, "defect")]
        print(f"  {split:<6} 정상 {n:>6,}  결함 {d:>6,}  (합 {n+d:,})")
    vid_n = stat["video train/normal"]
    print(f"\n  영상 정상이 train 정상에서 차지하는 비율: "
          f"{vid_n / max(c[('train','normal')], 1):.1%}")


if __name__ == "__main__":
    main()
