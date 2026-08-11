"""이미 정해진 라벨·분할은 그대로 두고, 이미지만 고해상도로 다시 만든다.

**왜 필요한가**

clsdata_v2/v3는 256x256으로 저장돼 있어 학습 입력을 384로 올려도 소용이 없다.
256을 384로 늘리는 것뿐이라 정보가 늘지 않는다. 원본에서 다시 뽑아야 한다.

1280x720을 224x224로 줄이면 가로가 0.175배가 된다. 이음부 벌어짐처럼 폭이 좁은
결함은 몇 픽셀로 뭉개진다. A·B·C 모두 결함의 절반을 놓치는 원인이 여기일 수 있다.

**원본을 어디서 가져오는가** (파일명이 같아서 그대로 찾을 수 있다)

  현장 영상 프레임  clsdata_v2_review/   1280x720
  조사표 사진       fieldset_v2_defects/photos/  1790x1000
  AIHub·S20        original/ 아래 클래스 폴더  (clsdata_v1 manifest의 src 경로)

라벨과 분할은 대상 데이터셋의 현재 폴더 구조를 그대로 따른다 — 사람이 검토하며
옮긴 결과가 거기 반영돼 있기 때문이다.

사용:
    python rebuild_hires.py --src E:/AI_CCTV_DATASET/clsdata_v3_mixed --size 384
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

DS = Path(r"E:/AI_CCTV_DATASET")


def build_index(v1_manifest: Path) -> dict[str, str]:
    """clsdata_v1의 저장 파일명 -> 원본 경로. AIHub/S20 고해상도를 찾는 데 쓴다."""
    if not v1_manifest.exists():
        return {}
    with open(v1_manifest, encoding="utf-8") as f:
        return {Path(r["dst"]).name: r["src"] for r in csv.DictReader(f)}


def find_source(name: str, v1_index: dict[str, str]) -> Path | None:
    """저장 파일명으로 고해상도 원본을 찾는다."""
    # 1) AIHub/S20 (clsdata_v3에서는 `aihub__<코드>__<원래이름>` 형태)
    key = name
    if name.startswith("aihub__"):
        key = name.split("__", 2)[-1]
    if key in v1_index:
        p = Path(v1_index[key])
        if p.exists():
            return p

    # 2) 조사표 사진 (`hwp__<원본stem>.jpg` 또는 `field__hwp__...`)
    stem = Path(name).stem
    for pre in ("field__hwp__", "hwp__"):
        if stem.startswith(pre):
            p = DS / "fieldset_v2_defects" / "photos" / f"{stem[len(pre):]}.png"
            if p.exists():
                return p

    # 3) 현장 영상 프레임 — 검토용(1280x720)에 같은 이름이 있다
    plain = stem[len("field__"):] if stem.startswith("field__") else stem
    for split in ("train", "val"):
        for lab in ("normal", "defect"):
            p = DS / "clsdata_v2_review" / split / lab / f"{plain}.jpg"
            if p.exists():
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="라벨·분할을 가져올 데이터셋")
    ap.add_argument("--out", default="")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--v1", default=str(DS / "clsdata_v1" / "manifest.csv"))
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src.parent / f"{src.name}_hires{args.size}"
    v1_index = build_index(Path(args.v1))
    print(f"AIHub 원본 색인 {len(v1_index):,}건")

    rows, stat, miss = [], Counter(), []
    for split in ("train", "val"):
        for lab in ("normal", "defect"):
            d = src / split / lab
            if not d.is_dir():
                continue
            (out / split / lab).mkdir(parents=True, exist_ok=True)
            for f in sorted(d.glob("*.jpg")):
                origin = find_source(f.name, v1_index)
                # 원본을 못 찾으면 저장본이라도 쓴다(해상도 이득은 없지만 데이터는 유지)
                use, hi = (origin, True) if origin else (f, False)
                try:
                    im = Image.open(use).convert("RGB")
                except Exception:
                    miss.append(f.name)
                    continue
                im.resize((args.size, args.size), Image.BICUBIC).save(
                    out / split / lab / f.name, "JPEG", quality=92, optimize=True)
                stat[(split, lab, "고해상도" if hi else "저해상도")] += 1
                if not hi:
                    miss.append(f.name)
                pipe = f.stem.split("__")[1] if "hwp__" in f.stem else f.stem.split("__")[0]
                rows.append({"split": split, "label": lab, "code": pipe[:24],
                             "hires": int(hi), "dst": f"{split}\\{lab}\\{f.name}"})

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== {args.size}px 재구성 ===")
    for k in sorted(stat):
        print(f"  {k[0]:<6} {k[1]:<7} {k[2]:<7} {stat[k]:>5}장")
    hi = sum(v for k, v in stat.items() if k[2] == "고해상도")
    print(f"\n  고해상도 확보 {hi:,} / 전체 {sum(stat.values()):,}장")
    if miss:
        print(f"  원본 못 찾음 {len(miss)}장 (저장본 사용): {miss[:3]}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
