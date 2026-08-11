"""신설관로 현장 프레임(fieldset_v1_label)을 필터 학습셋의 train에 더한다.

**왜 넣나** — 지금 학습 정상 762장이 전부 가평 상면 한 현장에서 나왔다. 하수관 내부에서
"정상"의 생김새는 관종·조명·카메라·물때에 따라 달라지는데, 현장이 하나면 그 폭을
못 배운다. 신설관로 데이터는 관로 22개에 지역 코드가 6종(HL·HS·MI·SD·SS·YH)이라
현장 다양성을 직접 채워준다.

**위험** — 신설관로의 정상은 '깨끗한 새 관'이고 노후관로의 정상은 '낡았지만 멀쩡한 관'이다.
"깨끗하면 정상"을 배우면 정상 노후관을 결함이라 할 수 있다. 지금 문제가 정확히 그
방향이라 도움이 될지 해가 될지는 돌려봐야 안다. val은 노후관로 5개 그대로 두므로
효과가 그대로 드러난다.

**val에는 넣지 않는다** — 다른 실험들과 같은 잣대로 비교해야 한다.

사용:
    python add_newpipe_data.py --base E:/AI_CCTV_DATASET/clsdata_v3_mixed
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

SRC = Path(r"E:/AI_CCTV_DATASET/fieldset_v1_label")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"E:/AI_CCTV_DATASET/clsdata_v3_mixed")
    ap.add_argument("--out", default=r"E:/AI_CCTV_DATASET/clsdata_v4_newpipe")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    base, out = Path(args.base), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)
    print(f"기반 복사: {base.name} -> {out.name}")

    with open(out / "manifest.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])

    added = Counter()
    for split in ("train", "val"):
        d = SRC / "images" / split
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() != ".jpg":
                continue
            parts = f.stem.split("__")
            pipe = parts[0]
            tag = parts[1] if len(parts) > 1 else "?"
            label = "normal" if tag == "NORMAL" else "defect"

            # NORMAL 태그인데 결함 박스가 그려진 프레임은 라벨이 모순이라 뺀다.
            lp = SRC / "labels" / split / f"{f.stem}.txt"
            n_box = 0
            if lp.exists():
                n_box = sum(1 for line in lp.read_text(encoding="utf-8").splitlines()
                            if len(line.split()) >= 5)
            if label == "normal" and n_box:
                added["제외(정상인데 박스 있음)"] += 1
                continue

            # 신설관로는 학습에만 넣는다. val은 노후관로 5개를 유지해야 비교가 된다.
            dst = out / "train" / label / f"newpipe__{f.name}"
            Image.open(f).convert("RGB").resize(
                (args.size, args.size), Image.BILINEAR).save(dst, "JPEG", quality=88)
            row = {k: "" for k in fields}
            row.update({"split": "train", "label": label,
                        "source": "newpipe", "code": pipe,
                        "dst": str(dst.relative_to(out))})
            rows.append(row)
            added[label] += 1

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n=== 신설관로 추가 ===")
    for k, v in sorted(added.items()):
        print(f"  {k:<26} {v:>4}장")
    print("\n=== 최종 구성 ===")
    for split in ("train", "val"):
        line = []
        for lab in ("normal", "defect"):
            line.append(f"{lab} {len(list((out / split / lab).glob('*.jpg'))):>5}")
        print(f"  {split:<6} " + "  ".join(line))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
