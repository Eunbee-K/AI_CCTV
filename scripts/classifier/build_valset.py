"""검증셋 `valset_bycode`만 따로 만든다 — 학습셋은 건드리지 않는다.

`build_v5_and_valset.py`가 학습셋까지 지우고 다시 만드는데, 외장하드에서 파일이
잠겨 있으면 rmtree가 PermissionError로 죽는다. 학습셋이 이미 만들어져 있으면
검증셋만 다시 만들면 되므로 그 부분만 떼어냈다.

검증셋 구성
    결함: 종류당 VAL_PER_CODE장. **야장을 먼저 쓰고 모자란 만큼만 원본에서 채운다.**
          야장이 실제 조사 상황에 가깝기 때문이다. ETC(기타)는 결함 종류가 아니라 제외.
    정상: 결함과 같은 수를 학습과 같은 비율(IN 30 / PJ 40 / OUT_* 각 10)로.
          PJ는 야장 정상을 먼저 쓴다 — 결함이 야장 우선인데 정상만 AIHub면 모델이
          결함/정상이 아니라 출처를 보고 갈라도 점수가 나온다(v1이 그렇게 망가졌다).
          야장 정상은 전부 관 안쪽이고 89%가 '이음부(접합부)존재'라 PJ에 대응한다.

**학습셋에 이미 쓴 파일은 제외한다.** 겹치면 성적이 부풀려진다.

사용:
    python build_valset.py
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))              # scripts/classifier/
from paths import DATASET  # noqa: E402
from build_v5_and_valset import (  # noqa: E402
    NORMAL_MIX, VAL_PER_CODE, V3, YAJANG_NAME_TO_CODE, interleave,
    rmtree_retry, scan_all,
)

TRAIN = DATASET / "clsdata_old_v5"
VAL_EXCLUDE = {"ETC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    val_out = DATASET / "valset_bycode"
    rmtree_retry(val_out)
    for lab in ("defect", "normal"):
        (val_out / lab).mkdir(parents=True)

    # 학습에 쓴 원본 파일명 — 검증에서 빼야 한다
    train_used = set()
    with open(TRAIN / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = Path(r["dst"]).name
            train_used.add(n.split("__", 1)[1] if "__" in n else n)
    print(f"학습에 쓴 파일 {len(train_used):,}개 제외\n")

    pool = scan_all()

    # 야장 — 결함은 코드별로, 정상은 한 묶음으로
    ya_def, ya_nor = defaultdict(list), []
    with open(V3 / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["area"] not in ("북면", "상면", "설악면"):
                continue
            src = V3 / r["dst"].replace("\\", "/")
            if not src.exists():
                continue
            if r["label"] == "defect":
                code = YAJANG_NAME_TO_CODE.get(r["name"])
                if code:
                    ya_def[code].append(src)
            else:
                ya_nor.append(src)
    for v in ya_def.values():
        random.shuffle(v)
    random.shuffle(ya_nor)

    vman = []
    codes = sorted((set(pool) | set(YAJANG_NAME_TO_CODE.values()))
                   - set(NORMAL_MIX) - VAL_EXCLUDE)

    print(f"{'코드':<7}{'야장':>6}{'원본':>6}{'합계':>6}")
    for code in codes:
        take_ya = ya_def.get(code, [])[:VAL_PER_CODE]
        for src in take_ya:
            dst = val_out / "defect" / f"{code}__yajang__{src.name}"
            shutil.copy2(src, dst)
            vman.append({"label": "defect", "code": code, "from": "yajang",
                         "file": dst.name, "src": str(src)})
        need = VAL_PER_CODE - len(take_ya)
        extra = [x for x in interleave(pool.get(code, []), args.seed)
                 if x[0].name not in train_used][:need] if need > 0 else []
        for f, tag in extra:
            dst = val_out / "defect" / f"{code}__{tag}__{f.name}"
            shutil.copy2(f, dst)
            vman.append({"label": "defect", "code": code, "from": tag,
                         "file": dst.name, "src": str(f)})
        total = len(take_ya) + len(extra)
        print(f"{code:<7}{len(take_ya):>6}{len(extra):>6}{total:>6}"
              + ("" if total == VAL_PER_CODE else "  <- 부족"))

    n_def = sum(1 for r in vman if r["label"] == "defect")
    print(f"\n정상 목표 {n_def}장 (결함과 같은 수) — 야장 우선")
    print(f"  {'코드':<12}{'목표':>6}{'야장':>6}{'AIHub':>7}")
    ya_i = 0
    for code, ratio in NORMAL_MIX.items():
        want = round(n_def * ratio)
        n_ya = 0
        if code == "PJ":                       # 야장 정상은 이음부 계열
            for src in ya_nor[ya_i:ya_i + want]:
                dst = val_out / "normal" / f"{code}__yajang__{src.name}"
                shutil.copy2(src, dst)
                vman.append({"label": "normal", "code": code, "from": "yajang",
                             "file": dst.name, "src": str(src)})
                n_ya += 1
            ya_i += n_ya
        need = want - n_ya
        take = [x for x in interleave(pool.get(code, []), args.seed)
                if x[0].name not in train_used][:need] if need > 0 else []
        for f, tag in take:
            dst = val_out / "normal" / f"{code}__{tag}__{f.name}"
            shutil.copy2(f, dst)
            vman.append({"label": "normal", "code": code, "from": tag,
                         "file": dst.name, "src": str(f)})
        print(f"  {code:<12}{want:>6}{n_ya:>6}{len(take):>7}")

    with open(val_out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["label", "code", "from", "file", "src"])
        w.writeheader()
        w.writerows(vman)

    print(f"\n검증셋 -> {val_out}")
    print(f"  결함 {n_def} · 정상 {sum(1 for r in vman if r['label'] == 'normal')}")


if __name__ == "__main__":
    main()
