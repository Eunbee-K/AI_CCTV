"""학습셋·검증셋 구성을 한눈에 보여준다.

만들어진 데이터가 의도대로인지 확인하는 용도다. 특히 세 가지를 검산한다.
  - **학습과 검증이 겹치지 않는가** (원본 파일명 기준). 겹치면 성적이 부풀려진다.
  - **정상/결함 비율**이 1:1인가.
  - **출처가 한쪽으로 쏠리지 않았는가.** 결함은 야장인데 정상은 전부 AIHub면
    모델이 결함/정상이 아니라 출처를 보고 갈라도 점수가 나온다(v1이 그랬다).

사용:
    python report_dataset.py
"""

from __future__ import annotations

import csv
import io
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
from paths import DATASET  # noqa: E402

TRAIN = DATASET / "clsdata_old_v5"
VAL = DATASET / "valset_bycode"


def rows(p: Path) -> list:
    with open(p / "manifest.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def bar(n: int, total: int, width: int = 22) -> str:
    return "█" * round(width * n / max(total, 1)) + "·" * (
        width - round(width * n / max(total, 1)))


def main():
    tr = rows(TRAIN)
    va = rows(VAL)
    t_def = [r for r in tr if r["label"] == "defect"]
    t_nor = [r for r in tr if r["label"] == "normal"]
    v_def = [r for r in va if r["label"] == "defect"]
    v_nor = [r for r in va if r["label"] == "normal"]

    print("=" * 60)
    print("학습셋  clsdata_old_v5   (AIHub / S20 / S22 만)")
    print("=" * 60)
    print(f"  결함 {len(t_def):>6,}   정상 {len(t_nor):>6,}   "
          f"비율 1:{len(t_nor) / max(len(t_def), 1):.2f}")

    print("\n  [결함 — 코드별]")
    codes = Counter(r["code"] for r in t_def)
    for k, v in sorted(codes.items()):
        print(f"    {k:<7}{v:>5}" + ("" if v >= 100 else "   ← 원본 부족"))
    print(f"    {'합계':<7}{sum(codes.values()):>5}   ({len(codes)}종)")

    print("\n  [정상 — 구성]")
    for k, v in Counter(r["code"] for r in t_nor).most_common():
        print(f"    {k:<12}{v:>6,}  {v / len(t_nor):>6.1%}  {bar(v, len(t_nor))}")
    out_n = sum(1 for r in t_nor if r["code"].startswith("OUT"))
    print(f"\n    관 밖(OUT_*) {out_n:,}장 = 정상의 {out_n / max(len(t_nor), 1):.1%}"
          f"   ← v1~v4는 0장이었다")

    print("\n" + "=" * 60)
    print("검증셋  valset_bycode   (야장 우선, 없으면 원본)")
    print("=" * 60)
    print(f"  결함 {len(v_def):>5}   정상 {len(v_nor):>5}   "
          f"비율 1:{len(v_nor) / max(len(v_def), 1):.2f}")

    print("\n  [결함 — 코드별 · 출처]")
    by = Counter((r["code"], r.get("from", "?")) for r in v_def)
    for code in sorted({r["code"] for r in v_def}):
        ya = by.get((code, "yajang"), 0)
        etc = sum(v for (c, f), v in by.items() if c == code and f != "yajang")
        print(f"    {code:<7}야장 {ya:>3}  원본 {etc:>3}   합 {ya + etc:>3}")

    print("\n  [정상 — 코드별 · 출처]")
    byn = Counter((r["code"], r.get("from", "?")) for r in v_nor)
    for code in sorted({r["code"] for r in v_nor}):
        ya = byn.get((code, "yajang"), 0)
        etc = sum(v for (c, f), v in byn.items() if c == code and f != "yajang")
        print(f"    {code:<12}야장 {ya:>3}  AIHub {etc:>3}   합 {ya + etc:>3}")

    print("\n" + "=" * 60)
    print("검산")
    print("=" * 60)
    val_names = {Path(r["src"]).name for r in va}
    train_names = set()
    for r in tr:
        n = Path(r["dst"]).name
        train_names.add(n.split("__", 1)[1] if "__" in n else n)
    dup = val_names & train_names
    print(f"  겹침 {len(dup)}장 {'← 문제!' if dup else '(없음)'}"
          f"   검증 {len(val_names)} · 학습 {len(train_names)}")

    ya_def = sum(1 for r in v_def if r.get("from", "?") == "yajang")
    ya_nor = sum(1 for r in v_nor if r.get("from", "?") == "yajang")
    print(f"  검증 야장 비중 — 결함 {ya_def}/{len(v_def)} ({ya_def/max(len(v_def),1):.0%}) · "
          f"정상 {ya_nor}/{len(v_nor)} ({ya_nor/max(len(v_nor),1):.0%})")

    tc = set(codes)
    vc = {r["code"] for r in v_def}
    if vc - tc:
        print(f"  검증에만 있는 코드: {sorted(vc - tc)}  ← 학습 0장")
    if tc - vc:
        print(f"  학습에만 있는 코드: {sorted(tc - vc)}  ← 검증 불가")


if __name__ == "__main__":
    main()
