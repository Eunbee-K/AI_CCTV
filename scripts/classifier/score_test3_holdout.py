"""test3 가중치를 유출 없는 검증셋(test3_holdout)으로 채점한다. Colab에서 돌린다.

test3_holdout은 build_test3_holdout.py가 만든다 — test3가 학습(train+val+test)에
실제로 쓴 파일을 `deterministic_pick`으로 재현해 전부 제외하고, 야장을 우선으로
16종 각 20장(부족한 3종만 못 채움) + 정상 200장을 모은 판이다(492장, 야장 80%).

바운딩박스 라벨은 없다 — 여기서는 "결함 종류를 맞히는가"만 본다(분류 정확도에
해당). test3가 실제로 위치까지 맞히는지는 어차피 여기 없는 정보이므로 재지 않는다.

사용(Colab):
    python score_test3_holdout.py --weights best.pt --data test3_holdout
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="test3_holdout")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    root = Path(args.data)
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8-sig")))
    files = [root / "images" / r["file"] for r in rows]
    print(f"검증셋 {root}: {len(rows)}장 "
          f"(결함 {sum(1 for r in rows if r['code'] not in ('PJ','IN'))} · "
          f"정상 {sum(1 for r in rows if r['code'] in ('PJ','IN'))})")
    print(f"야장 {sum(1 for r in rows if r['from']=='yajang')} · "
          f"S20/AIHub {sum(1 for r in rows if r['from']=='s20aihub')}\n")

    from ultralytics import YOLO
    m = YOLO(args.weights)
    ycls = set(m.names.values())
    print(f"모델 {args.weights} · {len(ycls)}종 (conf {args.conf}, imgsz {args.imgsz})\n")

    pred: dict[str, list[str]] = {}
    B = 32
    for i in range(0, len(files), B):
        chunk = [str(f) for f in files[i:i + B]]
        for f, r in zip(files[i:i + B],
                        m.predict(chunk, conf=args.conf, imgsz=args.imgsz, verbose=False)):
            got, seen = [], set()
            for b in r.boxes:
                nm = m.names[int(b.cls)]
                if nm not in seen:
                    seen.add(nm)
                    got.append(nm)
            if got:
                pred[str(f)] = got

    normal_codes = {"PJ", "IN"}
    by = defaultdict(lambda: defaultdict(list))   # code -> from -> [hit(0/1)]
    fp = defaultdict(lambda: defaultdict(int))    # 정상에 뭐라도 검출했나

    for r, f in zip(rows, files):
        k = str(f)
        got = pred.get(k, [])
        if r["code"] in normal_codes:
            fp[r["code"]][r["from"]] += bool(got)
            fp[r["code"]][r["from"] + "_n"] += 1
        else:
            by[r["code"]][r["from"]].append(int(r["code"] in got))

    print("=" * 62)
    print("결함별 재현율 — 야장 vs S20/AIHub")
    print("=" * 62)
    print(f"  {'코드':<7}{'야장(n)':>9}{'야장재현율':>11}{'S20/AI(n)':>11}{'재현율':>9}")
    tot = defaultdict(lambda: [0, 0])
    for code in sorted(by):
        ya = by[code].get("yajang", [])
        s2 = by[code].get("s20aihub", [])
        ya_s = f"{sum(ya)/len(ya):.0%}" if ya else "-"
        s2_s = f"{sum(s2)/len(s2):.0%}" if s2 else "-"
        print(f"  {code:<7}{len(ya):>9}{ya_s:>11}{len(s2):>11}{s2_s:>9}")
        if ya:
            tot["ya"][0] += sum(ya); tot["ya"][1] += len(ya)
        if s2:
            tot["s2"][0] += sum(s2); tot["s2"][1] += len(s2)

    print(f"\n  야장 전체:      {tot['ya'][0]}/{tot['ya'][1]} = "
          f"{tot['ya'][0]/max(tot['ya'][1],1):.1%}  (유출 걱정 없는 값)")
    if tot["s2"][1]:
        print(f"  S20/AIHub 전체: {tot['s2'][0]}/{tot['s2'][1]} = "
              f"{tot['s2'][0]/max(tot['s2'][1],1):.1%}  (test3도 안 쓴 것이라 이것도 유효)")

    print(f"\n{'='*62}\n정상(PJ/IN)에 뭐라도 검출했나 (오탐)\n{'='*62}")
    for code in normal_codes:
        for src in ("yajang", "s20aihub"):
            n = fp[code].get(src + "_n", 0)
            if n:
                h = fp[code].get(src, 0)
                print(f"  {code} ({src}): {h}/{n} = {h/n:.1%}")

    print(f"\n비교 참고 — test5(현재 배포본)를 같은 야장 사진으로 잰 값(2026-08-15):")
    print(f"  야장 기준 전체 재현율 22.5% (YOLO 아는 종류만 33.3%)")


if __name__ == "__main__":
    main()
