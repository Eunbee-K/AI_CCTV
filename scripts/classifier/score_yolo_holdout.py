"""유출 없는 판에서 YOLO를 두 잣대로 잰다 — (1) 결함/정상 구분 (2) 이름 정확도.

왜 두 개로 나누나
    "결함이 있다는 건 아는데 이름을 못 붙이는" 것과 "아예 못 보는" 것은
    실무에서 의미가 완전히 다르다. 앞은 검수자가 이름만 고르면 되고,
    뒤는 그 구간 자체가 조사표에서 빠진다.

    1. 결함/정상 구분 — **이름이 틀려도** 뭐라도 검출했으면 "결함이라 봤다"로 센다
       (정상 사진에서 뭐라도 검출하면 오탐)
    2. 이름 정확도 — 정답 코드를 실제로 맞혔나

**모델이 학습하지 않은 클래스는 이름 정확도에서 제외한다** — 낼 수 없는 답을
못 맞혔다고 세면 성적이 부당하게 깎인다. 표에 "모름"으로 따로 표시한다.

사용:
    python score_yolo_holdout.py --model test5
    python score_yolo_holdout.py --model test3 --data test3_holdout_pure
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FILTER_DATA, RESULTS  # noqa: E402

WEIGHTS = {
    "test5": RESULTS / "test5_17class_sweep01" / "test5_960_e50" / "weights" / "best.pt",
    "test3": RESULTS / "test3_18class_sweep01" / "test3_960_e50" / "weights" / "best.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(WEIGHTS), default="test5")
    ap.add_argument("--data", default="")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    root = FILTER_DATA / (args.data or f"{args.model}_holdout_pure")
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8-sig")))
    files = [root / "images" / r["file"] for r in rows]
    n_def = sum(1 for r in rows if r["label"] == "defect")
    print(f"검증셋 {root.name}: 결함 {n_def}장 · 정상 {len(rows)-n_def}장")

    from ultralytics import YOLO
    m = YOLO(str(WEIGHTS[args.model]))
    known = set(m.names.values())
    print(f"모델 {args.model} · {len(known)}종 (conf {args.conf}, imgsz {args.imgsz})\n")

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
            pred[str(f)] = got

    # ── 1. 결함/정상 구분 (이름 무관) ─────────────────────────────
    det_by_code = defaultdict(lambda: [0, 0])     # 뭐라도 검출 / 전체
    hit_by_code = defaultdict(lambda: [0, 0])     # 이름 정답 / 전체
    fp_by_code = defaultdict(lambda: [0, 0])      # 정상인데 검출 / 전체
    for r, f in zip(rows, files):
        got = pred.get(str(f), [])
        if r["label"] == "defect":
            d = det_by_code[r["code"]]
            d[1] += 1
            d[0] += bool(got)
            h = hit_by_code[r["code"]]
            h[1] += 1
            h[0] += int(r["code"] in got)
        else:
            fp = fp_by_code[r["code"]]
            fp[1] += 1
            fp[0] += bool(got)

    print("=" * 70)
    print("1. 결함/정상 구분 — 이름이 틀려도 '뭐라도 검출했나'")
    print("=" * 70)
    print(f"  {'코드':<7}{'장수':>5}{'검출':>7}{'검출률':>9}   {'모델이 아는 종류인가'}")
    tot_k, tot_u = [0, 0], [0, 0]
    for code in sorted(det_by_code, key=lambda c: -det_by_code[c][0] / det_by_code[c][1]):
        hit, n = det_by_code[code]
        k = code in known
        (tot_k if k else tot_u)[0] += hit
        (tot_k if k else tot_u)[1] += n
        print(f"  {code:<7}{n:>5}{hit:>7}{hit/n:>9.0%}   {'O' if k else '- (학습 안 함)'}")

    if tot_k[1]:
        print(f"\n  아는 종류 소계   {tot_k[0]}/{tot_k[1]} = {tot_k[0]/tot_k[1]:.1%}")
    if tot_u[1]:
        print(f"  모르는 종류 소계 {tot_u[0]}/{tot_u[1]} = {tot_u[0]/tot_u[1]:.1%}"
              f"   ← 이름은 못 대도 '뭔가 있다'고는 보는가")

    print(f"\n  {'정상 종류':<12}{'장수':>5}{'오탐':>7}{'오탐률':>9}")
    fa, fn = 0, 0
    for code in sorted(fp_by_code):
        hit, n = fp_by_code[code]
        fa += hit
        fn += n
        print(f"  {code:<12}{n:>5}{hit:>7}{hit/n:>9.0%}")
    if fn:
        print(f"  {'전체':<12}{fn:>5}{fa:>7}{fa/fn:>9.0%}")

    # ── 2. 이름 정확도 (아는 종류만) ──────────────────────────────
    print("\n" + "=" * 70)
    print("2. 결함별 이름 정확도 — 모델이 학습한 종류만")
    print("=" * 70)
    print(f"  {'코드':<7}{'장수':>5}{'정답':>7}{'정확도':>9}{'검출률':>9}{'이름만 틀림':>12}")
    t = [0, 0]
    for code in sorted(hit_by_code, key=lambda c: -hit_by_code[c][0] / hit_by_code[c][1]):
        if code not in known:
            continue
        hit, n = hit_by_code[code]
        det = det_by_code[code][0]
        t[0] += hit
        t[1] += n
        print(f"  {code:<7}{n:>5}{hit:>7}{hit/n:>9.0%}{det/n:>9.0%}{(det-hit)/n:>12.0%}")
    if t[1]:
        print(f"\n  이름 정확도 합계 {t[0]}/{t[1]} = {t[0]/t[1]:.1%}")
        dk = sum(det_by_code[c][0] for c in hit_by_code if c in known)
        print(f"  (같은 종류 검출률 {dk}/{t[1]} = {dk/t[1]:.1%} — "
              f"차이 {(dk-t[0])/t[1]:.1%}가 '보긴 봤는데 이름을 틀린' 몫)")

    skipped = [c for c in hit_by_code if c not in known]
    if skipped:
        print(f"\n  이름 채점에서 제외(모델이 학습 안 한 클래스): {', '.join(sorted(skipped))}")


if __name__ == "__main__":
    main()
