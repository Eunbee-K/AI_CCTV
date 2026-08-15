"""**탐지율만** 잰다 — 이름을 못 맞혀도 "결함이 있다"고는 보는가.

왜 따로 재나
    성능표에서 CM·PO·HL·LS·JS·SG·DE·TO·RT·IF 같은 종류의 탐지율 칸이 비어
    있었다. 이름정확도만 기록해뒀기 때문인데, **이름을 못 붙이는 것과 아예
    못 보는 것은 검수자에게 전혀 다른 문제다.** 앞은 드롭다운에서 고르면
    되지만, 뒤는 그 구간이 표에 없어서 놓친 것이 된다.

    YOLO가 학습하지 않은 종류(BC·DE·DF·DG·IF·LD)도 마찬가지다. 이름은
    구조적으로 못 내지만 "뭔가 있다"고는 검출할 수 있고, lead 구조에서는
    그것만으로도 행이 만들어져 값어치가 있다.

어느 판을 쓰나 — **모델마다 그 모델이 안 본 판을 쓴다**
    CLS  testset_bycode  (필터C 학습분과 안 겹침, 23종 각 20장)
    DET  test3_holdout   (test3 학습분 제외, 16종 · 야장 80%)

    두 판은 도메인도 장수도 다르므로 **CLS와 DET 값을 나란히 비교하지 말 것.**
    각자 "그 종류를 탐지는 하는가"를 보는 용도다.

사용:
    python score_detection_only.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "AI_CCTV"))
from paths import FILTER_DATA, RESULTS  # noqa: E402

YOLO_W = RESULTS / "test3_18class_sweep01" / "test3_960_e50" / "weights" / "best.pt"
NORMAL = {"PJ", "IN", "OUT_MH", "OUT_CAR", "OUT_INVERT"}


def load(name: str):
    """(rows, files) — manifest 구조가 판마다 조금 다르다."""
    root = FILTER_DATA / name
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8-sig")))
    if "label" in rows[0]:                       # testset_bycode 계열
        files = [root / r["label"] / r["file"] for r in rows]
    else:                                        # test3_holdout 계열
        files = [root / "images" / r["file"] for r in rows]
    return rows, files


def is_defect(r) -> bool:
    if "label" in r:
        return r["label"] == "defect"
    return r["code"] not in NORMAL


def score_cls(fpr: float):
    """필터C 탐지율 — testset_bycode에서. 정상 오탐을 fpr로 고정한 지점."""
    rows, files = load("testset_bycode")
    from backend import defect_classifier as dc
    ok, why = dc.availability()
    if not ok:
        raise SystemExit(f"분류기를 못 엽니다: {why}")
    probs, _ = dc.analyze(files)

    neg = [probs[str(f)] for r, f in zip(rows, files)
           if not is_defect(r) and str(f) in probs]
    thr = sorted(neg, reverse=True)[max(0, int(len(neg) * fpr) - 1)]

    hit = defaultdict(lambda: [0, 0])
    for r, f in zip(rows, files):
        if not is_defect(r):
            continue
        h = hit[r["code"]]
        h[1] += 1
        h[0] += int(probs.get(str(f), 0.0) >= thr)
    return hit, thr


def score_det(conf: float, imgsz: int):
    """test3 탐지율 — test3_holdout에서. **이름이 틀려도 뭐라도 검출하면 인정.**

    정상 사진의 오탐도 같은 순회에서 센다 — 어차피 다 훑으므로 추가 비용이 없고,
    "얼마나 잡나"와 "얼마나 헛짚나"는 같이 봐야 의미가 있다.
    """
    rows, files = load("test3_holdout")
    from ultralytics import YOLO
    m = YOLO(str(YOLO_W))
    known = set(m.names.values())

    hit = defaultdict(lambda: [0, 0])      # 결함 — 검출했나
    fp = defaultdict(lambda: [0, 0])       # 정상 — 헛짚었나
    B = 32
    for i in range(0, len(files), B):
        chunk = [str(f) for f in files[i:i + B]]
        preds = m.predict(chunk, conf=conf, imgsz=imgsz, verbose=False)
        for r, f, p in zip(rows[i:i + B], files[i:i + B], preds):
            got = len(p.boxes) > 0
            tgt = hit if is_defect(r) else fp
            t = tgt[r["code"]]
            t[1] += 1
            t[0] += int(got)
    return hit, fp, known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--cls", action="store_true",
                    help="CLS도 다시 잰다(기본은 8/15 측정값 사용)")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    # CLS 탐지율은 2026-08-15에 이미 쟀다(보고서 §3.2, testset_bycode 기준).
    # 40분짜리 채점을 되풀이할 이유가 없으므로 그 값을 쓴다. --cls를 주면 재측정.
    if args.cls:
        print("CLS 탐지율 — testset_bycode (필터C가 안 본 판)")
        cls_hit, thr = score_cls(args.fpr)
        print(f"  정상 오탐 {args.fpr:.0%} 지점 임계값 {thr:.4f}\n")
    else:
        REPORTED = {"BK": .80, "ETC": .80, "CL": .90, "DS": .90, "JD": .90,
                    "CC": .95, "JF": .95, "LP": .95, "SD": .95}   # 나머지는 100%
        cls_hit = {}
        for c in ("BC BK CC CL CM DE DF DG DS ETC HL IF JD JF JS LD LP LS "
                  "PO RT SD SG TO").split():
            n = 15 if c == "DE" else 20
            cls_hit[c] = [round(REPORTED.get(c, 1.00) * n), n]
        thr = None
        print("CLS 탐지율 — 2026-08-15 측정값 (testset_bycode, 보고서 §3.2)\n")

    print("DET 탐지율 — test3_holdout (test3가 안 본 판)")
    det_hit, det_fp, known = score_det(args.conf, args.imgsz)
    print()

    codes = sorted(set(cls_hit) | set(det_hit))
    print("=" * 74)
    print("탐지율 — 이름을 못 맞혀도 '결함이 있다'고는 보는가")
    print("=" * 74)
    print(f"  {'코드':<7}{'CLS(n)':>8}{'CLS 탐지':>10}{'DET(n)':>8}{'DET 탐지':>10}"
          f"   {'YOLO 학습'}")
    for c in codes:
        ch, cn = cls_hit.get(c, [0, 0])
        dh, dn = det_hit.get(c, [0, 0])
        cs = f"{ch/cn:.0%}" if cn else "-"
        ds = f"{dh/dn:.0%}" if dn else "-"
        mark = "O" if c in known else "-"
        print(f"  {c:<7}{cn:>8}{cs:>10}{dn:>8}{ds:>10}   {mark}")

    # 정상 오탐 — DET는 이 판에서 바로 재고, CLS는 8/15 값(보고서 §3.3)을 쓴다.
    CLS_FP = {"IN": (2, 40), "PJ": (8, 120),
              "OUT_MH": (0, 18), "OUT_CAR": (0, 11), "OUT_INVERT": (0, 11)}
    print()
    print("=" * 74)
    print("정상 오탐 — 결함이 아닌데 결함이라 했나")
    print("=" * 74)
    print(f"  {'정상종류':<12}{'CLS(n)':>8}{'CLS 오탐':>10}{'DET(n)':>8}{'DET 오탐':>10}")
    for c in ("IN", "PJ", "OUT_MH", "OUT_CAR", "OUT_INVERT"):
        ch, cn = CLS_FP.get(c, (0, 0))
        dh, dn = det_fp.get(c, [0, 0])
        cs = f"{ch/cn:.0%}" if cn else "-"
        ds = f"{dh/dn:.0%}" if dn else "-"
        print(f"  {c:<12}{cn:>8}{cs:>10}{dn:>8}{ds:>10}")
    print("  (CLS는 2026-08-15 testset_bycode 측정값 · DET는 test3_holdout에서 이번에 측정)")

    out = FILTER_DATA / "_detection_rates.json"
    out.write_text(json.dumps(
        {"cls": {k: v for k, v in cls_hit.items()},
         "det": {k: v for k, v in det_hit.items()},
         "det_fp": {k: v for k, v in det_fp.items()},
         "yolo_known": sorted(known), "thr": thr,
         "cls_source": ("재측정" if args.cls else "2026-08-15 testset_bycode"),
         "det_source": "test3_holdout"},
        ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
