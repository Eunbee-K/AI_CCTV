"""공통 검증셋에서 **필터C와 test3를 나란히** 잰다 — 상호보완성을 보기 위해.

두 모델을 같은 사진으로 재는 것이 요점이다. 지금까지는 판이 서로 달라
"필터 96% vs YOLO 36%"를 나란히 놓을 수 없었다(`build_common_holdout.py` 참고).

재는 것 (결함코드마다)
    CLS 탐지율     필터가 결함이라 판정했나 (정상 오탐 5% 지점 기준)
    CLS 이름정확도  필터가 종류까지 맞혔나 (확신 0.9 이상일 때만 이름을 낸다)
    DET 탐지율     YOLO가 뭐라도 검출했나 (이름이 틀려도 인정)
    DET 이름정확도  YOLO가 종류까지 맞혔나
    합집합         둘 중 하나라도 맞혔나  <- 상호보완의 증거
    겹침           둘 다 맞혔나

**평균은 측정 가능한 종류에서만 낸다.** YOLO가 클래스를 갖고 있지 않은 종류
(BC·DF·DG·LD)를 "0%"로 평균에 넣으면 성능이 아니라 커버리지를 성능처럼 재게 된다.

사용:
    python score_common_holdout.py
    python score_common_holdout.py --xlsx        # 엑셀도 함께 저장
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "AI_CCTV"))
from paths import FILTER_DATA, RESULTS  # noqa: E402

YOLO_W = RESULTS / "test3_18class_sweep01" / "test3_960_e50" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="common_holdout")
    ap.add_argument("--fpr", type=float, default=0.05, help="정상 오탐을 이 값으로 고정")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--xlsx", action="store_true")
    args = ap.parse_args()

    root = FILTER_DATA / args.data
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8-sig")))
    files = [root / "images" / r["file"] for r in rows]
    n_def = sum(1 for r in rows if r["label"] == "defect")
    print(f"검증셋 {args.data}: 결함 {n_def}장 · 정상 {len(rows)-n_def}장")

    # ── 필터C ──────────────────────────────────────────────
    from backend import defect_classifier as dc
    ok, why = dc.availability()
    if not ok:
        raise SystemExit(f"분류기를 못 엽니다: {why}")
    probs, cls_names = dc.analyze(files)

    # ── test3 ──────────────────────────────────────────────
    from ultralytics import YOLO
    ym = YOLO(str(YOLO_W))
    yolo_known = set(ym.names.values())
    print(f"필터C {len(dc._classes)}클래스 · test3 {len(yolo_known)}클래스\n")

    yolo_pred: dict[str, list[str]] = {}
    B = 32
    for i in range(0, len(files), B):
        chunk = [str(f) for f in files[i:i + B]]
        for f, r in zip(files[i:i + B],
                        ym.predict(chunk, conf=args.conf, imgsz=args.imgsz, verbose=False)):
            got, seen = [], set()
            for b in r.boxes:
                nm = ym.names[int(b.cls)]
                if nm not in seen:
                    seen.add(nm)
                    got.append(nm)
            yolo_pred[str(f)] = got

    # 정상 오탐 5%가 되는 필터 임계값
    neg = [probs[str(f)] for r, f in zip(rows, files)
           if r["label"] == "normal" and str(f) in probs]
    thr = sorted(neg, reverse=True)[max(0, int(len(neg) * args.fpr) - 1)]

    # ── 집계 ───────────────────────────────────────────────
    st = defaultdict(lambda: defaultdict(int))
    for r, f in zip(rows, files):
        k, code = str(f), r["code"]
        s = st[code]
        s["n"] += 1
        cls_hit = probs.get(k, 0.0) >= thr                       # 결함이라 봤나
        cls_name = cls_names.get(k, (None,))[0] == code          # 종류까지 맞혔나
        det_hit = bool(yolo_pred.get(k))
        det_name = code in yolo_pred.get(k, [])
        if r["label"] == "defect":
            s["cls_hit"] += cls_hit
            s["cls_name"] += cls_name
            s["det_hit"] += det_hit
            s["det_name"] += det_name
            s["union"] += (cls_name or det_name)
            s["both"] += (cls_name and det_name)
            s["cls_only"] += (cls_name and not det_name)
            s["det_only"] += (det_name and not cls_name)
            s["yolo_can"] = int(code in yolo_known)
        else:
            s["cls_fp"] += cls_hit
            s["det_fp"] += det_hit

    defect_codes = [c for c in st if any(r["code"] == c and r["label"] == "defect"
                                         for r in rows)]
    both_known = sorted(c for c in defect_codes if st[c]["yolo_can"])
    cls_only_codes = sorted(c for c in defect_codes if not st[c]["yolo_can"])

    def pct(a, b):
        return f"{a/b:.0%}" if b else "-"

    print("=" * 86)
    print("결함별 — 필터C vs test3 (같은 사진, 두 모델 학습분 모두 제외)")
    print("=" * 86)
    print(f"  {'코드':<6}{'장수':>5}{'CLS탐지':>9}{'CLS이름':>9}"
          f"{'DET탐지':>9}{'DET이름':>9}{'합집합':>9}{'겹침':>7}  비고")
    for group, label in ((both_known, "둘 다 아는 종류"),
                         (cls_only_codes, "필터C만 아는 종류 (YOLO에 클래스 없음)")):
        if not group:
            continue
        print(f"  --- {label} ---")
        for c in group:
            s = st[c]
            n = s["n"]
            note = "" if s["yolo_can"] else "YOLO 미학습"
            print(f"  {c:<6}{n:>5}{pct(s['cls_hit'],n):>9}{pct(s['cls_name'],n):>9}"
                  f"{pct(s['det_hit'],n):>9}{pct(s['det_name'],n):>9}"
                  f"{pct(s['union'],n):>9}{pct(s['both'],n):>7}  {note}")

    def avg(codes, key):
        a = sum(st[c][key] for c in codes)
        b = sum(st[c]["n"] for c in codes)
        return a, b

    print(f"\n  [둘 다 아는 {len(both_known)}종] — 공정 비교 구간")
    for key, label in (("cls_hit", "CLS 탐지율"), ("cls_name", "CLS 이름정확도"),
                       ("det_hit", "DET 탐지율"), ("det_name", "DET 이름정확도"),
                       ("union", "합집합(둘 중 하나)"), ("both", "겹침(둘 다)")):
        a, b = avg(both_known, key)
        print(f"    {label:<18}{a:>4}/{b:<5} {a/b:.1%}")
    a_c, _ = avg(both_known, "cls_only")
    a_d, b = avg(both_known, "det_only")
    print(f"    {'CLS만 맞힘':<18}{a_c:>4}/{b:<5} {a_c/b:.1%}")
    print(f"    {'DET만 맞힘':<18}{a_d:>4}/{b:<5} {a_d/b:.1%}")

    if cls_only_codes:
        a, b = avg(cls_only_codes, "cls_name")
        print(f"\n  [필터C만 아는 {len(cls_only_codes)}종] CLS 이름정확도 "
              f"{a}/{b} = {a/b:.1%}  (YOLO는 클래스가 없어 구조적으로 0%)")

    print(f"\n  {'정상 종류':<12}{'장수':>5}{'CLS오탐':>9}{'DET오탐':>9}")
    for c in sorted(st):
        s = st[c]
        if s["n"] and c not in defect_codes:
            print(f"  {c:<12}{s['n']:>5}{pct(s['cls_fp'],s['n']):>9}{pct(s['det_fp'],s['n']):>9}")

    # 채점은 CPU로 40분쯤 걸린다. 표 서식만 고치려고 다시 돌리지 않도록 저장해둔다.
    import json
    cache = root / "_scores.json"
    cache.write_text(json.dumps(
        {"st": {k: dict(v) for k, v in st.items()},
         "both_known": both_known, "cls_only": cls_only_codes,
         "defect_codes": defect_codes, "thr": thr}, ensure_ascii=False), encoding="utf-8")

    if args.xlsx:
        _write_xlsx(st, both_known, cls_only_codes, defect_codes, thr, args)


def _write_xlsx(st, both_known, cls_only_codes, defect_codes, thr, args):
    """31종 전체를 놓되 이 판으로 잰 12종만 굵게. 나머지는 기존 판 참고치를 병기한다."""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    # 이 판으로 못 재는 19종 — 기존 판의 값(비교 아님, 각자의 값)
    # 필터C는 testset_bycode(23종×20장), test3는 test3_holdout(야장 80%) 기준.
    REF = {  # code: (CLS 이름정확도, DET 재현율, 비고)
        "CM": (0.60, 0.00, "야장 20장"), "HL": (0.50, 0.00, "야장 3장"),
        "JS": (0.00, 0.00, "야장 20장"), "LS": (0.45, 0.30, "야장 20장"),
        "PO": (0.30, 0.00, "야장 20장"), "RT": (0.65, 0.00, "야장 19장"),
        "SG": (0.95, 0.00, "야장 10장"), "TO": (0.75, 0.80, "야장 20장"),
        "DE": (0.93, None, "YOLO 미학습"), "IF": (1.00, None, "YOLO 미학습"),
        "CX": (None, None, "데이터 없음"), "NS": (None, None, "데이터 없음"),
        "PB": (None, None, "데이터 없음"), "ETC": (0.00, None, "기타(결함 종류 아님)"),
    }
    ALL31 = ["CC", "CL", "CM", "SD", "BC", "LD", "DF", "BK", "CX", "PO", "HL",
             "LP", "LS", "JS", "JF", "JD", "NS", "SG", "DE", "DS", "DG", "TO",
             "RT", "IF", "PB", "ETC"]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "필터C vs test3"
    bold = Font(bold=True)
    hdr = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="4472C4")
    grey = Font(color="999999")
    note_f = Font(italic=True, size=9, color="666666")
    center = Alignment(horizontal="center", vertical="center")

    ws["A1"] = "결함별 성능 — 필터C(분류기) vs test3(YOLO)"
    ws["A1"].font = Font(bold=True, size=13)
    ws.merge_cells("A1:H1")
    ws["A2"] = ("굵은 글씨 = 공통 검증셋(common_holdout, 두 모델 학습분 모두 제외)으로 "
                "직접 비교한 12종 · 회색 = 이 판으로 측정 불가, 기존 판 참고치")
    ws["A2"].font = note_f
    ws.merge_cells("A2:H2")

    heads = ["결함코드", "장수", "CLS 탐지율", "CLS 이름정확도",
             "DET 탐지율", "DET 이름정확도", "합집합", "비고"]
    for i, h in enumerate(heads, 1):
        c = ws.cell(row=4, column=i, value=h)
        c.font = hdr; c.fill = fill; c.alignment = center

    def put(row, col, value, *, font=None, fmt=None):
        """숫자 서식을 확실히 건다.

        30/30 같은 값이 int 1로 저장되면 도구에 따라 '1'로 읽힌다(엑셀 자체는
        0% 서식이라 100%로 보이지만, 다른 프로그램에서 열면 다르다). 비율 칸은
        항상 float으로 넣는다.
        """
        if fmt and isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
        c = ws.cell(row=row, column=col, value=value)
        c.alignment = center
        if font:
            c.font = font
        if fmt and isinstance(value, (int, float)):
            c.number_format = fmt
        return c

    r = 5
    measured = set(both_known) | set(cls_only_codes)
    for code in ALL31:
        put(r, 1, code, font=(bold if code in measured else None))
        if code in measured:
            s = st[code]; n = s["n"]
            put(r, 2, n, font=bold)
            for col, key in ((3, "cls_hit"), (4, "cls_name"),
                             (5, "det_hit"), (6, "det_name"), (7, "union")):
                put(r, col, s[key] / n, font=bold, fmt="0%")
            note = "" if s["yolo_can"] else "YOLO 미학습(구조적 0%)"
            ws.cell(row=r, column=8, value=note).font = note_f
        else:
            cls_v, det_v, note = REF.get(code, (None, None, "미측정"))
            put(r, 2, "-")
            put(r, 3, "-")                                   # CLS 탐지율(미측정)
            put(r, 4, cls_v if cls_v is not None else "-", font=grey, fmt="0%")
            put(r, 5, "-")                                   # DET 탐지율(미측정)
            put(r, 6, det_v if det_v is not None else "-", font=grey, fmt="0%")
            put(r, 7, "-")                                   # 합집합(미측정)
            ws.cell(row=r, column=8, value=f"참고치 · {note}").font = note_f
        r += 1

    # ── 평균 ──────────────────────────────────────────────
    # 12종 전체가 이 판의 대표값이다. 8종만 따로 내는 것은 "같은 종류를 두고
    # 겨루면 어떤가"를 보기 위한 보조 지표다.
    def block(codes, title):
        nonlocal r
        r += 1
        ws.cell(row=r, column=1, value=title).font = bold
        r += 1
        for key, label in (("cls_hit", "CLS 탐지율"), ("cls_name", "CLS 이름정확도"),
                           ("det_hit", "DET 탐지율"), ("det_name", "DET 이름정확도"),
                           ("union", "합집합(둘 중 하나라도 맞힘)"),
                           ("both", "겹침(둘 다 맞힘)"),
                           ("cls_only", "CLS만 맞힘"), ("det_only", "DET만 맞힘")):
            a = sum(st[c][key] for c in codes)
            b = sum(st[c]["n"] for c in codes)
            ws.cell(row=r, column=1, value=label)
            put(r, 2, a / b, font=bold, fmt="0.0%")
            r += 1

    block(sorted(measured), f"[측정한 {len(measured)}종 전체 평균] — 이 판의 대표값")
    block(both_known, f"[둘 다 아는 {len(both_known)}종만] — 같은 종류 맞대결")

    for i, w in enumerate([11, 7, 12, 15, 12, 15, 10, 26], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A5"

    out = Path(__file__).resolve().parents[2] / "docs" / "reports" / \
        "2026-08-16-결함별-성능비교.xlsx"
    wb.save(out)
    print(f"\n엑셀 저장: {out}")


if __name__ == "__main__":
    main()
