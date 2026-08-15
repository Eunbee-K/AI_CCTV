"""**앱 전체 구조**(분류기 + YOLO 병렬)를 테스트셋으로 채점한다.

score_deployed.py는 분류기만 봤다. 실제 앱은 YOLO가 나란히 이름을 붙이므로
둘을 합친 값이 사용자가 보는 성능이다.

앱의 이름 합치기 규칙을 그대로 따른다(analysis.py `_build_lead_rows`)
    YOLO가 붙였으면 YOLO를 앞에, 분류기가 추가로 본 것을 뒤에.
    YOLO가 못 붙였으면 분류기 것을 쓴다.
    -> 정답이 후보 안에 있으면 맞은 것으로 센다. 검수자는 후보에서 고르면 되고,
       빈 칸에서 직접 찾는 것보다 훨씬 빠르다. 다만 후보가 많아지면 고르는
       비용도 커지므로 **평균 후보 개수**를 같이 본다.

YOLO는 17종만 안다(test5). 분류기에만 있는 9종(BC DE DF DG IF LD PO RT TO)은
YOLO가 구조적으로 못 맞힌다 — 그 종류에서 분류기가 유일한 판독자다.

사용:
    python score_deployed_full.py
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
from paths import FILTER_DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="testset_bycode")
    ap.add_argument("--yolo-conf", type=float, default=None)
    args = ap.parse_args()

    root = FILTER_DATA / args.testset
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8-sig")))
    files = [root / r["label"] / r["file"] for r in rows]

    from backend import defect_classifier as dc
    from backend.config import CLASSIFIER_MIN_CONF, YOLO_CONF, YOLO_MODEL_PATH
    conf = args.yolo_conf if args.yolo_conf is not None else YOLO_CONF

    ok, why = dc.availability()
    if not ok:
        raise SystemExit(f"분류기를 못 엽니다: {why}")

    n_def = sum(1 for r in rows if r["label"] == "defect")
    n_nor = len(rows) - n_def
    print(f"테스트셋 {args.testset}: 결함 {n_def}장 · 정상 {n_nor}장")
    print(f"분류기 {dc.CLASSIFIER_MODEL_PATH.name} (임계값 {CLASSIFIER_MIN_CONF})")

    from ultralytics import YOLO
    ym = YOLO(str(YOLO_MODEL_PATH))
    print(f"YOLO {YOLO_MODEL_PATH.name} · {len(ym.names)}종 (conf {conf})\n")

    _, cls_names = dc.analyze(files)

    # YOLO 예측 — 박스가 하나라도 있으면 그 이름들
    yolo_names: dict[str, list[str]] = {}
    B = 32
    for i in range(0, len(files), B):
        chunk = [str(f) for f in files[i:i + B]]
        for f, r in zip(files[i:i + B], ym.predict(chunk, conf=conf, verbose=False)):
            got, seen = [], set()
            for b in r.boxes:
                nm = ym.names[int(b.cls)]
                if nm not in seen:
                    seen.add(nm)
                    got.append(nm)
            if got:
                yolo_names[str(f)] = got

    ycls = set(ym.names.values())
    stat = defaultdict(lambda: defaultdict(int))
    cand_total, cand_rows = 0, 0

    for r, f in zip(rows, files):
        k, code = str(f), r["code"]
        y = yolo_names.get(k, [])
        c = cls_names.get(k)
        c_list = [c[0]] if c else []
        merged = y + [x for x in c_list if x not in y]
        s = stat[code if r["label"] == "defect" else f"(정상){code}"]
        s["n"] += 1
        if r["label"] != "defect":
            # 정상에 이름을 붙였으면 오탐
            s["yolo_fp"] += bool(y)
            s["cls_fp"] += bool(c_list)
            s["any_fp"] += bool(merged)
            continue
        s["yolo_named"] += bool(y)
        s["cls_named"] += bool(c_list)
        s["named"] += bool(merged)
        s["yolo_hit"] += code in y
        s["cls_hit"] += code in c_list
        s["hit"] += code in merged
        s["yolo_can"] += code in ycls
        cand_total += len(merged)
        cand_rows += 1

    print("=" * 76)
    print("결함별 — 정답이 후보에 들어갔나 (YOLO / 분류기 / 합침)")
    print("=" * 76)
    print(f"  {'코드':<6}{'장수':>5}{'Y가능':>7}{'YOLO':>8}{'분류기':>9}"
          f"{'합침':>8}{'이름없음':>9}")
    defect_codes = [c for c in stat if not c.startswith("(정상)")]
    tot = defaultdict(int)
    for code in sorted(defect_codes, key=lambda c: stat[c]["hit"] / stat[c]["n"]):
        s = stat[code]
        n = s["n"]
        for k in ("n", "yolo_hit", "cls_hit", "hit", "named"):
            tot[k] += s[k]
        can = "O" if s["yolo_can"] else "-"
        print(f"  {code:<6}{n:>5}{can:>7}{s['yolo_hit']/n:>8.0%}"
              f"{s['cls_hit']/n:>9.0%}{s['hit']/n:>8.0%}{(n-s['named'])/n:>9.0%}")

    print(f"\n  전체 {tot['n']}장")
    print(f"    YOLO 단독   {tot['yolo_hit']:>4}장 {tot['yolo_hit']/tot['n']:>6.1%}")
    print(f"    분류기 단독 {tot['cls_hit']:>4}장 {tot['cls_hit']/tot['n']:>6.1%}")
    print(f"    둘 합침     {tot['hit']:>4}장 {tot['hit']/tot['n']:>6.1%}"
          f"   <- 앱이 실제로 내는 값")
    print(f"    아무 이름도 없음 {tot['n']-tot['named']:>3}장 "
          f"{(tot['n']-tot['named'])/tot['n']:>5.1%} (확인필요로 남는다)")
    print(f"    평균 후보 {cand_total/max(cand_rows,1):.2f}개/장")

    print("\n" + "=" * 76)
    print("정상 200장에 이름을 잘못 붙였나 (오탐)")
    print("=" * 76)
    print(f"  {'종류':<12}{'장수':>5}{'YOLO':>8}{'분류기':>9}{'합침':>8}")
    ft = defaultdict(int)
    for code in sorted(c for c in stat if c.startswith("(정상)")):
        s = stat[code]
        n = s["n"]
        for k in ("n", "yolo_fp", "cls_fp", "any_fp"):
            ft[k] += s[k]
        print(f"  {code[4:]:<12}{n:>5}{s['yolo_fp']/n:>8.0%}"
              f"{s['cls_fp']/n:>9.0%}{s['any_fp']/n:>8.0%}")
    print(f"\n  전체 {ft['n']}장 · YOLO {ft['yolo_fp']} "
          f"({ft['yolo_fp']/ft['n']:.1%}) · 분류기 {ft['cls_fp']} "
          f"({ft['cls_fp']/ft['n']:.1%}) · 합침 {ft['any_fp']} "
          f"({ft['any_fp']/ft['n']:.1%})")

    print("\n" + "=" * 76)
    print("YOLO가 아는 17종만 따로 (공정한 비교)")
    print("=" * 76)
    a = defaultdict(int)
    for code in defect_codes:
        if stat[code]["yolo_can"]:
            for k in ("n", "yolo_hit", "cls_hit", "hit"):
                a[k] += stat[code][k]
    if a["n"]:
        print(f"  {a['n']}장 · YOLO {a['yolo_hit']/a['n']:.1%} · "
              f"분류기 {a['cls_hit']/a['n']:.1%} · 합침 {a['hit']/a['n']:.1%}")
    b = defaultdict(int)
    for code in defect_codes:
        if not stat[code]["yolo_can"]:
            for k in ("n", "yolo_hit", "cls_hit", "hit"):
                b[k] += stat[code][k]
    if b["n"]:
        print(f"\n  YOLO가 모르는 종류 {b['n']}장 "
              f"(BC DE DF DG IF LD PO RT TO 등)")
        print(f"    YOLO {b['yolo_hit']/b['n']:.1%} · "
              f"분류기 {b['cls_hit']/b['n']:.1%} · 합침 {b['hit']/b['n']:.1%}"
              f"   <- 분류기가 유일한 판독자")


if __name__ == "__main__":
    main()
