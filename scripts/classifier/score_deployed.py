"""**앱에 배포된 구조 그대로** 테스트셋을 채점한다.

왜 또 만드나
    score_testset_bycode.py는 체크포인트(.pt)를 직접 연다. 그건 "모델이 얼마나
    좋은가"를 재는 것이고, 여기서는 **앱이 실제로 내는 답**을 잰다. 사이에
    ONNX 변환·전처리·임계값(CLASSIFIER_MIN_CONF)이 끼어 있어 값이 달라질 수
    있고, 실제로 그 어긋남 때문에 한 번 크게 헤맨 적이 있다(2026-08-12).

    그래서 `backend.defect_classifier.analyze()`를 그대로 부른다 — 앱이 부르는
    바로 그 함수다.

무엇을 재나
    1) 정상/결함 구분  AUC · 오탐 고정 지점의 재현율 · 임계값별 표
    2) 결함별 재현율   어떤 종류를 놓치는가
    3) 이름 정확도     붙인 이름이 맞는가 (앱 임계값 0.9 적용 전/후)

사용:
    python score_deployed.py
    python score_deployed.py --testset valset_bycode
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


def auc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    v = [x for x, _ in s]
    r, i = {}, 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        for k in range(i, j + 1):
            r[k] = (i + j) / 2 + 1
        i = j + 1
    rs = sum(r[k] for k, (_, lab) in enumerate(s) if lab)
    return (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="testset_bycode")
    args = ap.parse_args()

    root = FILTER_DATA / args.testset
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8-sig")))
    files = [root / r["label"] / r["file"] for r in rows]

    from backend import defect_classifier as dc
    from backend.config import CLASSIFIER_MIN_CONF
    ok, why = dc.availability()
    if not ok:
        raise SystemExit(f"분류기를 못 엽니다: {why}")

    n_def = sum(1 for r in rows if r["label"] == "defect")
    n_nor = len(rows) - n_def
    print(f"테스트셋 {args.testset}: 결함 {n_def}장 · 정상 {n_nor}장")
    print(f"모델 {dc.CLASSIFIER_MODEL_PATH.name} · {len(dc._classes)}클래스 · "
          f"img {dc._size} · 이름 임계값 {CLASSIFIER_MIN_CONF}\n")

    probs, names = dc.analyze(files)

    is_def, score, code = [], [], []
    for r, f in zip(rows, files):
        if str(f) not in probs:
            continue
        is_def.append(r["label"] == "defect")
        score.append(probs[str(f)])
        code.append(r["code"])
    pos = [s for s, d in zip(score, is_def) if d]
    neg = [s for s, d in zip(score, is_def) if not d]

    print("=" * 64)
    print("1. 정상/결함 구분")
    print("=" * 64)
    print(f"  AUC {auc(pos, neg):.4f}")
    print(f"\n  {'정상 오탐':>9}{'임계값':>10}{'결함 재현율':>13}{'놓친 결함':>11}")
    for fpr in (0.01, 0.02, 0.05, 0.10, 0.20):
        thr = sorted(neg, reverse=True)[max(0, int(len(neg) * fpr) - 1)]
        rec = sum(1 for s in pos if s >= thr) / len(pos)
        print(f"  {fpr:>8.0%}{thr:>10.4f}{rec:>13.1%}{int(len(pos)*(1-rec)):>11}")

    print(f"\n  {'결함 재현율':>11}{'임계값':>10}{'정상 오탐':>11}")
    for want in (0.95, 0.98, 0.99, 1.00):
        thr = sorted(pos, reverse=True)[min(len(pos) - 1, int(len(pos) * want) - 1)]
        fp = sum(1 for s in neg if s >= thr) / len(neg)
        print(f"  {want:>10.0%}{thr:>10.4f}{fp:>11.1%}")

    # 앱이 실제로 쓰는 방식은 순위 자르기다
    print(f"\n  [앱 방식] 상위 N%를 표에 올릴 때 결함이 얼마나 남나")
    order = sorted(range(len(score)), key=lambda i: -score[i])
    for ratio in (0.10, 0.20, 0.30):
        k = max(1, round(len(score) * ratio))
        kept = sum(1 for i in order[:k] if is_def[i])
        print(f"    상위 {ratio:>4.0%} ({k:>3}장) -> 결함 {kept}/{len(pos)} "
              f"({kept/len(pos):.1%}) · 이 판은 결함이 {len(pos)/len(score):.0%}라 "
              f"상한이 {min(k,len(pos))}장")

    print("\n" + "=" * 64)
    print("2. 결함별 재현율 (정상 오탐 5% 지점)")
    print("=" * 64)
    thr5 = sorted(neg, reverse=True)[max(0, int(len(neg) * 0.05) - 1)]
    by = defaultdict(list)
    for s, d, c in zip(score, is_def, code):
        by[(d, c)].append(s)
    worst = []
    for (d, c), vs in by.items():
        if d:
            worst.append((sum(1 for v in vs if v >= thr5) / len(vs), c, len(vs),
                          sorted(vs)[len(vs) // 2], auc(vs, neg)))
    print(f"  {'코드':<7}{'장수':>5}{'재현율':>9}{'중앙값':>10}{'AUC':>9}")
    for rec, c, n, med, a in sorted(worst):
        print(f"  {c:<7}{n:>5}{rec:>9.0%}{med:>10.4f}{a:>9.4f}"
              + ("  ←" if rec < 0.9 else ""))

    print(f"\n  {'정상 종류':<12}{'장수':>5}{'오탐률':>9}{'중앙값':>10}")
    for (d, c), vs in sorted(by.items()):
        if not d:
            fp = sum(1 for v in vs if v >= thr5) / len(vs)
            print(f"  {c:<12}{len(vs):>5}{fp:>9.0%}{sorted(vs)[len(vs)//2]:>10.4f}")

    print("\n" + "=" * 64)
    print(f"3. 이름 정확도 (앱 임계값 {CLASSIFIER_MIN_CONF} 적용)")
    print("=" * 64)
    hit = defaultdict(lambda: [0, 0, 0])      # [맞음, 이름붙음, 전체]
    for r, f in zip(rows, files):
        if r["label"] != "defect":
            continue
        h = hit[r["code"]]
        h[2] += 1
        got = names.get(str(f))
        if got:
            h[1] += 1
            h[0] += int(got[0] == r["code"])
    print(f"  {'코드':<7}{'장수':>5}{'이름붙음':>10}{'맞음':>7}{'정확도':>9}{'정밀도':>9}")
    for c, (h, n1, n2) in sorted(hit.items(), key=lambda kv: kv[1][0] / kv[1][2]):
        print(f"  {c:<7}{n2:>5}{n1:>10}{h:>7}{h/n2:>9.0%}{h/max(n1,1):>9.0%}")
    H = sum(v[0] for v in hit.values())
    N1 = sum(v[1] for v in hit.values())
    N2 = sum(v[2] for v in hit.values())
    print(f"\n  전체: {N2}장 중 {N1}장에 이름 · {H}장 정답")
    print(f"    정확도(전체 기준) {H/N2:.1%}   정밀도(이름 붙인 것 중) {H/max(N1,1):.1%}")
    fp_name = sum(1 for r, f in zip(rows, files)
                  if r["label"] == "normal" and str(f) in names)
    print(f"    정상에 잘못 붙인 이름 {fp_name}/{n_nor}장 ({fp_name/max(n_nor,1):.1%})")


if __name__ == "__main__":
    main()
