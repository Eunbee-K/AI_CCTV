"""결함 종류별 테스트셋 채점 — 무엇을 잡고 무엇을 놓치는가.

임계값을 하나로 못 박지 않는다. 필터의 임계값은 점수 분포에 붙어 있는 값이라
모델마다 다르다. 대신 **정상 오탐률을 고정해 놓고**(기본 5%) 그 지점에서 결함별
재현율을 본다 — 모델이 달라도 같은 잣대로 비교된다.

사용:
    python score_testset_bycode.py --model apps/AI_CCTV/assets/filter.onnx
    python score_testset_bycode.py --model a.pt b.pt --gray
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "AI_CCTV"))
from paths import DATASET, FILTER_DATA  # noqa: E402


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


def score(model: Path, files, gray: bool) -> dict:
    import os
    if model.suffix == ".onnx":
        os.environ["FILTER_MODEL_PATH"] = str(model)
        os.environ["FILTER_GRAYSCALE"] = "1" if gray else "0"
        for m in [k for k in sys.modules if k.startswith("backend")]:
            del sys.modules[m]
        from backend import defect_filter
        ok, why = defect_filter.availability()
        if not ok:
            raise SystemExit(f"필터를 열지 못했습니다: {why}")
        return defect_filter.defect_probs(files)

    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_binary import build_model
    ck = torch.load(model, map_location="cpu", weights_only=False)
    if bool(ck.get("gray", False)) != gray:
        print(f"  ! 경고: 체크포인트 gray={ck.get('gray')} 인데 --gray={gray}")
    net = build_model(ck.get("arch", "effb0"))
    net.load_state_dict(ck["model"])
    net.eval()
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
    out = {}
    with torch.no_grad():
        for i in range(0, len(files), 32):
            batch = []
            for f in files[i:i + 32]:
                im = Image.open(f).convert("RGB").resize((256, 256), Image.BILINEAR)
                im = im.resize((224, 224), Image.BILINEAR)
                if gray:
                    im = im.convert("L").convert("RGB")
                a = np.asarray(im, np.float32) / 255.0
                batch.append((a.transpose(2, 0, 1) - mean) / std)
            p = torch.softmax(net(torch.from_numpy(np.stack(batch))), 1)[:, 1]
            for f, v in zip(files[i:i + 32], p.tolist()):
                out[str(f)] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+",
                    default=[str(Path(__file__).resolve().parents[2]
                                 / "apps/AI_CCTV/assets/filter.onnx")])
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--testset", default="")
    ap.add_argument("--fpr", type=float, default=0.05,
                    help="정상 오탐률을 이 값으로 고정하고 재현율을 본다")
    args = ap.parse_args()

    root = Path(args.testset) if args.testset else FILTER_DATA / "testset_bycode"
    with open(root / "manifest.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    files = [root / r["label"] / r["file"] for r in rows]
    print(f"테스트셋: 결함 {sum(1 for r in rows if r['label']=='defect')}장 · "
          f"정상 {sum(1 for r in rows if r['label']=='normal')}장\n")

    for m in args.model:
        path = Path(m)
        if not path.exists():
            print(f"! 없는 파일: {path}")
            continue
        p = score(path, files, args.gray)

        by = defaultdict(list)
        for r, f in zip(rows, files):
            v = p.get(str(f))
            if v is not None:
                by[(r["label"], r["code"])].append(v)
        neg = [v for (lab, _), vs in by.items() if lab == "normal" for v in vs]
        pos = [v for (lab, _), vs in by.items() if lab == "defect" for v in vs]

        # 정상 오탐률을 args.fpr로 만드는 임계값
        thr = sorted(neg, reverse=True)[max(0, int(len(neg) * args.fpr) - 1)]

        print(f"{'='*62}\n{path.parent.name}/{path.name}"
              + ("  [흑백]" if args.gray else "  [컬러]") + f"\n{'='*62}")
        print(f"  전체 AUC {auc(pos, neg):.4f} · 정상 오탐 {args.fpr:.0%} 지점 임계값 {thr:.4f}")
        print(f"\n  {'결함코드':<10}{'장수':>5}{'재현율':>8}{'중앙값':>9}{'AUC':>8}")
        worst = []
        for (lab, code), vs in sorted(by.items()):
            if lab != "defect":
                continue
            rec = sum(1 for v in vs if v >= thr) / len(vs)
            med = sorted(vs)[len(vs) // 2]
            a = auc(vs, neg)
            worst.append((rec, code, len(vs), med, a))
        for rec, code, n, med, a in sorted(worst):
            mark = "  ←" if rec < 0.8 else ""
            print(f"  {code:<10}{n:>5}{rec:>8.0%}{med:>9.4f}{a:>8.4f}{mark}")

        print(f"\n  {'정상 종류':<10}{'장수':>5}{'오탐률':>8}{'중앙값':>9}")
        for (lab, code), vs in sorted(by.items()):
            if lab != "normal":
                continue
            fp = sum(1 for v in vs if v >= thr) / len(vs)
            print(f"  {code:<10}{len(vs):>5}{fp:>8.0%}{sorted(vs)[len(vs)//2]:>9.4f}")
        print()


if __name__ == "__main__":
    main()
