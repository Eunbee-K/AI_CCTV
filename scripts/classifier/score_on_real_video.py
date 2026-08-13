"""필터 후보를 **실영상**으로 채점한다 — 야장 val AUC는 실영상 성능을 예측하지 못한다.

배포본 OLD_v3는 야장 val AUC 0.9503인데 SM2 실영상에서는 0.5205였다(2026-08-12).
그러니 어느 체크포인트를 쓸지는 야장이 아니라 여기서 정해야 한다.

**무작위 대조군을 같이 찍는다.** 결함 하나가 프레임 수십 장에 걸쳐 있으면 아무
프레임이나 20% 골라도 살아남는다 — 대조군 없이 "유실 0건"만 보고 8/11에 잘못
판단했다.

정답은 `filter_eval_SM2_onnx.csv`(또는 `filter_eval_SM2.csv`)에 저장된 프레임별
`is_defect`를 그대로 쓴다. OCR로 거리 곡선을 다시 만들지 않으므로 빠르다.
그 캐시는 eval_series_vs_parallel.py가 만든다.

사용:
    # ONNX 하나 채점
    python score_on_real_video.py --model E:/.../OLD_v4_gray/best.onnx --gray

    # 체크포인트(.pt)도 받는다 — 내부에서 ONNX 없이 torch로 돌린다
    python score_on_real_video.py --model E:/.../OLD_v4_gray/best.pt --gray

    # 여러 개를 한 번에 비교
    python score_on_real_video.py --model a.onnx b.pt --gray
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "AI_CCTV"))
from paths import DATASET, SANGMYEON_0128  # noqa: E402

FRAME_STEP = 2.0        # 캐시가 2초 간격으로 만들어졌다. 정답과 맞추려면 바꾸면 안 된다.


def load_truth(prefix: str) -> dict:
    """{관로: [(초, 정답여부), ...]}. 캐시 CSV에서 읽는다."""
    for name in (f"filter_eval_{prefix}_onnx.csv", f"filter_eval_{prefix}.csv"):
        p = DATASET / name
        if p.exists():
            break
    else:
        raise SystemExit(
            f"정답 캐시가 없습니다: {DATASET}/filter_eval_{prefix}*.csv\n"
            f"먼저 eval_series_vs_parallel.py --prefix {prefix} 를 돌려 만드세요."
        )
    by = defaultdict(list)
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by[r["pipe"]].append((int(r["frame"].split(".")[0]), int(r["is_defect"])))
    print(f"정답 캐시: {p.name} · 관로 {len(by)}개 · 프레임 {sum(len(v) for v in by.values()):,}")
    return by


def extract(pipe: str, secs) -> list:
    """그 관로 영상에서 필요한 시각의 프레임만 뽑는다."""
    mp4 = next((SANGMYEON_0128 / pipe).glob("*.mp4"), None)
    if not mp4:
        return []
    out = Path(tempfile.mkdtemp())
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    for t in secs:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, im = cap.read()
        if not ok:
            continue
        fp = out / f"{t:06d}.jpg"
        cv2.imwrite(str(fp), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames.append(fp)
    cap.release()
    return frames


def score_onnx(model: Path, frames, gray: bool) -> dict:
    """앱이 쓰는 경로 그대로 채점한다 — 평가와 배포가 어긋나면 잰 값이 무효다."""
    import os
    os.environ["FILTER_MODEL_PATH"] = str(model)
    os.environ["FILTER_GRAYSCALE"] = "1" if gray else "0"
    for m in [k for k in sys.modules if k.startswith("backend")]:
        del sys.modules[m]                      # 환경변수를 다시 읽게 한다
    from backend import defect_filter
    ok, why = defect_filter.availability()
    if not ok:
        raise SystemExit(f"필터를 열지 못했습니다: {why}")
    return defect_filter.defect_probs(frames)


def score_torch(model: Path, frames, gray: bool) -> dict:
    """체크포인트를 ONNX 없이 바로 채점. 전처리는 앱과 같은 256->224."""
    import numpy as np
    import torch
    from PIL import Image
    from train_binary import build_model

    ck = torch.load(model, map_location="cpu", weights_only=False)
    ck_gray = bool(ck.get("gray", False))
    if ck_gray != gray:
        print(f"  ! 경고: 체크포인트 gray={ck_gray} 인데 --gray={gray} 로 채점합니다")
    net = build_model(ck.get("arch", "effb0"))
    net.load_state_dict(ck["model"])
    net.eval()

    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
    out = {}
    with torch.no_grad():
        for i in range(0, len(frames), 32):
            batch = []
            for f in frames[i:i + 32]:
                im = Image.open(f).convert("RGB").resize((256, 256), Image.BILINEAR)
                im = im.resize((224, 224), Image.BILINEAR)
                if gray:
                    im = im.convert("L").convert("RGB")
                a = np.asarray(im, np.float32) / 255.0
                batch.append((a.transpose(2, 0, 1) - mean) / std)
            p = torch.softmax(net(torch.from_numpy(np.stack(batch))), 1)[:, 1]
            for f, v in zip(frames[i:i + 32], p.tolist()):
                out[str(f)] = v
    return out


def auc(scores, labels) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    s = sum(ranks[i] for i in range(len(labels)) if labels[i])
    return (s - pos * (pos + 1) / 2) / (pos * neg)


def report(tag: str, per_pipe: list, top_ratio: float) -> None:
    allp = [p for _, _, ps, _ in per_pipe for p in ps]
    alll = [d for _, _, _, ds in per_pipe for d in ds]

    def runs(secs, ds):
        out, cur = [], []
        for t, d in zip(secs, ds):
            if d:
                if cur and t - cur[-1] <= FRAME_STEP:
                    cur.append(t)
                else:
                    if cur:
                        out.append(cur)
                    cur = [t]
        if cur:
            out.append(cur)
        return out

    lost = tot = 0
    rng = random.Random(0)
    rand_trials = []
    for _ in range(500):
        rand_trials.append(0)
    for pipe, secs, ps, ds in per_pipe:
        k = max(1, round(len(secs) * top_ratio))
        top = {secs[i] for i in sorted(range(len(ps)), key=lambda i: -ps[i])[:k]}
        g = runs(secs, ds)
        tot += len(g)
        lost += sum(1 for r in g if not (set(r) & top))
        for n in range(500):
            rt = set(rng.sample(secs, k))
            rand_trials[n] += sum(1 for r in g if not (set(r) & rt))
    rand = sum(rand_trials) / len(rand_trials)

    a = auc(allp, alll)
    print(f"\n{'='*58}\n{tag}\n{'='*58}")
    print(f"  프레임 AUC            {a:.4f}   (0.5 = 무작위)")
    print(f"  상위 {top_ratio:.0%} 결함 유실    {lost}/{tot}건   ← 무작위로 골라도 {rand:.1f}건")
    d = sorted(p for p, l in zip(allp, alll) if l)
    n = sorted(p for p, l in zip(allp, alll) if not l)
    print(f"  결함 확률 중앙값       {d[len(d)//2]:.4f}")
    print(f"  그 외 확률 중앙값      {n[len(n)//2]:.4f}   ← 갈라져야 한다")
    verdict = ("무작위와 구별 안 됨" if a < 0.60 else
               "약하지만 신호 있음" if a < 0.75 else "쓸 만함")
    print(f"  판정: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, help=".onnx 또는 .pt")
    ap.add_argument("--gray", action="store_true", help="흑백으로 채점")
    ap.add_argument("--prefix", default="SM2")
    ap.add_argument("--top-ratio", type=float, default=0.20)
    args = ap.parse_args()

    truth = load_truth(args.prefix)
    cached = {}
    for pipe, items in sorted(truth.items()):
        items.sort()
        secs = [t for t, _ in items]
        frames = extract(pipe, secs)
        if not frames:
            print(f"  ! {pipe}: 영상 없음 — 건너뜀")
            continue
        got = {int(f.stem) for f in frames}
        cached[pipe] = (frames, [t for t in secs if t in got],
                        [d for t, d in items if t in got])
    print(f"프레임 준비 완료: {sum(len(v[0]) for v in cached.values()):,}장\n")

    for m in args.model:
        path = Path(m)
        if not path.exists():
            print(f"! 없는 파일: {path}")
            continue
        per_pipe = []
        for pipe, (frames, secs, ds) in cached.items():
            probs = (score_onnx(path, frames, args.gray) if path.suffix == ".onnx"
                     else score_torch(path, frames, args.gray))
            per_pipe.append((pipe, secs, [probs[str(f)] for f in frames], ds))
        report(f"{path.parent.name}/{path.name}"
               + ("  [흑백]" if args.gray else "  [컬러]"), per_pipe, args.top_ratio)


if __name__ == "__main__":
    main()
