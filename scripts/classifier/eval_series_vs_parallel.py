"""직렬/병렬 결정을 위한 실측 — 조사표 결함 위치를 정답으로 삼는다.

두 질문에 답한다.

**직렬** — 임계값 아래 프레임을 버릴 때 실제 결함이 사라지는가.
   버려진 프레임 중에 정답 구간이 있으면 그 결함은 YOLO에 도달조차 못 한다.
   되돌릴 수 없으므로 이 숫자가 0에 가까워야 직렬을 쓸 수 있다.

**병렬** — "결함 의심"으로 올라오는 프레임이 얼마나 되고 그중 진짜는 몇이나 되는가.
   병렬은 아무것도 버리지 않는 대신 검수자에게 표시를 남긴다. 표시가 너무 많으면
   검토 부담만 늘어난다.

정답은 한글 조사표에서 뽑은 결함 거리다. 영상 자막 거리를 OCR로 읽어 시간↔거리
곡선을 만들고, 결함 거리 ±TOL_M 안의 프레임을 정답 구간으로 본다.

사용:
    python eval_series_vs_parallel.py --prefix SM2
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fieldset"))
sys.path.insert(0, str(ROOT / "apps" / "AI_CCTV"))

from build_field_dataset import longest_nondecreasing, read_distance  # noqa: E402
from train_binary import build_model  # noqa: E402

DS = Path(r"E:/AI_CCTV_DATASET")
SRC = Path(r"C:/Users/SAMSUNG/Desktop/상면 CCTV/20260128.가평군 상면 노후하수관로 정비공사")
TOL_M = 1.0


@torch.no_grad()
def score(model, tf, frames):
    out = []
    for i in range(0, len(frames), 32):
        b = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames[i:i + 32]])
        out += torch.softmax(model(b).float(), 1)[:, 1].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        Path(r"E:/AI_CCTV_RESULTS/filter/runs/OLD_v3-20260811T174845Z-1-001/OLD_v3/best.pt")))
    ap.add_argument("--prefix", default="SM2")
    ap.add_argument("--thresholds", default="0.032,0.043,0.071")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--curve-step", type=float, default=4.0)
    args = ap.parse_args()

    from backend.ocr import ocr_distance_from_frame

    with open(DS / "fieldset_v2_defects/defects.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["pipe_dir"].startswith(args.prefix)]
    by_pipe = defaultdict(list)
    for r in rows:
        by_pipe[r["pipe_dir"]].append(float(r["distance_m"]))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck.get("arch", "effb0"))
    model.load_state_dict(ck["model"])
    model.eval()
    img = ck.get("img", 224)
    tf = transforms.Compose([
        transforms.Resize(img), transforms.CenterCrop(img), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    tmp = Path(tempfile.mkdtemp()) / "_ocr.jpg"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",")]

    # 관로별로 (프레임 확률, 정답 여부)를 모아둔다
    all_probs, all_hit, per_pipe = [], [], []
    for pipe, dists in sorted(by_pipe.items()):
        mp4 = next((SRC / pipe).glob("*.mp4"), None)
        if not mp4:
            continue
        out = Path(tempfile.mkdtemp())
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps

        curve, t = [], 0.0
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if not ok:
                break
            d = read_distance(ocr_distance_from_frame, fr, tmp)
            if d is not None:
                curve.append((t, d))
            t += args.curve_step
        curve = longest_nondecreasing(curve)
        if len(curve) < 3:
            cap.release()
            continue
        ts = np.array([a for a, _ in curve]); ds = np.array([b for _, b in curve])

        frames, dist_of = [], {}
        t = 0.0
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, im = cap.read()
            if not ok:
                break
            fp = out / f"{int(t):06d}.jpg"
            cv2.imwrite(str(fp), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames.append(fp)
            dist_of[fp] = float(np.interp(t, ts, ds)) if ts[0] <= t <= ts[-1] else None
            t += args.step
        cap.release()

        probs = score(model, tf, frames)
        # 각 결함마다 정답 구간(±TOL_M) 프레임 집합
        groups = []
        for target in sorted(dists):
            g = {f for f in frames
                 if dist_of[f] is not None and abs(dist_of[f] - target) <= TOL_M}
            if g:
                groups.append((target, g))
        hit = [any(f in g for _, g in groups) for f in frames]
        all_probs += probs; all_hit += hit
        per_pipe.append((pipe, frames, probs, groups))

    # 재측정용 캐시 — OCR+추론이 비싸다
    cache = DS / f"filter_eval_{args.prefix}.csv"
    with open(cache, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["pipe", "frame", "prob", "is_defect"])
        for pipe, frames, probs, groups in per_pipe:
            gset = set().union(*[g for _, g in groups]) if groups else set()
            for fr, pr in zip(frames, probs):
                w.writerow([pipe, fr.name, f"{pr:.6f}", int(fr in gset)])
    print(f"캐시 저장: {cache}")

    n = len(all_probs)
    n_def_frames = sum(all_hit)
    print(f"\n{args.prefix} 관로 {len(per_pipe)}개 · 프레임 {n:,} · 결함 {sum(len(g) for _,_,_,g in per_pipe)}건")
    print(f"정답 구간 프레임 {n_def_frames}장 ({n_def_frames/n:.1%})\n")

    print("=== 직렬 — 버려서 사라지는 결함 ===")
    print(f"{'임계값':>8}{'통과':>10}{'제거':>10}{'사라진 결함':>14}")
    for thr in thresholds:
        keep = sum(1 for p in all_probs if p >= thr)
        lost = 0
        for _, frames, probs, groups in per_pipe:
            pmap = dict(zip(frames, probs))
            for _, g in groups:
                if all(pmap[f] < thr for f in g):     # 그 결함의 모든 프레임이 버려짐
                    lost += 1
        total = sum(len(g) for _, _, _, g in per_pipe)
        print(f"{thr:>8.3f}{keep:>10,}{n-keep:>10,}{f'{lost}/{total}건':>14}")

    print("\n=== 병렬 — '결함 의심' 표시 부담 ===")
    print(f"{'임계값':>8}{'표시 프레임':>12}{'그중 진짜':>11}{'정밀도':>9}{'검토 비율':>10}")
    for thr in thresholds:
        flag = [(p, h) for p, h in zip(all_probs, all_hit) if p >= thr]
        real = sum(1 for _, h in flag if h)
        print(f"{thr:>8.3f}{len(flag):>12,}{real:>11}{real/max(len(flag),1):>9.1%}{len(flag)/n:>10.1%}")

    percentile_report(per_pipe)


def percentile_report(per_pipe):
    """영상별 상위 N%로 자를 때 — 절대 임계값은 영상마다 분포가 달라 못 쓴다."""
    print("\n=== 영상별 상위 N% 로 자를 때 ===")
    print(f"{'상위':>6}{'남는 프레임':>12}{'사라진 결함':>14}{'제거율':>9}")
    total_def = sum(len(g) for _, _, _, g in per_pipe)
    total_fr = sum(len(f) for _, f, _, _ in per_pipe)
    for pct in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        keep_n = lost = 0
        for _, frames, probs, groups in per_pipe:
            k = max(1, int(len(frames) * pct))
            top = {f for f, _ in sorted(zip(frames, probs), key=lambda kv: -kv[1])[:k]}
            keep_n += len(top)
            for _, g in groups:
                if not (g & top):
                    lost += 1
        print(f"{pct:>5.0%}{keep_n:>12,}{f'{lost}/{total_def}건':>14}{1-keep_n/total_fr:>9.1%}")


if __name__ == "__main__":
    main()
