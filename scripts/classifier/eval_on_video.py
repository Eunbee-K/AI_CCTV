"""조사표 결함 위치를 정답으로 삼아, 연속 영상에서 필터의 실제 쓸모를 잰다.

val 점수(AUC·정밀도)는 실제 검토 부담을 대표하지 못한다. 야장 val은 조사원이 결함·정상
지점을 골라 찍은 프레임이라 결함 비율이 45%지만, 연속 영상은 1~3%다. 같은 AUC라도
정밀도가 완전히 달라진다.

그래서 여기서는 **"결함을 다 찾으려면 상위 몇 %를 봐야 하는가"**를 잰다. 검수자가
체감하는 것이 그것이기 때문이다.

정답은 한글 조사표에서 뽑은 결함 거리(fieldset_v2_defects/defects.csv)다. 영상의 자막
거리를 OCR로 읽어 시간↔거리 곡선을 만들고, 결함 거리에 해당하는 시각을 찾는다.

사용:
    python eval_on_video.py --prefix SM2
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
TOL_M = 1.0     # 결함 거리에서 이 안쪽 프레임을 정답으로 인정한다


def load_model(ckpt: Path):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_model(ck.get("arch", "effb0"))
    m.load_state_dict(ck["model"])
    m.eval()
    img = ck.get("img", 224)
    tf = transforms.Compose([
        transforms.Resize(img), transforms.CenterCrop(img), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return m, tf


@torch.no_grad()
def score(model, tf, frames):
    out = []
    for i in range(0, len(frames), 32):
        b = torch.stack([tf(Image.open(f).convert("RGB")) for f in frames[i:i + 32]])
        out += torch.softmax(model(b).float(), 1)[:, 1].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path(
        r"E:/AI_CCTV_RESULTS/filter/runs/OLD_v3-20260811T174845Z-1-001/OLD_v3/best.pt")))
    ap.add_argument("--prefix", default="SM2")
    ap.add_argument("--step", type=float, default=2.0, help="추출 간격(초). 앱과 같게")
    ap.add_argument("--curve-step", type=float, default=4.0)
    args = ap.parse_args()

    from backend.ocr import ocr_distance_from_frame

    with open(DS / "fieldset_v2_defects/defects.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["pipe_dir"].startswith(args.prefix)]
    by_pipe: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_pipe[r["pipe_dir"]].append(float(r["distance_m"]))

    model, tf = load_model(Path(args.ckpt))
    tmp = Path(tempfile.mkdtemp()) / "_ocr.jpg"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    tot_frames = tot_def = 0
    need_all = []
    print(f"{'관로':<34}{'프레임':>7}{'결함':>5}{'전부 찾는 검토량':>16}")
    for pipe, dists in sorted(by_pipe.items()):
        mp4 = next((SRC / pipe).glob("*.mp4"), None)
        if not mp4:
            continue
        out = Path(tempfile.mkdtemp())
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps

        # 시간 -> 거리 곡선 (성긴 OCR)
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
            print(f"{pipe[:32]:<34}  거리 곡선 실패")
            cap.release()
            continue
        ts = np.array([a for a, _ in curve])
        ds = np.array([b for _, b in curve])

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
        order = sorted(zip(frames, probs), key=lambda kv: -kv[1])
        rank = {f: i + 1 for i, (f, _) in enumerate(order)}

        worst = 0
        for target in sorted(dists):
            cand = [rank[f] for f in frames
                    if dist_of[f] is not None and abs(dist_of[f] - target) <= TOL_M]
            worst = max(worst, min(cand)) if cand else worst
        if worst:
            need_all.append((worst, len(frames)))
            print(f"{pipe[:32]:<34}{len(frames):>7}{len(dists):>5}"
                  f"{f'상위 {worst}개 ({worst/len(frames):.0%})':>16}")
        tot_frames += len(frames)
        tot_def += len(dists)

    if need_all:
        avg = sum(w / n for w, n in need_all) / len(need_all)
        print(f"\n관로 {len(need_all)}개 · 프레임 {tot_frames:,} · 결함 {tot_def}건")
        print(f"결함을 전부 찾으려면 평균 상위 {avg:.0%} 검토")


if __name__ == "__main__":
    main()
