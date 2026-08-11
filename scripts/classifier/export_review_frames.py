"""검토용 프레임을 원본 해상도로 다시 뽑는다.

학습 저장본은 256x256으로 줄여져 있다(모델 입력이 224x224라 그 크기에 맞춘 것).
그 크기로는 사람이 결함 유무를 판단할 수 없다. 그렇다고 학습본을 키우면 학습이
느려지고 v1과 조건이 달라진다. 그래서 **검토용만 따로** 원본 해상도로 만든다.

거리 곡선은 clsdata_v2를 만들 때 이미 계산해 manifest에 시각이 남아 있으므로
OCR을 다시 돌릴 필요가 없다. 영상에서 그 시각만 꺼내면 된다.

파일명에 관로·시각·거리가 들어가므로, 검토 후 지울 것을 학습 폴더에서 같은 이름으로
찾아 지우면 된다.

사용:
    python export_review_frames.py --videos "C:/.../상면 CCTV/2026..."
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--data", default=r"E:/AI_CCTV_DATASET/clsdata_v2")
    ap.add_argument("--out", default=r"E:/AI_CCTV_DATASET/clsdata_v2_review")
    ap.add_argument("--max-width", type=int, default=0,
                    help="0이면 원본 그대로. 화면에서 보기 편한 폭으로 줄이려면 지정")
    args = ap.parse_args()

    data, out, src = Path(args.data), Path(args.out), Path(args.videos)
    with open(data / "manifest.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("src") == "video"]

    # 영상을 한 번만 열도록 관로별로 묶는다
    by_pipe: dict[str, list[dict]] = {}
    for r in rows:
        by_pipe.setdefault(r["pipe"], []).append(r)

    n = Counter()
    for pipe, items in sorted(by_pipe.items()):
        mp4 = next((src / pipe).glob("*.mp4"), None)
        if not mp4:
            print(f"{pipe[:34]:<36} 영상 없음")
            continue
        cap = cv2.VideoCapture(str(mp4))
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for r in sorted(items, key=lambda x: float(x["time_s"])):
            d = out / r["split"] / r["label"]
            d.mkdir(parents=True, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_MSEC, float(r["time_s"]) * 1000)
            ok, fr = cap.read()
            if not ok:
                continue
            if args.max_width and fr.shape[1] > args.max_width:
                k = args.max_width / fr.shape[1]
                fr = cv2.resize(fr, (args.max_width, int(fr.shape[0] * k)))
            # 학습본과 같은 이름을 쓴다 — 검토 후 지울 파일을 그대로 찾을 수 있다.
            # cv2.imwrite는 Windows에서 한글 경로에 저장하지 못하고 False만 돌려주므로
            # (관로 폴더명에 한글이 있다) PIL로 쓴다.
            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).save(
                d / Path(r["dst"]).name, "JPEG", quality=92)
            n[(r["split"], r["label"])] += 1
        cap.release()
        print(f"{pipe[:34]:<36} {w0}x{h0}  {len(items):>4}장")

    print("\n=== 검토용 (원본 해상도) ===")
    for k in sorted(n):
        print(f"  {k[0]:<6} {k[1]:<7} {n[k]:>5}장")
    print(f"\n-> {out}")
    print("학습본과 파일명이 같으므로, 여기서 지울 것을 골라 clsdata_v2에서 같은 이름을 지우면 된다.")


if __name__ == "__main__":
    main()
