"""Stage-1 필터 재학습용 데이터셋 — 노후관로 현장 영상에서 직접 만든다.

**왜 다시 만드는가**

clsdata_v1로 학습한 필터는 현장에서 쓸 수 없었다. 노후관로 영상 2편에서 관 안에
들어간 뒤 97%를 결함으로 판정했고, 조사표로 확정된 결함 사진 59장의 점수 중앙값
0.9757과 영상에서 무작위로 뽑은 관 안 프레임 303장의 0.9751이 사실상 같았다.
임계값을 어디로 옮겨도 살릴 수 없다 — 순위 자체가 잡음이다.

원인은 학습 데이터의 정상(IN·PJ)이 전부 AIHub 출신이었다는 것이다. 모델이 배운
정상은 "관 밖"(맨홀·인버트)뿐이고, "현장 화질의 관 내부 정상"은 배우지 못했다.

**설계에서 지키는 것 — 정상과 결함을 같은 출처에서 뽑는다**

clsdata_v1의 두 번째 실패는 정상이 전부 AIHub, 결함 16종이 S20 전용이라 모델이
결함이 아니라 출처를 보고 갈라도 점수가 나왔던 것이다. 그래서 여기서는 정상도
결함도 **같은 현장 영상의 같은 프레임 추출 경로**에서 나온다. 양이 적어도 이게 낫다.

**어떻게 정상을 특정하는가**

한글 조사표에서 뽑은 결함 위치(`fieldset_v2_defects/defects.csv`)를 쓴다.
영상 자막의 주행거리를 성기게 읽어 시간↔거리 곡선을 만든 뒤,
  - 결함 위치 ±DEFECT_M  -> 결함 프레임
  - 어떤 결함과도 NORMAL_GAP_M 이상 떨어진 구간 -> 정상 프레임
사이 구간은 애매하므로 양쪽 어디에도 넣지 않는다.

**분할은 관로 단위로** 한다. 같은 관로의 프레임이 train과 val에 섞이면 검증이
무의미해진다 — clsdata_v1에서 이미 겪었다.

사용:
    python build_field_filter_dataset.py --videos "C:/.../상면 CCTV/2026..."
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fieldset"))
sys.path.insert(0, str(ROOT / "apps" / "AI_CCTV"))

from build_field_dataset import longest_nondecreasing, read_distance  # noqa: E402

DEFECT_M = 0.8      # 결함 위치에서 이 거리 안쪽은 결함으로 본다
NORMAL_GAP_M = 3.0  # 모든 결함에서 이만큼 떨어져야 정상으로 인정한다
STORE_PX = 256      # train_binary.py가 기대하는 저장 크기


def build_curve(cap, ocr_fn, tmp: Path, fps: float, dur: float, step_s: float):
    """(시각, 거리) 곡선. 거리는 단조증가하므로 어긋난 점은 OCR 오독으로 보고 버린다."""
    raw = []
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        d = read_distance(ocr_fn, fr, tmp)
        if d is not None:
            raw.append((t, d))
        t += step_s
    return longest_nondecreasing(raw)


def label_at(dist: float, defects: list[float]) -> str | None:
    """이 거리의 프레임을 무엇으로 볼 것인가. 애매하면 None."""
    if not defects:
        return None
    near = min(abs(dist - d) for d in defects)
    if near <= DEFECT_M:
        return "defect"
    if near >= NORMAL_GAP_M:
        return "normal"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="관로 폴더들이 있는 현장 폴더")
    ap.add_argument("--defects", default=r"E:/AI_CCTV_DATASET/fieldset_v2_defects/defects.csv")
    ap.add_argument("--photos", default=r"E:/AI_CCTV_DATASET/fieldset_v2_defects/photos")
    ap.add_argument("--out", default=r"E:/AI_CCTV_DATASET/clsdata_v2")
    ap.add_argument("--curve-step", type=float, default=4.0, help="곡선용 성긴 OCR 간격(초)")
    ap.add_argument("--frame-step", type=float, default=2.0, help="추출 간격(초). 앱과 같게 둔다")
    ap.add_argument("--val-pipes", type=int, default=5, help="검증 전용으로 뺄 관로 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from backend.ocr import ocr_distance_from_frame

    out = Path(args.out)
    for split in ("train", "val"):
        for b in ("normal", "defect"):
            (out / split / b).mkdir(parents=True, exist_ok=True)
    tmp = out / "_ocr.jpg"

    with open(args.defects, encoding="utf-8-sig") as f:
        drows = list(csv.DictReader(f))
    by_pipe: dict[str, list[float]] = defaultdict(list)
    photo_of: dict[str, list[str]] = defaultdict(list)
    for r in drows:
        by_pipe[r["pipe_dir"]].append(float(r["distance_m"]))
        photo_of[r["pipe_dir"]].append(r["photo"])

    pipes = sorted(by_pipe)
    rng = random.Random(args.seed)
    val_pipes = set(rng.sample(pipes, min(args.val_pipes, len(pipes))))
    print(f"관로 {len(pipes)}개 — 검증 전용 {len(val_pipes)}개: {sorted(val_pipes)}\n")

    manifest, counts = [], defaultdict(int)
    src = Path(args.videos)

    for pipe in pipes:
        d = src / pipe
        mp4 = next(d.glob("*.mp4"), None)
        if not mp4:
            print(f"{pipe[:34]:<36} 영상 없음 — 건너뜀")
            continue
        split = "val" if pipe in val_pipes else "train"
        defects = sorted(by_pipe[pipe])

        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        curve = build_curve(cap, ocr_distance_from_frame, tmp, fps, dur, args.curve_step)
        if len(curve) < 3:
            print(f"{pipe[:34]:<36} 거리 곡선 실패({len(curve)}점) — 건너뜀")
            cap.release()
            continue

        # 곡선을 시각->거리로 뒤집어 쓴다(추출은 시각 기준이라 이쪽이 자연스럽다)
        ts = np.array([t for t, _ in curve])
        ds = np.array([dd for _, dd in curve])

        n = {"normal": 0, "defect": 0}
        t = 0.0
        while t < dur:
            if t >= ts[0] and t <= ts[-1]:
                dist = float(np.interp(t, ts, ds))
                lab = label_at(dist, defects)
                if lab:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ok, fr = cap.read()
                    if ok:
                        name = f"{pipe[:28]}__{int(t):05d}s__{dist:06.2f}m.jpg"
                        dst = out / split / lab / name
                        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                        im.resize((STORE_PX, STORE_PX), Image.BILINEAR).save(
                            dst, "JPEG", quality=88, optimize=True)
                        manifest.append({"pipe": pipe, "split": split, "label": lab,
                                         "time_s": round(t, 1), "dist_m": round(dist, 2),
                                         "src": "video", "dst": str(dst.relative_to(out))})
                        n[lab] += 1
                        counts[(split, lab)] += 1
            t += args.frame_step
        cap.release()

        # 조사표 사진도 결함 학습 자료로 넣는다. 같은 관로이므로 같은 split에 둔다.
        for ph in photo_of[pipe]:
            p = Path(args.photos) / ph
            if not p.exists():
                continue
            dst = out / split / "defect" / f"hwp__{p.stem}.jpg"
            Image.open(p).convert("RGB").resize((STORE_PX, STORE_PX), Image.BILINEAR).save(
                dst, "JPEG", quality=88, optimize=True)
            manifest.append({"pipe": pipe, "split": split, "label": "defect",
                             "time_s": "", "dist_m": "", "src": "hwp",
                             "dst": str(dst.relative_to(out))})
            counts[(split, "defect")] += 1
            n["defect"] += 1

        print(f"{pipe[:34]:<36} [{split:<5}] 곡선 {len(curve):>3}점 "
              f"{ds[0]:.1f}~{ds[-1]:.1f}m  정상 {n['normal']:>3} 결함 {n['defect']:>3}")

    tmp.unlink(missing_ok=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    print("\n=== 합계 ===")
    for split in ("train", "val"):
        a, b = counts[(split, "normal")], counts[(split, "defect")]
        print(f"  {split:<6} 정상 {a:>5}  결함 {b:>5}  합계 {a+b:>5}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
