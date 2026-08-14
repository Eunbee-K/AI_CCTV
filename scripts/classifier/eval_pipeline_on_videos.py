"""필터+YOLO 직렬 파이프라인을 검증용 영상으로 채점한다.

정답은 `인식률.xlsx` Sheet1 — 관로별 결함 위치(m)와 종류. 조사원이 적은 거리이고
우리는 프레임을 초 단위로 뽑아 자막 거리를 OCR로 읽으므로 정확히 같은 지점이 나올 수
없다. **거리 ±TOL_M, 시간 ±TOL_S 안에 들면 맞힌 것으로 본다.**

두 단계로 나눈 이유
    직렬(lead)에서 **행을 만드는 것은 필터**고 YOLO는 이름만 붙인다. 그러니 재현율은
    YOLO를 바꿔도 변하지 않는다. 필터·OCR 결과를 한 번만 계산해 캐시하고(1단계),
    YOLO 모델만 바꿔가며 이름을 붙여 비교한다(2단계).

    캐시가 없으면 1단계가 오래 걸린다(CPU 기준 30분 남짓). 두 번째 모델부터는 빠르다.

사용:
    python eval_pipeline_on_videos.py --yolo A/best.pt B/best.pt
    python eval_pipeline_on_videos.py --yolo A/best.pt --limit 5 --rebuild
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "AI_CCTV"))
from paths import DATASET, FILTER_DATA  # noqa: E402

VIDEO_DIR = Path(r"C:/Users/SAMSUNG/Desktop/CCTV/검증용 영상data")
XLSX = Path(r"C:/Users/SAMSUNG/Desktop/CCTV/인식률.xlsx")
CACHE = FILTER_DATA / "pipeline_eval_cache"
TOL_M = 0.3
TOL_S = 3.0

# 엑셀 정답의 결함명 -> 우리 코드. 엑셀은 거친 분류라 하나가 여러 코드에 걸린다.
TRUTH_TO_CODES = {
    "이탈": {"JS", "JD"},
    "손상": {"JF", "SD", "BK", "LS", "CC", "CL", "CM"},
    "침하": {"SG", "DS"},
    "파손": {"BK", "HL"},
    "영구장애물": {"PO", "TO", "OBST"},
    "관처짐": {"SG", "DF"},
}


def norm_pipe(name: str) -> str:
    """`CP-C-17`과 `CP-C-017`을 같게 본다 — 엑셀과 파일명 표기가 다르다."""
    return "-".join(p.zfill(3) if p.isdigit() else p.upper()
                    for p in re.split(r"[-_]", name.strip()))


def load_truth() -> dict:
    import openpyxl
    ws = openpyxl.load_workbook(XLSX, data_only=True)["Sheet1"]
    out, cur = defaultdict(list), None
    for r in ws.iter_rows(min_row=4, values_only=True):
        if r[2]:
            cur = str(r[2]).strip()
        if not cur or r[3] is None:
            continue
        raw, kind = str(r[3]).strip(), str(r[4] or "").strip()
        m = re.match(r"^([\d.]+)\s*~\s*([\d.]+)$", raw)     # "7.0~10" 구간 표기
        try:
            lo, hi = (float(m.group(1)), float(m.group(2))) if m else (float(raw), float(raw))
        except ValueError:
            continue
        out[norm_pipe(cur)].append({"lo": lo, "hi": hi, "kind": kind, "raw": raw})
    return out


def find_videos() -> dict:
    out = {}
    for v in sorted(VIDEO_DIR.glob("*.mp4")):
        m = re.match(r"^\d+\.\s*(.+)$", v.stem)
        out[norm_pipe(m.group(1) if m else v.stem)] = v
    return out


def build_cache(pipes, videos, rebuild: bool, curve_step: float = 8.0):
    """1단계 — 프레임별 필터 점수 + 시간↔거리 곡선. YOLO·상위비율과 무관하다.

    **모든 프레임의 점수를 저장한다.** 상위 몇 %로 자를지는 나중에 정하는 값인데,
    구간만 저장하면 비율을 바꿀 때마다 30분씩 다시 돌려야 한다.

    거리는 성기게(curve_step초마다) 읽어 곡선을 만든다. 전 프레임 OCR은 8,600장에
    두 시간이 넘는다. 거리는 단조증가하므로 보간으로 충분하다.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    import cv2
    from backend import analysis
    from backend.config import FRAME_INTERVAL
    from backend.frames import extract_frames
    from backend.ocr import ocr_distance_from_frame
    from backend.state import state

    for i, p in enumerate(pipes, 1):
        out = CACHE / f"{p}.json"
        if out.exists() and "probs" in out.read_text(encoding="utf-8")[:200] and not rebuild:
            continue
        v = videos[p]
        root = Path(tempfile.mkdtemp())
        state.frames_root = root
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps)
        cap.release()

        frames = extract_frames(v, root, 0, dur, FRAME_INTERVAL)
        probs = analysis._run_filter(frames, v.name)
        outside = analysis._outside_frames(frames)

        # 시간 -> 거리 곡선 (성기게)
        pts = []
        for f in frames:
            t = analysis._frame_sec(f)
            if t % int(curve_step):
                continue
            m = re.search(r"(\d+(?:\.\d+)?)", ocr_distance_from_frame(Path(f)) or "")
            if m:
                pts.append([t, float(m.group(1))])
        mono = []
        for t, d in sorted(pts):
            if not mono or d >= mono[-1][1]:
                mono.append([t, d])

        rec = {"pipe": p, "video": v.name, "frames": len(frames),
               "probs": {str(analysis._frame_sec(f)): probs.get(str(f)) for f in frames},
               "outside": sorted(analysis._frame_sec(f) for f in outside),
               "curve": mono, "frames_dir": str(root / v.stem)}
        out.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        print(f"  [{i}/{len(pipes)}] {p:<14} 프레임 {len(frames):>4} "
              f"· 관밖 {len(outside):>3} · 거리표본 {len(mono):>3}")


def dist_at(c, t: float):
    """곡선에서 그 시각의 거리를 보간한다."""
    pts = c.get("curve") or []
    if len(pts) < 2:
        return None
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (t0, d0), (t1, d1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return d0 + (d1 - d0) * (t - t0) / max(t1 - t0, 1e-9)
    return None


def runs_at(c, top_ratio: float, gap: int = 1):
    """상위 top_ratio를 골라 이어진 것끼리 묶는다. 관 밖은 뺀다."""
    outside = set(c.get("outside") or [])
    items = [(int(t), v) for t, v in c["probs"].items()
             if v is not None and int(t) not in outside]
    if not items:
        return []
    keep_n = max(1, round(len(items) * top_ratio))
    top = sorted(sorted(items, key=lambda kv: -kv[1])[:keep_n])
    runs, cur = [], []
    for t, _ in top:
        if cur and t - cur[-1] <= gap:
            cur.append(t)
        else:
            if cur:
                runs.append(cur)
            cur = [t]
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        rep = max(r, key=lambda t: c["probs"][str(t)])
        out.append({"secs": r, "rep": rep, "dist": dist_at(c, rep),
                    "prob": c["probs"][str(rep)]})
    return out


def speed_mps(c) -> float:
    pts = c.get("curve") or []
    if len(pts) < 2:
        return 0.0
    (t0, d0), (t1, d1) = pts[0], pts[-1]
    return (d1 - d0) / (t1 - t0) if t1 > t0 else 0.0


def effective_tol(c, tol_m: float, tol_s: float) -> float:
    """**시간 허용오차를 거리로 바꿔 넉넉한 쪽을 쓴다.**

    프레임을 1초 간격으로 끊어 보므로 조사원이 적은 그 지점의 프레임을 정확히
    집지 못하고 앞뒤 프레임에 걸릴 수 있다. 거리 ±tol_m만 쓰면 카메라가 빠른
    관로에서 부당하게 불리하다 — 0.5m/s면 1초만 어긋나도 0.5m가 벌어진다.
    """
    return max(tol_m, speed_mps(c) * tol_s)


def match(runs, c, d, tol_m: float, tol_s: float):
    tol = effective_tol(c, tol_m, tol_s)
    for i, r in enumerate(runs):
        if r["dist"] is not None and d["lo"] - tol <= r["dist"] <= d["hi"] + tol:
            return True, i
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tol-m", type=float, default=TOL_M)
    ap.add_argument("--tol-s", type=float, default=TOL_S)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--top-ratio", type=float, default=0.20,
                    help="필터 상위 몇 %를 결함 구간으로 볼지")
    args = ap.parse_args()

    truth, videos = load_truth(), find_videos()
    pipes = sorted(set(truth) & set(videos))
    if args.limit:
        pipes = pipes[:args.limit]
    total_def = sum(len(truth[p]) for p in pipes)
    print(f"영상 {len(pipes)}개 · 조사표 결함 {total_def}건 · 허용 거리 ±{args.tol_m}m\n")

    print("1단계 — 필터가 고르는 구간 (YOLO와 무관, 한 번만)")
    build_cache(pipes, videos, args.rebuild)

    cached = {p: json.loads((CACHE / f"{p}.json").read_text(encoding="utf-8"))
              for p in pipes if (CACHE / f"{p}.json").exists()}

    # ── 재현율은 필터만으로 정해진다 ──
    hit = miss = rows = 0
    print(f"\n{'='*70}\n필터(직렬) 단독 — 행을 만드는 주체\n{'='*70}")
    print(f"  {'관로':<14}{'프레임':>6}{'행':>5}{'결함':>5}{'맞힘':>5}{'±m':>6}  놓친 것")
    for p in pipes:
        c = cached.get(p)
        if not c:
            continue
        rr = runs_at(c, args.top_ratio)
        rows += len(rr)
        got, lost = 0, []
        for d in truth[p]:
            ok, _ = match(rr, c, d, args.tol_m, args.tol_s)
            got += ok
            if not ok:
                lost.append(f"{d['raw']}m {d['kind']}")
        hit += got
        miss += len(truth[p]) - got
        print(f"  {p:<14}{c['frames']:>6}{len(rr):>5}{len(truth[p]):>5}{got:>5}"
              f"{effective_tol(c, args.tol_m, args.tol_s):>6.1f}  "
              + ", ".join(lost[:3]))
    print(f"\n  재현율 {hit}/{hit+miss} = {hit/max(hit+miss,1):.1%}")
    print(f"  행 {rows}개 (관로당 {rows/max(len(pipes),1):.1f})")

    # ── YOLO는 이름만 붙인다 ──
    for model in args.yolo:
        os.environ["YOLO_MODEL_PATH"] = str(model)
        for m in [k for k in sys.modules if k.startswith("backend")]:
            del sys.modules[m]
        from backend.config import defect_korean
        from backend.yolo_infer import call_yolo, init_yolo
        net, err = init_yolo()
        if not net:
            print(f"! {model}: {err}")
            continue

        named = unnamed = right = wrong = 0
        kinds = defaultdict(lambda: [0, 0])
        for p in pipes:
            c = cached.get(p)
            if not c:
                continue
            fdir = Path(c["frames_dir"])
            for r in runs_at(c, args.top_ratio):
                fs = [fdir / f"{s:06d}.jpg" for s in r["secs"]]
                fs = [f for f in fs if f.exists()]
                if not fs:
                    continue
                got, _ = call_yolo(net, fs)
                codes = {d for it in got for d in it.get("defects", [])}
                if not codes:
                    unnamed += 1
                    continue
                named += 1
                # 이 구간이 조사표 결함과 맞는가 → 이름이 맞는가
                tol = effective_tol(c, args.tol_m, args.tol_s)
                for d in truth[p]:
                    if r["dist"] is None:
                        continue
                    if d["lo"] - tol <= r["dist"] <= d["hi"] + tol:
                        want = set()
                        for k in re.split(r"[,\s]+", d["kind"]):
                            want |= TRUTH_TO_CODES.get(k.strip(), set())
                        ok = bool(codes & want)
                        kinds[d["kind"]][0] += ok
                        kinds[d["kind"]][1] += 1
                        right += ok
                        wrong += not ok
                        break

        print(f"\n{'='*70}\n{Path(model).parents[1].name} — 이름 붙이기\n{'='*70}")
        print(f"  이름 붙은 행 {named} · 확인필요 {unnamed} "
              f"({named/max(named+unnamed,1):.0%} 명명)")
        print(f"  조사표와 맞은 행 중 종류까지 맞은 것 {right}/{right+wrong} "
              f"= {right/max(right+wrong,1):.0%}")
        if kinds:
            print(f"\n  {'조사표 결함명':<14}{'맞힘':>6}{'전체':>6}")
            for k, (a, b) in sorted(kinds.items()):
                print(f"  {k:<14}{a:>6}{b:>6}")


if __name__ == "__main__":
    main()
