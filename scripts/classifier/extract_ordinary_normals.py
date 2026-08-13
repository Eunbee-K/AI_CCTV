"""연속 영상에서 **평범한 구간**을 정상으로 뽑는다 — 지금 학습 데이터에 없는 종류다.

왜 필요한가
    필터가 야장 val에서 AUC 0.95인데 실영상에서 0.52다. 원인은 "정상"으로 배운 것이
    전부 조사원이 뭔가 적을 만해서 적은 순간이라는 것이다. 현장 정상 6,744장의 89%가
    `이음부(접합부)존재`고, 아무것도 없는 밋밋한 관 내부는 한 장도 없다. 실제 영상은
    대부분이 그 구간이라 모델 입장에서는 처음 보는 그림이다.

    야장은 결함(`관로`)과 정상(`상태`)을 **둘 다** 시각과 함께 적어둔다. 그러니
    "야장이 아무것도 안 적은 시각" = 조사원이 기록할 것이 없다고 본 구간이다.
    그게 우리에게 없는 바로 그 종류다.

주의 — 두 가지 함정을 피한다
    1. **관 밖을 정상으로 배우면 안 된다.** 첫 필터(AIHub 78,388장)가 정확히 그렇게
       망가져 현장 AUC 0.516이었다. 영상 앞부분은 맨홀·관 밖이므로 자막 거리가
       읽히고 MIN_DIST_M 이상인 프레임만 쓴다.
    2. **결함 옆을 정상이라 부르면 안 된다.** 신설(fieldset)에서 "하자에서 3m 밖"을
       정상으로 삼았다가 토사퇴적·임시장애물이 섞였다. 여기서는 야장 기록 전체를
       기준으로 ±GAP초를 잘라낸다.

    야장 기록이 평균 10.3초에 1건이라 ±3초를 빼면 32%만 남는다. 남는 양이 적은 것은
    문제가 아니다 — 없던 종류를 채우는 것이 목적이다.

**뽑은 다음 반드시 눈으로 볼 것.** --review 로 검수용 대지를 만든다.

사용:
    python extract_ordinary_normals.py --area 상면 --gap 3 --review
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "AI_CCTV"))
from paths import DATASET, FIELD_VIDEO  # noqa: E402

YAJANG = DATASET / "yajang_frames" / "yajang.csv"
MIN_DIST_M = 0.5        # 이보다 앞은 맨홀·관 밖으로 본다


def hhmmss(s: str) -> int:
    p = [int(x) for x in str(s).split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]


def find_videos(area: str) -> dict:
    """{관로번호: 영상경로}. 파일명이 4자리 관로번호로 시작한다."""
    out = {}
    for v in sorted(FIELD_VIDEO.rglob("*.mp4")):
        m = re.match(r"^(\d{4})", v.stem.strip())
        if m:
            out.setdefault(m.group(1), v)
        _ = area
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="상면")
    ap.add_argument("--gap", type=float, default=3.0, help="야장 기록에서 이만큼 떨어진 것만")
    ap.add_argument("--step", type=float, default=1.0, help="프레임 간격(초)")
    ap.add_argument("--out", default="")
    ap.add_argument("--review", action="store_true", help="검수용 대지 이미지도 만든다")
    ap.add_argument("--limit-per-pipe", type=int, default=120)
    args = ap.parse_args()

    out = Path(args.out) if args.out else DATASET / f"ordinary_normals_{args.area}"
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with open(YAJANG, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["area"] == args.area]
    marked = defaultdict(list)
    for r in rows:
        marked[r["pipe"]].append(hhmmss(r["time"]))

    videos = find_videos(args.area)
    pipes = sorted(set(marked) & set(videos))
    print(f"{args.area}: 야장 관로 {len(marked)}개 · 영상 있는 관로 {len(pipes)}개 "
          f"· 제외 폭 ±{args.gap}초\n")

    from backend.ocr import ocr_distance_from_frame

    manifest, stat = [], defaultdict(int)
    print(f"{'관로':<8}{'길이':>7}{'후보':>7}{'관밖제외':>9}{'채택':>7}")
    for pipe in pipes:
        v = videos[pipe]
        times = sorted(marked[pipe])
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps

        # 야장이 아무것도 안 적은 시각
        cand, t = [], 0.0
        while t < dur:
            if all(abs(t - m) > args.gap for m in times):
                cand.append(t)
            t += args.step
        # 고르게 솎아낸다 — 한쪽에 몰리면 같은 장면만 잔뜩 나온다
        if len(cand) > args.limit_per_pipe:
            k = len(cand) / args.limit_per_pipe
            cand = [cand[int(i * k)] for i in range(args.limit_per_pipe)]

        kept = outside = 0
        for t in cand:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, im = cap.read()
            if not ok:
                continue
            fp = frames_dir / f"{args.area}__{pipe}__t{int(t):05d}.jpg"
            cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 88])[1].tofile(str(fp))

            # 관 밖(맨홀)을 정상으로 배우면 안 된다 — v1이 그렇게 망가졌다
            dist = ocr_distance_from_frame(fp)
            m = re.search(r"(\d+(?:\.\d+)?)", dist or "")
            d = float(m.group(1)) if m else None
            if d is None or d < MIN_DIST_M:
                fp.unlink(missing_ok=True)
                outside += 1
                continue

            kept += 1
            manifest.append({"pipe": pipe, "time_s": f"{t:.1f}", "dist_m": f"{d:.1f}",
                             "file": fp.name, "video": v.name})
        cap.release()
        stat["kept"] += kept
        stat["outside"] += outside
        print(f"{pipe:<8}{int(dur)//60}:{int(dur)%60:02d}{len(cand):>9}{outside:>9}{kept:>7}")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["pipe", "time_s", "dist_m", "file", "video"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\n채택 {stat['kept']:,}장 · 관 밖으로 제외 {stat['outside']:,}장")
    print(f"저장: {out}")

    if args.review:
        make_review(out, manifest)


def make_review(out: Path, manifest: list, cols: int = 6, rows: int = 5) -> None:
    """검수용 대지. 한 장에 30개씩 붙여 눈으로 훑을 수 있게 한다.

    **이 단계를 건너뛰지 말 것.** 신설에서 "결함에서 3m 밖"을 정상으로 삼았다가
    토사퇴적이 섞여 있는 것을 한참 뒤에 발견했다.
    """
    import numpy as np
    rev = out / "review"
    rev.mkdir(exist_ok=True)
    per = cols * rows
    cell = 320
    for page in range((len(manifest) + per - 1) // per):
        sheet = np.full((rows * (cell + 22), cols * cell, 3), 255, np.uint8)
        for i, item in enumerate(manifest[page * per:(page + 1) * per]):
            r, c = divmod(i, cols)
            img = cv2.imdecode(np.fromfile(str(out / "frames" / item["file"]), np.uint8), 1)
            if img is None:
                continue
            img = cv2.resize(img, (cell, cell))
            y = r * (cell + 22)
            sheet[y:y + cell, c * cell:(c + 1) * cell] = img
            cv2.putText(sheet, f"{item['pipe']} {item['dist_m']}m t{item['time_s']}s",
                        (c * cell + 4, y + cell + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tofile(
            str(rev / f"page{page + 1:03d}.jpg"))
    print(f"검수용 대지 {len(list(rev.glob('*.jpg')))}장: {rev}")
    print("  → 결함이 섞여 있으면 그 프레임을 지우고 manifest에서도 빼세요.")


if __name__ == "__main__":
    main()
