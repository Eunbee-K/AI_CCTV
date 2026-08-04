"""추론 없이 보고서 출력만 확인하고 싶을 때 쓰는 샘플 데이터 생성기.

Colab이 죽었거나 로컬에 모델(assets/best.pt)이 없어도, 결과표·엑셀·파이프에셋 야장
PDF를 그대로 눌러볼 수 있게 세션 파일과 프레임 이미지를 만들어 넣는다.

    # 가진 영상에서 실제 프레임을 뽑아 채우기 (권장 — 보고서가 진짜처럼 보인다)
    python tools/seed_demo.py --video "D:/영상/1052. SD-A04-006.mp4"

    # 영상이 없으면 대체 이미지로 채우기
    python tools/seed_demo.py

    # 이미 작업 중인 세션이 있으면 덮어쓰지 않는다. 지우고 채우려면
    python tools/seed_demo.py --force

만든 뒤 서버를 (다시) 실행하면 결과가 올라와 있다.
"""
import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from backend.config import DATA_DIR, SESSION_FILE

# (코드, 등급) — 실제 조사에서 흔한 분포를 대충 따라간다
SAMPLE_DEFECTS = [
    ("PJ", "중"), ("DS", "소"), ("BK", "대"), ("PJ", "중"), ("CL", "소"),
    ("LP", "소"), ("DS", "중"), ("JF", "중"), ("PJ", "중"), ("CM", "중"),
    ("SD", "소"), ("DS", "대"), ("JD", "중"), ("PJ", "중"), ("IF", "중"),
    ("LS", "소"), ("DS", "중"), ("BK", "중"),
]


def _write_jpg(img, path: Path) -> None:
    """cv2.imwrite는 경로에 한글이 있으면 조용히 실패한다(파일이 안 생김).
    인코딩과 파일 쓰기를 나눠서 처리한다 — analysis.py도 같은 방식을 쓴다."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise SystemExit(f"이미지 인코딩 실패: {path}")
    buf.tofile(str(path))


def placeholder_frame(idx: int, dist_m: float, path: Path) -> None:
    """영상이 없을 때 쓰는 대체 이미지. 관 내부처럼 보이게만 만든다."""
    img = np.full((480, 640, 3), 45, np.uint8)
    for r, c in ((210 + (idx % 3) * 15, 105), (150, 88), (95, 62)):
        cv2.circle(img, (320, 240), r, (c, c + 8, c + 14), -1)
    cv2.circle(img, (320, 240), 45, (18, 18, 22), -1)
    cv2.putText(img, f"{dist_m:06.2f}m", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 235, 235), 1)
    cv2.putText(img, "SAMPLE - not real footage", (12, 462),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (170, 170, 170), 1)
    _write_jpg(img, path)


def frames_from_video(video: Path, out_dir: Path, count: int):
    """영상에서 고르게 count장을 뽑는다. 반환: [(time_s, 저장경로)]"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = int(total / fps) if fps > 0 else 0
    if duration < 2:
        cap.release()
        raise SystemExit(f"영상이 너무 짧습니다({duration}초): {video}")

    step = max(1, duration // count)
    made = []
    for i in range(count):
        t = min(duration - 1, i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        fp = out_dir / f"{t:06d}.jpg"
        _write_jpg(frame, fp)
        made.append((t, fp))
    cap.release()
    return made


def main():
    ap = argparse.ArgumentParser(description="보고서 확인용 샘플 데이터를 만든다")
    ap.add_argument("--video", help="프레임을 뽑아올 영상 파일(없으면 대체 이미지 사용)")
    ap.add_argument("--count", type=int, default=len(SAMPLE_DEFECTS), help="만들 결함 행 수")
    ap.add_argument("--force", action="store_true", help="기존 세션을 지우고 새로 만든다")
    args = ap.parse_args()

    if SESSION_FILE.exists() and not args.force:
        raw = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        n = sum(len(v.get("rows", [])) for v in raw.get("videos", []))
        if n:
            print(f"이미 작업 중인 세션이 있습니다 (결함 {n}행).")
            print("덮어쓰려면 --force 를 붙이세요. 기존 결과는 사라집니다.")
            return 1

    count = max(1, min(args.count, len(SAMPLE_DEFECTS)))
    if args.video:
        video = Path(args.video)
        name = video.name
    else:
        video = DATA_DIR / "uploads" / "샘플영상.mp4"
        name = video.name

    stem = Path(name).stem
    frames_dir = DATA_DIR / "frames" / stem
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        print(f"영상에서 프레임 추출: {video}")
        picked = frames_from_video(video, frames_dir, count)
    else:
        print("영상이 지정되지 않아 대체 이미지를 만듭니다 (--video 로 실제 영상 지정 가능)")
        picked = []
        for i in range(count):
            t = i * 6 + 2
            fp = frames_dir / f"{t:06d}.jpg"
            placeholder_frame(i, i * 4.7, fp)
            picked.append((t, fp))

        # 재생·미리보기도 눌러볼 수 있게 짧은 영상까지 만들어 둔다
        video.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480))
        for i in range(count):
            frame = cv2.imdecode(np.fromfile(str(picked[i][1]), dtype=np.uint8), cv2.IMREAD_COLOR)
            for _ in range(60):     # 프레임당 6초
                vw.write(frame)
        vw.release()
        if not video.exists():
            print("  ! 영상 파일 생성에 실패했습니다(코덱). 결과표·보고서는 그대로 확인됩니다.")

    rows = []
    for i, (t, fp) in enumerate(picked):
        code, grade = SAMPLE_DEFECTS[i % len(SAMPLE_DEFECTS)]
        rows.append({
            "time": t, "dist": f"{i * 4.7:.1f}", "defects": [code], "grade": grade,
            "note": "", "direction": "", "fp": False,
            "boxes": [], "boxes_norm": [],
            "frame_path": str(fp), "frame_annot_path": None,
        })

    session = {
        "version": 2,
        "site_name": "샘플 현장 (추론 없이 만든 확인용 데이터)",
        "pipe_condition": "노후",
        # 현장 전체 공통 항목
        "project_meta": {
            "사업기간": f"{date.today().year}-01-01~{date.today().year}-12-31",
            "조사목적": "기존관상태",
            "조사자": "(조사자명 입력)",
        },
        "remote_yolo_url": "",
        "videos": [{
            "name": name, "path": str(video),
            "pipe_id": stem, "dia": "Φ 450", "rows": rows,
            # 관로별 항목 — 맨홀번호·관종·거리는 관로마다 다르다
            "meta": {
                "보고서번호": "SAMPLE-001",
                "조사위치": "경기 가평군 북면",
                "조사일자": date.today().isoformat(),
                "구분": "관로", "관종": "흄관(HP)", "배수방식": "우수", "주행방향": "정주행",
                "연장": "84.70m", "총주행거리": "84.70m", "미주행거리": "0.0m",
                "상류맨홀번호": "4809", "상류맨홀종류": "NM", "상류맨홀재질": "RC", "상류맨홀깊이": "1.00",
                "하류맨홀번호": "5236", "하류맨홀종류": "NM", "하류맨홀재질": "RC", "하류맨홀깊이": "1.33",
                "맨홀구조적상태등급": "1.00", "맨홀운영적상태등급": "1.00",
                "관로구조적상태등급": "2.03", "관로운영적상태등급": "1.21",
                "미주행사유": "완주",
            },
        }],
    }

    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n완료: 결함 {len(rows)}행 / 프레임 {len(picked)}장")
    print(f"  세션 : {SESSION_FILE}")
    print(f"  프레임: {frames_dir}")
    if not args.video:
        print("  ※ 대체 이미지입니다. 실제 사진으로 보려면 --video 로 영상을 지정하세요.")
    print("\n이제 서버를 (다시) 실행하면 결과표에 올라와 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
