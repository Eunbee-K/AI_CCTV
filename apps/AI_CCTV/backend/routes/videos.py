import re
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .. import session_store, ws_manager
from ..state import state

router = APIRouter(prefix="/api/queue", tags=["queue"])
preview_router = APIRouter(prefix="/api/preview", tags=["preview"])


class AddVideosBody(BaseModel):
    paths: list[str]


class SelectVideoBody(BaseModel):
    name: str


@router.post("/add")
def add_videos(body: AddVideosBody):
    for p in body.paths:
        state.add_video(Path(p))
    session_store.save()
    return list_queue()


def _safe_name(raw: str) -> str:
    """업로드 파일명을 파일시스템에 안전한 이름으로 바꾼다.

    브라우저가 보낸 이름을 그대로 쓰면 경로 조작(`../`)이나 윈도우 금지문자로
    엉뚱한 곳에 쓰일 수 있으므로 basename만 남기고 위험한 문자를 치환한다.
    """
    name = Path(raw or "").name                      # 디렉터리 성분 제거
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "video.mp4"


def _unique_path(directory: Path, name: str) -> Path:
    """같은 이름이 이미 큐에 있으면 `이름 (2).mp4` 식으로 비켜서 저장한다."""
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = directory / name
    n = 2
    while candidate.exists() or state.get_path_by_name(candidate.name):
        candidate = directory / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


@router.post("/upload")
async def upload_videos(files: list[UploadFile] = File(...)):
    """브라우저에서 고른/끌어놓은 영상 파일을 서버에 저장하고 큐에 넣는다.

    웹으로 쓸 때는 클라이언트 PC의 경로를 서버가 알 수 없으므로, 경로를 받는
    `/add` 대신 파일 자체를 받아야 한다. (데스크톱 exe는 계속 `/add` 사용)
    """
    saved, skipped = [], []

    for uf in files:
        name = _safe_name(uf.filename)
        dest = _unique_path(state.uploads_root, name)
        size = 0
        try:
            # 큰 영상도 메모리에 통째로 올리지 않도록 스트리밍 복사한다.
            with dest.open("wb") as out:
                shutil.copyfileobj(uf.file, out, length=1024 * 1024)
            size = dest.stat().st_size
        except Exception as e:
            dest.unlink(missing_ok=True)
            msg = f"{name} — 저장 실패: {e}"
            ws_manager.log(f"[upload] {msg}", "ERROR")
            skipped.append(msg)
            continue
        finally:
            await uf.close()

        if size == 0:
            dest.unlink(missing_ok=True)
            msg = f"{name} — 빈 파일(0바이트)입니다."
            ws_manager.log(f"[upload] {msg}", "ERROR")
            skipped.append(msg)
            continue

        # 확장자로 거르지 않고, 실제로 열리는지로 판단한다. 현장 영상은 확장자가
        # 제각각이라 화이트리스트를 두면 멀쩡한 파일이 거부된다.
        cap = cv2.VideoCapture(str(dest))
        opened = cap.isOpened()
        frame_ok = False
        if opened:
            frame_ok, _ = cap.read()
        cap.release()

        if not (opened and frame_ok):
            dest.unlink(missing_ok=True)
            msg = (f"{name} — 영상을 열 수 없습니다 "
                   f"({size / 1024 / 1024:.1f}MB, 확장자 '{Path(name).suffix or '없음'}'). "
                   f"코덱이 지원되지 않거나 파일이 손상됐을 수 있습니다.")
            ws_manager.log(f"[upload] {msg}", "ERROR")
            skipped.append(msg)
            continue

        state.add_video(dest)
        saved.append(dest.name)
        ws_manager.log(f"[upload] 추가됨: {dest.name} ({size / 1024 / 1024:.1f}MB)")

    session_store.save()
    return {"queue": list_queue(), "saved": saved, "skipped": skipped}


@router.get("")
def list_queue():
    return [{"name": p.name} for p in state.video_queue]


@router.post("/clear")
def clear_queue(video: Optional[str] = None):
    """video를 주면 그 영상만 목록에서 뺀다. 없으면 전체(예전 동작)."""
    removed = state.clear_videos(video)
    session_store.save()
    return {"status": "ok", "removed": removed}


@router.post("/select")
def select_video(body: SelectVideoBody):
    path = state.get_path_by_name(body.name)
    if not path:
        raise HTTPException(404, f"Unknown video: {body.name}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise HTTPException(400, f"Cannot open video: {body.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_s = int(total / fps) if fps > 0 else 0
    cap.release()
    return {"name": body.name, "duration_s": duration_s, "fps": fps}


@preview_router.get("/frame")
def preview_frame(name: str, t: float = 0):
    path = state.get_path_by_name(name)
    if not path:
        raise HTTPException(404, f"Unknown video: {name}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HTTPException(400, f"Cannot open video: {name}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.0, t) * fps))
        ok, frame = cap.read()
        if not ok:
            raise HTTPException(404, "Frame not found")
        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok2:
            raise HTTPException(500, "Encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    finally:
        cap.release()


def _mjpeg_generator(path: Path, start_t: float, speed: float = 1.0,
                     progress: Optional[dict] = None):
    """영상을 MJPEG로 흘려보낸다.

    progress를 주면 **지금 보낸 프레임이 몇 초 지점인지** 매 프레임 적어 넣는다.
    클라이언트는 벽시계로 재생 위치를 추정하는데, 이 스트림은 프레임마다
    `1/fps`를 자고 그 위에 디코딩·인코딩·전송 시간이 더 붙어서 **항상 실시간보다
    느리다**. 그래서 정지하는 순간 벽시계 값으로 스냅샷을 받으면 화면이 앞으로
    훌쩍 건너뛴다 — 결함 지점으로 제멋대로 이동하는 것처럼 보이던 게 이거다.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0.0, start_t) * fps))
        delay = 1.0 / fps if fps > 0 else 1.0 / 30
        skip_acc = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok2:
                break
            if progress is not None:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                progress["t"] = (pos_ms / 1000.0) if pos_ms > 0 else start_t
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
            # 배속: 초당 전송 프레임 수는 그대로 두고, 그 사이 프레임을 건너뛴다
            # (1.5배속이면 프레임 2개당 1개 스킵 — 소수 누적 방식)
            skip_acc += max(0.0, speed - 1.0)
            while skip_acc >= 1.0:
                cap.grab()
                skip_acc -= 1.0
            time.sleep(delay)
    finally:
        cap.release()


# 재생 중인 스트림이 실제로 어디까지 보냈는지. 키는 `영상이름|스트림id`.
# 한 사람이 쓰는 로컬 앱이라 메모리 dict로 충분하다.
_stream_progress: dict = {}


@preview_router.get("/stream")
def preview_stream(name: str, start_t: float = 0, speed: float = 1.0,
                   sid: str = ""):
    path = state.get_path_by_name(name)
    if not path:
        raise HTTPException(404, f"Unknown video: {name}")

    key = f"{name}|{sid}"
    prog = {"t": start_t}
    # 스트림은 하나만 살아 있으면 되므로, 새로 열리면 옛 기록을 정리한다.
    for k in [k for k in _stream_progress if k.startswith(f"{name}|")]:
        _stream_progress.pop(k, None)
    _stream_progress[key] = prog

    return StreamingResponse(
        _mjpeg_generator(path, start_t, max(0.25, min(4.0, speed)), prog),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@preview_router.get("/stream_pos")
def stream_pos(name: str, sid: str = ""):
    """재생 중인 스트림이 **실제로** 보여준 마지막 프레임의 시각(초).

    정지할 때 이 값으로 멈춰야 화면이 튀지 않는다. 스트림이 없으면 t=null.
    """
    prog = _stream_progress.get(f"{name}|{sid}")
    return {"t": prog.get("t") if prog else None}
