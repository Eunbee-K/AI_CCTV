import time
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

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
    return list_queue()


@router.get("")
def list_queue():
    return [{"name": p.name} for p in state.video_queue]


@router.post("/clear")
def clear_queue():
    state.clear_videos()
    return {"status": "ok"}


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


def _mjpeg_generator(path: Path, start_t: float, speed: float = 1.0):
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


@preview_router.get("/stream")
def preview_stream(name: str, start_t: float = 0, speed: float = 1.0):
    path = state.get_path_by_name(name)
    if not path:
        raise HTTPException(404, f"Unknown video: {name}")
    return StreamingResponse(
        _mjpeg_generator(path, start_t, max(0.25, min(4.0, speed))),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
