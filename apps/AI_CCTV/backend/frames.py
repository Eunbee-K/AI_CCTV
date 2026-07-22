from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import EXTRACT_JPEG_QUALITY


def seconds_to_mmss(sec: int) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def mmss_to_seconds(mmss: str) -> int:
    try:
        parts = mmss.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def _img_bytes_downscaled(
    fp: Path, quality: int, max_side: int = 0, fixed_width: int = 0
) -> Tuple[bytes, int, int]:
    try:
        img_array = np.fromfile(str(fp), dtype=np.uint8)
        im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("Decode fail")
        h, w = im.shape[:2]
        if fixed_width > 0:
            scale = fixed_width / w
            new_w, new_h = fixed_width, int(h * scale)
        elif max_side > 0:
            scale = min(1.0, max_side / max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = w, h
        if (new_w, new_h) != (w, h):
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), new_w, new_h
    except Exception:
        return b'', 0, 0


def extract_frames(
    video_path: Path, out_dir: Path, start_s: int, end_s: int, interval: int
) -> List[Path]:
    v_name = video_path.stem
    save_dir = out_dir / v_name
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved = []
    next_target = start_s
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / fps
        frame_idx += 1
        if ts < start_s:
            continue
        if ts > end_s:
            break
        if ts >= next_target:
            fp = save_dir / f"{int(ts):06d}.jpg"
            try:
                cv2.imencode(
                    ".jpg", frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), EXTRACT_JPEG_QUALITY]
                )[1].tofile(str(fp))
                saved.append(fp)
            except Exception:
                pass
            next_target += interval
    cap.release()
    return saved
