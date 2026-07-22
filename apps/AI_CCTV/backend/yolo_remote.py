from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .config import YOLO_CONF, YOLO_IGNORE_CLASSES, YOLO_IMGSZ
from .yolo_infer import _normalize_yolo_class_name, _yolo_display_label

REMOTE_TIMEOUT_S = 180
# 한 요청에 보내는 프레임 수. 전체(수백 장)를 한 번에 보내면 Colab 쪽에서
# 리스트 전체가 GPU 한 배치로 올라가 CUDA OOM(500 에러)이 난다.
REMOTE_CHUNK_SIZE = 24


def _post_chunk(url: str, chunk: List[Path]) -> dict:
    opened = [open(fp, "rb") for fp in chunk]
    try:
        files = [
            ("files", (fp.name, fobj, "image/jpeg"))
            for fp, fobj in zip(chunk, opened)
        ]
        resp = requests.post(
            url,
            files=files,
            data={"conf": YOLO_CONF, "imgsz": YOLO_IMGSZ},
            timeout=REMOTE_TIMEOUT_S,
        )
        if resp.status_code != 200:
            # 서버가 에러 내용을 JSON으로 보내주면 그걸 그대로 보여준다
            try:
                detail = resp.json().get("error", "")
            except Exception:
                detail = (resp.text or "")[:200]
            raise RuntimeError(f"HTTP {resp.status_code}: {detail or resp.reason}")
        return resp.json()
    finally:
        for fobj in opened:
            fobj.close()


def call_yolo_remote(frames: List[Path], remote_url: str) -> Tuple[list, Optional[str]]:
    """Colab 등에서 띄운 /predict 서버로 프레임들을 나눠 보내고 결과를 받아온다.
    반환 형태는 yolo_infer.call_yolo()와 동일하게 맞춰서, analysis.py에서
    로컬/원격 추론을 그대로 바꿔 쓸 수 있도록 한다.
    청크 하나가 실패해도 나머지 청크 결과는 살리고, 에러 메시지에 실패 내역을 담는다.
    """
    if not frames:
        return [], None

    url = remote_url.rstrip("/") + "/predict"
    by_name = {}
    errors = []

    for i in range(0, len(frames), REMOTE_CHUNK_SIZE):
        chunk = frames[i:i + REMOTE_CHUNK_SIZE]
        try:
            data = _post_chunk(url, chunk)
        except Exception as e:
            errors.append(f"프레임 {i + 1}~{i + len(chunk)}: {e}")
            continue
        for r in data.get("results", []):
            by_name[r.get("filename")] = r

    err = None
    if errors:
        err = f"Colab 서버 호출 일부 실패 ({len(errors)}개 청크): " + " | ".join(errors[:3])

    items = []
    for fp in frames:
        r = by_name.get(fp.name)
        if not r:
            continue

        defects = set()
        boxes_out = []
        for box in r.get("boxes", []):
            raw_name = str(box.get("class_name", ""))
            if _normalize_yolo_class_name(raw_name) in YOLO_IGNORE_CLASSES:
                continue
            label = _yolo_display_label(raw_name)
            defects.add(label)
            boxes_out.append({**box, "label": label})

        if defects:
            items.append({
                "time_s": int(fp.stem),
                "defects": sorted(defects),
                "distance_text": "",
                "diameter_text": "",
                "pipe_id": "",
                "site_name": "",
                "direction": "",
                "model": "yolo-remote",
                "boxes": boxes_out,
            })

    return items, None
