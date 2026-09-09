import re
from pathlib import Path
from typing import List, Tuple

from .config import (
    YOLO_CHUNK_SIZE, YOLO_CONF, YOLO_IGNORE_CLASSES, YOLO_IMGSZ,
    YOLO_MODEL_PATH, YOLO_CLASS_MAP,
)
from .overlay_filter import is_overlay_text_box

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _normalize_yolo_class_name(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name or "").strip().lower())


def _yolo_display_label(name: str) -> str:
    raw = str(name or "").strip()
    normalized = _normalize_yolo_class_name(raw)
    return (
        YOLO_CLASS_MAP.get(raw)
        or YOLO_CLASS_MAP.get(raw.lower())
        or YOLO_CLASS_MAP.get(normalized)
        or raw
    )


def init_yolo():
    if YOLO is None:
        return None, "ultralytics package is not installed. Run: pip install ultralytics"
    if not YOLO_MODEL_PATH.exists():
        return None, f"YOLO model not found: {YOLO_MODEL_PATH}"
    try:
        return YOLO(str(YOLO_MODEL_PATH)), None
    except Exception as e:
        return None, f"YOLO model load failed: {e}"


def call_yolo(model, frames: List[Path]) -> Tuple[list, str]:
    if not model:
        return [], "YOLO model not loaded"
    if not frames:
        return [], None

    # 프레임 리스트를 통째로 predict에 넘기면 ultralytics가 전부 한 배치로 올려서
    # 수백 장이면 수 GB를 한 번에 요구한다(112장 imgsz 960에서 3.7GB 할당 실패).
    # 원격 경로가 REMOTE_CHUNK_SIZE로 같은 문제를 피하는 것과 동일하게 나눠 돌린다.
    preds = []
    for i in range(0, len(frames), YOLO_CHUNK_SIZE):
        chunk = frames[i:i + YOLO_CHUNK_SIZE]
        try:
            preds.extend(model.predict(
                source=[str(fp) for fp in chunk],
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                verbose=False
            ))
        except Exception as e:
            # 여기까지 처리한 프레임 결과는 살려서 돌려준다
            return _collect(model, frames[:len(preds)], preds), f"YOLO Error: {e}"

    return _collect(model, frames, preds), None


def _collect(model, frames: List[Path], preds: list) -> list:
    items = []
    for fp, pred in zip(frames, preds):
        defects = set()
        boxes_out = []

        boxes = getattr(pred, "boxes", None)
        if boxes is None:
            continue

        names = getattr(pred, "names", {}) or getattr(model, "names", {}) or {}
        oh, ow = getattr(pred, "orig_shape", (0, 0))
        for box in boxes:
            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                raw_name = str(names.get(cls_id, cls_id))
                if _normalize_yolo_class_name(raw_name) in YOLO_IGNORE_CLASSES:
                    continue

                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                # 화면 자막(거리/현장명/위경도 등) 위치의 오탐 제외
                if is_overlay_text_box(xyxy, ow, oh):
                    continue

                label = _yolo_display_label(raw_name)
                defects.add(label)
                boxes_out.append({
                    "class_id": cls_id,
                    "class_name": raw_name,
                    "label": label,
                    "confidence": round(conf, 4),
                    "xyxy": xyxy,
                })
            except Exception:
                continue

        if defects:
            items.append({
                "time_s": int(fp.stem),
                "defects": sorted(defects),
                "distance_text": "",
                "diameter_text": "",
                "pipe_id": "",
                "site_name": "",
                "direction": "",
                "model": "yolo",
                "boxes": boxes_out,
            })

    return items
