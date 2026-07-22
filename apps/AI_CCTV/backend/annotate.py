"""탐지 결과(bbox + 라벨 + 신뢰도)를 프레임 이미지에 그려 넣는 유틸.

라벨이 한글(파손/균열 등)이라 cv2.putText로는 깨지기 때문에 PIL로 텍스트를
그린다. 원본 프레임은 그대로 두고 `<이름>_annot.jpg`를 옆에 저장한다.
"""
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

BOX_COLOR = (0, 60, 255)      # BGR — 주황빛 빨강
TEXT_COLOR = (255, 255, 255)  # 흰색

_FONT = None
_FONT_FAILED = False

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",                          # Windows 맑은고딕
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Ubuntu 나눔고딕
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",            # macOS
]


def _get_font(size: int = 18):
    global _FONT, _FONT_FAILED
    if _FONT is not None or _FONT_FAILED:
        return _FONT
    try:
        from PIL import ImageFont
        for path in _FONT_CANDIDATES:
            try:
                _FONT = ImageFont.truetype(path, size)
                return _FONT
            except Exception:
                continue
    except Exception:
        pass
    _FONT_FAILED = True
    return None


def _read_image(path: Path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _write_image(path: Path, img) -> bool:
    try:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def annotate_frame(frame_path: Path, boxes: List[dict]):
    """박스를 그린 이미지를 저장하고, UI 오버레이용 정규화(0~1) 좌표도 계산한다.

    반환: (annot_path 또는 None, norm_boxes 리스트)
    norm_boxes 항목: {"label", "confidence", "nxyxy": [x1,y1,x2,y2] (0~1)}
    — 프레임 해상도와 무관하게 화면 어디에든 겹쳐 그릴 수 있도록 정규화해 둔다.
    """
    annot_path = draw_boxes_on_frame(frame_path, boxes)

    norm_boxes = []
    img = _read_image(frame_path)
    if img is not None and boxes:
        h, w = img.shape[:2]
        for box in boxes:
            xyxy = box.get("xyxy") or []
            if len(xyxy) != 4 or not w or not h:
                continue
            norm_boxes.append({
                "label": box.get("label") or box.get("class_name") or "?",
                "confidence": box.get("confidence"),
                "nxyxy": [xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h],
            })
    return annot_path, norm_boxes


def draw_boxes_on_frame(frame_path: Path, boxes: List[dict]) -> Optional[Path]:
    """프레임 jpg에 박스들을 그려 `<stem>_annot.jpg`로 저장하고 그 경로를 반환.

    boxes 항목 형식: {"label" 또는 "class_name", "confidence", "xyxy": [x1,y1,x2,y2]}
    (yolo_infer.call_yolo / yolo_remote.call_yolo_remote가 만드는 형식 그대로)
    실패하면 None을 반환하고 원본 표시로 폴백하게 둔다.
    """
    if not boxes:
        return None
    img = _read_image(frame_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    thickness = max(2, round(min(h, w) / 300))

    labels = []  # (x1, y1, text)
    for box in boxes:
        xyxy = box.get("xyxy") or []
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, thickness)
        name = box.get("label") or box.get("class_name") or "?"
        conf = box.get("confidence")
        text = f"{name} {conf * 100:.0f}%" if isinstance(conf, (int, float)) else str(name)
        labels.append((x1, y1, text))

    if not labels:
        return None

    font = _get_font()
    if font is not None:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for x1, y1, text in labels:
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = y1 - th - 8 if y1 - th - 8 >= 0 else y1 + 2
            draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 8], fill=(255, 60, 0))
            draw.text((x1 + 4, ty + 2), text, font=font, fill=(255, 255, 255))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        # 폰트를 못 찾으면 한글이 깨지므로 영문 class_name 기준 cv2 텍스트로 폴백
        for x1, y1, text in labels:
            ty = y1 - 8 if y1 - 8 > 12 else y1 + 16
            cv2.putText(img, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

    annot_path = frame_path.with_name(f"{frame_path.stem}_annot.jpg")
    if not _write_image(annot_path, img):
        return None
    return annot_path
