import os
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .config import USE_OCR

# paddlepaddle 3.x 기본 MKLDNN 가속이 이 환경(CPU 빌드)에서 텍스트 감지 모델과
# 충돌해 NotImplementedError를 낸다. PaddleOCR import 전에 꺼야 적용된다.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

# ───────── PaddleOCR 전역 캐시 ─────────
_PADDLEOCR_READER = None


def get_paddleocr_reader():
    """PaddleOCR 인스턴스를 전역에서 1번만 생성해서 재사용."""
    global _PADDLEOCR_READER
    if _PADDLEOCR_READER is not None:
        return _PADDLEOCR_READER
    try:
        from paddleocr import PaddleOCR
        _PADDLEOCR_READER = PaddleOCR(
            # 기본값인 server급 감지 모델(PP-OCRv5_server_det)은 MKLDNN이 꺼진
            # 이 환경의 CPU에서 호출당 30초 이상 걸린다. 오버레이 자막처럼
            # 크고 선명한 텍스트에는 mobile 감지로 충분하고 수십 배 빠르다.
            text_detection_model_name="PP-OCRv5_mobile_det",
            # 주의: PaddleOCR 3.x는 model_name을 지정하면 lang="korean"을 무시하고
            # 기본(비한글) 인식 모델을 로드한다. 그러면 현장명/조사시작 등 한글이
            # 전부 빈 문자열로 나온다. 반드시 한글 인식 모델을 명시해야 한다.
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception:
        _PADDLEOCR_READER = None
    return _PADDLEOCR_READER


def _paddleocr_read_text(reader, image) -> List[str]:
    """PaddleOCR predict() 결과에서 인식된 텍스트만 뽑아 리스트로 반환."""
    try:
        results = reader.predict(image)
    except Exception:
        return []
    texts = []
    for res in results:
        texts.extend(res.get("rec_texts") or [])
    return texts


def try_ocr_find_range(video_path: Path) -> Tuple[int, int]:
    """
    영상 앞부분(최대 30초)에서 2초 간격으로 OCR 돌려서
    '조사시작' 또는 '시작' 텍스트가 나온 시점을 start_s로 사용.
    찾지 못하면 start_s=0. end_s는 항상 전체 길이(dur).
    """
    if not USE_OCR:
        return (0, 0)

    reader = get_paddleocr_reader()
    if not reader:
        return (0, 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (0, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = int(total / fps) if fps > 0 else 0

    start_s = 0
    end_s = dur

    try:
        for t in range(0, min(30, dur), 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, f = cap.read()
            if not ok:
                continue

            h, w = f.shape[:2]
            roi = f[int(h * 0.3):int(h * 0.7), int(w * 0.2):int(w * 0.8)]

            txt = " ".join(_paddleocr_read_text(reader, roi))

            if "조사시작" in txt or "시작" in txt:
                start_s = t
                break
    finally:
        cap.release()

    return (start_s, end_s)


def _read_frame_image(frame_path: Path):
    try:
        img_array = np.fromfile(str(frame_path), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _ocr_region(im, y1_ratio, y2_ratio, x1_ratio, x2_ratio, upscale: float = 2.0) -> str:
    reader = get_paddleocr_reader()
    if not reader:
        return ""
    try:
        h, w = im.shape[:2]
        y1, y2 = int(h * y1_ratio), int(h * y2_ratio)
        x1, x2 = int(w * x1_ratio), int(w * x2_ratio)
        roi = im[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if roi.size == 0:
            return ""
        if upscale and upscale != 1.0:
            roi = cv2.resize(roi, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        txt_list = _paddleocr_read_text(reader, roi)
        return " ".join(str(x).strip() for x in txt_list if str(x).strip())
    except Exception:
        return ""


def _clean_ocr_text(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _parse_distance_from_text(text: str) -> str:
    s = _clean_ocr_text(text)
    s = s.replace(",", ".").replace("O", "0").replace("o", "0")

    # 화면 중앙의 이동속도 표시("x.x m/min")는 거리와 형식이 같아 혼동된다.
    # 거리로 오인하지 않도록 먼저 제거한다.
    s = re.sub(r'\d{1,6}\.\d{1,2}\s*(?:m|M|ｍ|Ｍ)?\s*/?\s*min', ' ', s, flags=re.IGNORECASE)

    # 거리는 좌상단(가장 왼쪽/위)에 표시되므로 첫 번째 매치를 사용한다.
    with_unit = re.findall(
        r'(?<![\dA-Za-z])(\d{1,6}\.\d{1,2})\s*(?:m|M|ｍ|Ｍ)\b',
        s
    )
    if with_unit:
        return with_unit[0] + "m"

    without_unit = re.findall(r'(?<![\dA-Za-z])(\d{1,6}\.\d{1,2})(?![\dA-Za-z])', s)
    if without_unit:
        return without_unit[0] + "m"

    return ""


def _parse_pipe_id_from_text(text: str) -> str:
    s = _clean_ocr_text(text)
    candidates = re.findall(r'\b[A-Za-z0-9]{1,12}(?:-[A-Za-z0-9]{1,12})+\b', s)
    for cand in candidates:
        if not re.search(r'\d+\.\d+\s*m', cand, flags=re.IGNORECASE):
            return cand
    return ""


def _parse_diameter_from_text(text: str) -> str:
    s = _clean_ocr_text(text).replace("O", "0").replace("o", "0")

    # 우상단 날짜(2024-10-24)·시간(10:46:49)이 관경 숫자로 오인되지 않도록 먼저 제거.
    s = re.sub(r'\d{4}\s*[-.\s]\s*\d{1,2}\s*[-.\s]\s*\d{1,2}', ' ', s)
    s = re.sub(r'\d{1,2}\s*[:：]\s*\d{1,2}(?:\s*[:：]\s*\d{1,2})?', ' ', s)

    material = r'(?:PVC|PE|VG|PC|흄관|흄|오수|우수|하수|합류)'
    # 1) 재질/관종 바로 앞의 3자리(관경). 사이 잡음(공백/오인된 1 등) 허용.
    m = re.search(rf'(\d{{3}})[\s\d/·|!lIi]{{0,4}}{material}', s, flags=re.IGNORECASE)
    if m:
        return f"Φ{m.group(1)}"
    # 2) Φ/Ø 기호 뒤 3자리.
    m = re.search(r'[ΦØ]\s*(\d{3})', s)
    if m:
        return f"Φ{m.group(1)}"
    # 3) 그 외 독립된 3자리(4자리 연도는 경계 조건으로 자동 배제).
    m = re.search(r'(?<!\d)(\d{3})(?!\d)', s)
    if m:
        return f"Φ{m.group(1)}"
    return ""


def _parse_site_from_text(text: str) -> str:
    s = _clean_ocr_text(text)
    m = re.search(
        r'(?:현장명|공사명|사업명|조사명|위치|SITE|Site)\s*[:：=\-]?\s*(.{2,60})',
        s,
        flags=re.IGNORECASE
    )
    if m:
        return re.split(r'\s{2,}|[|]', m.group(1).strip())[0].strip()

    parts = [p.strip() for p in re.split(r'[|/]', s) if p.strip()]
    parts = [
        p for p in parts
        if len(p) >= 2
        and not _parse_distance_from_text(p)
        and not _parse_diameter_from_text(p)
        and not _parse_pipe_id_from_text(p)
    ]
    return max(parts, key=len).strip() if parts else ""


def ocr_overlay_metadata(frames: List[Path]) -> dict:
    """오버레이 메타데이터(거리/관로ID/관경/현장명)를 프레임 1장에서 1번만 읽는다.

    같은 정보가 모든 프레임에 반복 표시되므로 여러 프레임을 읽을 필요가 없다.
    중간 프레임을 우선 쓰고, 읽기 실패 시 첫/마지막 프레임으로 폴백한다.
    전체 화면 OCR은 관경(Φ) 폴백 파싱용으로 1번만, 업스케일 없이 돌린다
    (2배 업스케일 전체 화면 OCR이 기존 병목이었음).
    """
    empty = {"distance_text": "", "pipe_id": "", "diameter_text": "", "site_name": ""}
    if not USE_OCR or not frames:
        return empty

    im = None
    for idx in (len(frames) // 2, 0, len(frames) - 1):
        im = _read_frame_image(frames[idx])
        if im is not None:
            break
    if im is None:
        return empty

    top_left_text = _ocr_region(im, 0.00, 0.24, 0.00, 0.30)
    bottom_left_text = _ocr_region(im, 0.68, 1.00, 0.00, 0.55)
    right_text = _ocr_region(im, 0.06, 0.22, 0.55, 1.00)  # 관경/재질/관종 라인(우상단)
    all_text = " ".join([top_left_text, bottom_left_text, right_text, _ocr_region(im, 0.00, 1.00, 0.00, 1.00, upscale=1.0)])

    return {
        "distance_text": _parse_distance_from_text(top_left_text),
        "pipe_id": _parse_pipe_id_from_text(top_left_text),
        "diameter_text": _parse_diameter_from_text(right_text) or _parse_diameter_from_text(all_text),
        "site_name": _parse_site_from_text(bottom_left_text),
    }


def ocr_distance_from_frame(frame_path: Path) -> str:
    if not USE_OCR:
        return ""
    try:
        im = _read_frame_image(frame_path)
        if im is None:
            return ""
        top_left_text = _ocr_region(im, 0.00, 0.24, 0.00, 0.30)
        return _parse_distance_from_text(top_left_text)
    except Exception:
        return ""


def normalize_diameter_text(raw: str) -> str:
    """
    직경 텍스트에서 'Φ' 뒤 숫자만 뽑아서 'Φ 200' 형식으로 변환.
    예) '관경 Φ200mm' -> 'Φ 200'
    """
    if not raw:
        return ""
    s = str(raw)
    m = re.search(r'[Φφ]\s*([0-9]+)', s)
    if m:
        return f"Φ {m.group(1)}"
    return s
