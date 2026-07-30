"""화면에 박힌 오버레이 자막(거리/현장명/위경도 등) 위치의 YOLO 오탐을 걸러낸다.

CCTV 영상 위에 흰 글씨로 겹쳐진 자막(특히 우하단의 '위도/경도' 숫자 블록)은
고대비 선형 텍스처라 파손(BK)·균열 등으로 자주 오탐된다. 자막 위치는 고정이고
OCR이 읽는 위치와 동일하므로, 그 영역에 대부분 걸친 탐지 박스는 결함에서 제외한다.
"""
from typing import Sequence

# (x1, y1, x2, y2) — 프레임 크기로 정규화(0~1). OCR 오버레이 영역과 동일한 위치.
OVERLAY_TEXT_REGIONS = [
    (0.00, 0.00, 0.32, 0.24),   # 좌상단: 거리 / 관로ID
    (0.33, 0.00, 0.63, 0.12),   # 상단중앙: 이동속도(m/min)
    (0.72, 0.00, 1.00, 0.16),   # 우상단: 날짜 / 시간
    (0.54, 0.09, 1.00, 0.22),   # 우중단: Φ관경 / 재질 / 관종
    (0.00, 0.80, 0.60, 1.00),   # 좌하단: 현장명
    (0.60, 0.68, 1.00, 1.00),   # 우하단: 맨홀깊이 / 위도 / 경도
]

# 박스 면적의 이 비율 이상이 한 자막 영역 안에 들어가면 자막 오탐으로 보고 버린다.
# 0.70이면 화면 대부분을 덮는 큰 탐지(관 전체)나 자막을 살짝 걸친 실제 결함은 유지된다.
OVERLAY_COVER_THRESH = 0.70


def is_overlay_text_box(xyxy: Sequence[float], img_w: float, img_h: float,
                        thresh: float = OVERLAY_COVER_THRESH) -> bool:
    """xyxy(절대 픽셀 좌표) 박스가 자막 영역 안에 thresh 이상 들어가면 True."""
    if not xyxy or len(xyxy) != 4 or not img_w or not img_h:
        return False
    x1, y1, x2, y2 = xyxy
    barea = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if barea <= 0:
        return False
    for rx1, ry1, rx2, ry2 in OVERLAY_TEXT_REGIONS:
        RX1, RY1, RX2, RY2 = rx1 * img_w, ry1 * img_h, rx2 * img_w, ry2 * img_h
        ix = max(0.0, min(x2, RX2) - max(x1, RX1))
        iy = max(0.0, min(y2, RY2) - max(y1, RY1))
        if (ix * iy) / barea >= thresh:
            return True
    return False
