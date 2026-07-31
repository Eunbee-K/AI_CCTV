import os
import sys
from pathlib import Path


def resource_path(rel: str) -> Path:
    """Dev 실행과 PyInstaller(onedir/onefile) 실행 모두에서 리소스 경로를 통일해서 반환."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / rel


APP_ROOT = Path(__file__).resolve().parent.parent

# ───────── YOLO 설정 ─────────
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", str(resource_path("assets/best.pt"))))
# Colab 원격 추론 서버 기본 주소 (ngrok 고정 도메인). 빈 문자열이면 로컬 CPU 추론.
REMOTE_YOLO_URL = os.getenv("REMOTE_YOLO_URL", "https://financial-elitism-ebook.ngrok-free.dev")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))
YOLO_IGNORE_CLASSES = {
    x.strip().lower()
    for x in os.getenv("YOLO_IGNORE_CLASSES", "normal,none,background,ok").split(",")
    if x.strip()
}

# 필요하면 여기에 클래스별 한글 표시명을 추가/수정. 목록에 없는 클래스는
# best.pt에 저장된 클래스명이 그대로 표시된다.
YOLO_CLASS_MAP = {
    "breakage": "파손",
    "separation": "이탈",
    "damage": "손상",
    "subsidence": "침하",
    "obstacle": "연결구장애물",
    "crack": "균열",
    "intruding_pipe": "가지관 돌출",
    "intruding pipe": "가지관 돌출",
    "other": "기타",
}

# ───────── 결함 코드 ↔ 한글명 ─────────
# 하수관로 상태등급 매뉴얼 기준 25종(0~24) + 데이터셋 보조 클래스(25~30).
# 출처: docs/메타데이터(총괄).xlsx, scripts/sweep/label_formats.py::GLOBAL_CLASS_ID
DEFECT_CODE_KO = {
    "CC": "균열-원주",
    "CL": "균열-길이",
    "CM": "균열-복합",
    "SD": "표면손상",
    "BC": "좌굴",
    "LD": "라이닝 결함",
    "DF": "변형",
    "BK": "파손",
    "CX": "붕괴",
    "PO": "영구장애물",
    "HL": "천공",
    "LP": "연결관-돌출",
    "LS": "연결관-접합부이상",
    "JS": "이음부-이탈",
    "JF": "이음부-손상",
    "JD": "이음부-단차",
    "NS": "역경사",
    "SG": "침하",
    "DE": "내피생성",
    "DS": "토사퇴적",
    "DG": "폐유부착",
    "TO": "임시장애물",
    "RT": "뿌리침입",
    "IF": "침입수",
    "PB": "막힘",
    # 매뉴얼 25종에는 없지만 데이터셋에 존재하는 클래스
    "PJ": "이음부(정상)",
    "ETC": "기타",
    "IN": "내부(정상)",
    "OUT_MH": "외부-맨홀",
    "OUT_INVERT": "외부-인버트",
    "OUT_CAR": "외부-자동차",
}

# 모델이 코드 대신 한글을 내보내는 경우(현재 best.pt + YOLO_CLASS_MAP)도 있어서 역방향도 둔다
_DEFECT_KO_TO_CODE = {ko: code for code, ko in DEFECT_CODE_KO.items()}


def defect_korean(label: str) -> str:
    """결함 라벨의 한글명을 반환. 모르는 라벨이면 빈 문자열.

    모델에 따라 라벨이 코드('BK')로도, 한글('파손')로도 올 수 있어 양쪽을 받는다.
    """
    raw = str(label or "").strip()
    if not raw:
        return ""
    ko = DEFECT_CODE_KO.get(raw.upper())
    if ko:
        return ko
    return raw if raw in _DEFECT_KO_TO_CODE else ""

# ───────── 프레임 추출 설정 ─────────
FRAME_INTERVAL = 2
EXTRACT_MAX_SIDE = 768
EXTRACT_JPEG_QUALITY = 80
USE_OCR = True

# ───────── 서버 설정 ─────────
HOST = "127.0.0.1"
