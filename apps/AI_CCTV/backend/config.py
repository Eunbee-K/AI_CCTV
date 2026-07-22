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

# ───────── 프레임 추출 설정 ─────────
FRAME_INTERVAL = 2
EXTRACT_MAX_SIDE = 768
EXTRACT_JPEG_QUALITY = 80
USE_OCR = True

# ───────── 서버 설정 ─────────
HOST = "127.0.0.1"
