import os
import sys
from pathlib import Path


def resource_path(rel: str) -> Path:
    """Dev 실행과 PyInstaller(onedir/onefile) 실행 모두에서 리소스 경로를 통일해서 반환."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / rel


APP_ROOT = Path(__file__).resolve().parent.parent


def _default_data_dir() -> Path:
    """분석 결과·프레임·업로드 영상을 두는 고정 폴더.

    tempfile.mkdtemp()는 실행할 때마다 새로 만들어져서 서버를 껐다 켜면 결과가
    통째로 사라진다. OS가 청소하지 않는 사용자 데이터 폴더에 둬야 재시작 후에도
    살아남는다. AI_CCTV_DATA_DIR 환경변수로 옮길 수 있다.
    """
    env = os.getenv("AI_CCTV_DATA_DIR")
    if env:
        return Path(env)
    base = os.getenv("LOCALAPPDATA") or os.getenv("XDG_DATA_HOME")
    if base:
        return Path(base) / "AI_CCTV"
    return Path.home() / ".ai_cctv"


DATA_DIR = _default_data_dir()
SESSION_FILE = DATA_DIR / "session.json"

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


# 파이프에셋 야장(하수관거 현황 조사 보고서) 캡션 표기.
# 야장은 "(구분)결함명(등급) [시각]" 형식이고, 결함명이 우리 코드표와 미묘하게 다르다
# (하이픈 없음: 균열-원주 → 균열원주). 구분은 실제 관로 결함이면 "관로",
# 정상 구조물이나 조사 진행 상태면 "상태"를 쓴다.
# 출처: 2.3b CCTV조사 야장 (가평군 노후하수관로 정밀조사용역) 실물 PDF
DEFECT_PIPEASSET = {
    "CC": ("관로", "균열원주"),
    "CL": ("관로", "균열길이"),
    "CM": ("관로", "균열복합"),
    "SD": ("관로", "표면손상"),
    "BC": ("관로", "좌굴"),
    "LD": ("관로", "라이닝결함"),
    "DF": ("관로", "변형"),
    "BK": ("관로", "파손"),
    "CX": ("관로", "붕괴"),
    "PO": ("관로", "영구장애물"),
    "HL": ("관로", "천공"),
    "LP": ("관로", "연결관돌출"),
    "LS": ("관로", "연결관접합부"),
    "JS": ("관로", "이음부이탈"),
    "JF": ("관로", "이음부손상"),
    "JD": ("관로", "이음부단차"),
    "NS": ("관로", "역경사"),
    "SG": ("관로", "침하"),
    "DE": ("관로", "내피생성"),
    "DS": ("관로", "토사퇴적"),
    "DG": ("관로", "폐유부착"),
    "TO": ("관로", "임시장애물"),
    "RT": ("관로", "뿌리침입"),
    "IF": ("관로", "침입수"),
    "PB": ("관로", "막힘"),
    # 결함이 아니라 구조물·맥락 → 야장에서는 "상태"로 분류된다
    "PJ": ("상태", "이음부(접합부)존재"),
    "IN": ("상태", "관로내부"),
    "ETC": ("상태", "기타"),
    "OUT_MH": ("상태", "맨홀"),
    "OUT_INVERT": ("상태", "인버트"),
    "OUT_CAR": ("상태", "외부"),
}

DEFECT_GRADES = ("소", "중", "대")


def pipeasset_caption(label: str, grade: str, time_s: int) -> str:
    """야장 사진 캡션 한 줄. 예: "(관로)토사퇴적(중) [00:04:39]"

    매핑에 없는 라벨은 구분을 "관로"로 두고 라벨을 그대로 쓴다.
    """
    code = str(label or "").strip().upper()
    cat, name = DEFECT_PIPEASSET.get(code, ("관로", str(label or "").strip()))
    g = grade if grade in DEFECT_GRADES else "중"
    t = max(0, int(time_s or 0))
    return f"({cat}){name}({g}) [{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}]"


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

# 영상 앞 30초를 훑어 '조사시작' 자막을 찾는 기능. 영상당 OCR을 최대 15번 더 돌려
# 느린 데다, 못 찾으면 어차피 0초부터 보므로 기본으로 꺼둔다.
# 다시 켜려면 OCR_FIND_START=1 로 실행.
OCR_FIND_START = os.getenv("OCR_FIND_START", "0") == "1"

# ───────── 서버 설정 ─────────
HOST = "127.0.0.1"
