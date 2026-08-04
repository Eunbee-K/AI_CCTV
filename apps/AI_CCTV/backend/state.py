import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional

# 야장 상단 표의 입력 항목. 순서가 곧 입력 패널에 뜨는 순서다.
# 실물 야장에서도 발주처·처리구역·시공자 등은 비어 있는 경우가 많아 공란을 허용한다.

# 현장(사업) 전체에 공통인 항목 — 관로가 바뀌어도 같다.
PROJECT_META_FIELDS: Dict[str, str] = {
    "발주처": "",
    "사업기간": "",
    "처리구역": "",
    "배수구역": "",
    "배수분구": "",
    "시공자": "",
    "조사목적": "기존관상태",
    "조사자": "",
}

# 관로(영상)마다 다른 항목 — 맨홀번호·관종·거리 등은 관로별로 따로 관리한다.
PIPE_META_FIELDS: Dict[str, str] = {
    "보고서번호": "",
    "조사위치": "",
    "조사일자": "",
    "구분": "관로",
    "관종": "",
    "배수방식": "",
    "주행방향": "",
    "연장": "",
    "총주행거리": "",
    "미주행거리": "0.0m",
    "위도": "",
    "경도": "",
    "상류맨홀번호": "",
    "상류맨홀종류": "",
    "상류맨홀재질": "",
    "상류맨홀깊이": "",
    "하류맨홀번호": "",
    "하류맨홀종류": "",
    "하류맨홀재질": "",
    "하류맨홀깊이": "",
    "맨홀구조적상태등급": "",
    "맨홀운영적상태등급": "",
    "관로구조적상태등급": "",
    "관로운영적상태등급": "",
    "미주행사유": "",
}

# 전체 항목(저장·검증용). 패널에는 공통 → 관로별 순으로 나온다.
REPORT_META_FIELDS: Dict[str, str] = {**PROJECT_META_FIELDS, **PIPE_META_FIELDS}

# 입력 패널에서 항목을 어떻게 그릴지. 없는 항목은 일반 텍스트 입력.
#  readonly : 다른 곳(현장명 칸 / 영상별 관로번호)에서 관리하는 값. 확인용으로만 보여준다.
#  select   : 정해진 값 중 고르되, "기타 입력"을 고르면 직접 칠 수 있다.
REPORT_META_SPEC: Dict[str, dict] = {
    "사업명": {"type": "readonly", "hint": "현장명 칸에서 수정"},
    "관로번호": {"type": "readonly", "hint": "관로번호 칸에서 수정"},
    "배수방식": {"type": "select", "options": ["우수", "오수", "합류"]},
    "주행방향": {"type": "select", "options": ["정주행", "역주행"]},
}

# 영상 자막에서 읽어 채우는 항목 (OCR 결과 키 → report_meta 키)
OCR_TO_META = {
    "survey_date": "조사일자",
    "material": "관종",
    "drainage": "배수방식",
    "direction": "주행방향",
    "manhole_up": "상류맨홀번호",
    "manhole_down": "하류맨홀번호",
    "lat": "위도",
    "lon": "경도",
}


class AppState:
    """앱 전역 상태를 담는 싱글톤. 단일 사용자 로컬 데스크톱 앱이므로 DB 없이 메모리로 관리."""

    def __init__(self):
        self.video_queue: List[Path] = []
        self.video_data_map: Dict[str, dict] = {}  # name -> {path, pipe_id, dia, rows}
        self.site_name: str = ""
        # 관로 구분: "신설" 또는 "노후". 분석 실행 전에 반드시 골라야 한다
        # (결함 판정 기준이 달라서 빈 값으로 돌리면 보고서가 무의미해진다).
        self.pipe_condition: str = ""
        self.analyzing: bool = False

        # 파이프에셋 야장 상단 표 중 현장 전체에 공통인 항목.
        # 관로별 항목은 video_data_map[name]["meta"]에 따로 들어간다.
        self.project_meta: Dict[str, str] = dict(PROJECT_META_FIELDS)

        # 프레임·업로드 영상은 고정 폴더에 둔다. 임시 폴더에 두면 서버를 껐다 켤 때
        # 분석 결과의 사진이 전부 사라진다.
        from .config import DATA_DIR
        self.temp_dir = DATA_DIR
        self.frames_root = DATA_DIR / "frames"
        self.frames_root.mkdir(parents=True, exist_ok=True)
        # 웹에서 업로드한 영상이 저장되는 곳. 데스크톱(exe)은 로컬 경로를 직접
        # 쓰므로 사용하지 않는다.
        self.uploads_root = DATA_DIR / "uploads"
        self.uploads_root.mkdir(parents=True, exist_ok=True)

        self.yolo_model = None
        self.yolo_load_error: Optional[str] = None

        # 비어있으면 로컬 CPU로 추론. 값이 있으면 (예: Colab ngrok 고정 도메인)
        # 해당 서버로 프레임을 전송해서 추론 결과를 받아온다.
        # 기본값은 config.REMOTE_YOLO_URL (환경변수 REMOTE_YOLO_URL로 덮어쓰기 가능).
        from .config import REMOTE_YOLO_URL
        self.remote_yolo_url: str = REMOTE_YOLO_URL

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.lock = threading.Lock()

    def add_video(self, path: Path):
        if path not in self.video_queue:
            self.video_queue.append(path)
        if path.name not in self.video_data_map:
            self.video_data_map[path.name] = {
                "path": path,
                "pipe_id": path.stem,
                "dia": "",
                "rows": [],
                "meta": dict(PIPE_META_FIELDS),   # 이 관로만의 야장 항목
            }

    def meta_for(self, video_name: str) -> Dict[str, str]:
        """야장에 쓸 값 한 벌. 현장 공통 + 이 관로의 항목을 합쳐서 준다."""
        v = self.video_data_map.get(video_name) or {}
        return {**self.project_meta, **(v.get("meta") or PIPE_META_FIELDS)}

    def pipe_meta(self, video_name: str) -> Dict[str, str]:
        """관로별 항목 dict. 예전 세션에서 복원된 영상이면 없을 수 있어 만들어준다."""
        v = self.video_data_map.setdefault(video_name, {})
        if not isinstance(v.get("meta"), dict):
            v["meta"] = dict(PIPE_META_FIELDS)
        return v["meta"]

    def clear_videos(self):
        self.video_queue.clear()
        self.video_data_map.clear()

    def get_path_by_name(self, name: str) -> Optional[Path]:
        return next((p for p in self.video_queue if p.name == name), None)


state = AppState()
