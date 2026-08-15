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

        # LLM 판독(Gemini+GPT). 외부 API를 부르므로 회사망에서는 못 쓴다.
        # 그래서 화면에서 껐다 켰다 한다. 기본은 꺼짐.
        from .config import LLM_ENABLED_DEFAULT
        self.llm_enabled: bool = LLM_ENABLED_DEFAULT
        # 화면에서 넣은 API 키. **메모리에만 둔다** — 세션 파일에 저장하면 결과를
        # 주고받을 때 키가 딸려 나간다.
        self.llm_keys: dict = {"openai": "", "google": ""}

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.lock = threading.Lock()

    def add_video(self, path: Path):
        if path not in self.video_queue:
            self.video_queue.append(path)
        if path.name not in self.video_data_map:
            self.video_data_map[path.name] = {
                "path": path,
                # 등록 시점에는 자막을 아직 못 읽으므로 파일명을 임시로 쓴다.
                # 분석을 돌리면 자막 OCR로 읽은 실제 관로번호로 덮어쓴다
                # (analysis.py). 파일명은 현장 일련번호인 경우가 많아
                # 실제 관로번호와 다르다 — `5982.mp4` 자막은 `SM1-121-002`.
                "pipe_id": path.stem,
                "dia": "",
                "rows": [],
                "meta": dict(PIPE_META_FIELDS),   # 이 관로만의 야장 항목
            }

    def meta_for(self, video_name: str) -> Dict[str, str]:
        """야장에 쓸 값 한 벌. 현장 공통 + 이 관로의 항목을 합쳐서 준다."""
        v = self.video_data_map.get(video_name) or {}
        return {**self.project_meta, **(v.get("meta") or PIPE_META_FIELDS)}

    def site_name_of(self, video_name: str) -> str:
        """그 관로의 현장명. 영상별로 따로 두되, 없으면 전체 현장명으로 채운다.

        한 번에 여러 현장의 영상을 돌리는 경우가 있어 전역 하나로는 부족하다.
        """
        v = self.video_data_map.get(video_name) or {}
        return (v.get("site_name") or "").strip() or self.site_name

    def set_site_name_of(self, video_name: str, value: str) -> None:
        v = self.video_data_map.get(video_name)
        if v is None:
            return
        v["site_name"] = (value or "").strip()
        # 아직 전역이 비어 있으면 첫 입력을 기본값으로 삼는다(대개 같은 현장이다)
        if not self.site_name:
            self.site_name = v["site_name"]

    def pipe_meta(self, video_name: str) -> Dict[str, str]:
        """관로별 항목 dict. 예전 세션에서 복원된 영상이면 없을 수 있어 만들어준다."""
        v = self.video_data_map.setdefault(video_name, {})
        if not isinstance(v.get("meta"), dict):
            v["meta"] = dict(PIPE_META_FIELDS)
        return v["meta"]

    def clear_videos(self):
        self.video_queue.clear()
        self.video_data_map.clear()

    def reset_all(self, purge_files: bool = True) -> dict:
        """작업을 처음 상태로 되돌린다. 반환값은 무엇을 얼마나 지웠는지 요약.

        영상 목록·결함 행뿐 아니라 현장명/야장 항목까지 비운다. 다음 현장에
        지난 현장 정보가 남아 있으면 보고서에 그대로 섞여 들어가기 때문이다.
        추론 설정(모델·원격주소)은 작업물이 아니므로 건드리지 않는다.
        """
        import shutil

        summary = {
            "videos": len(self.video_data_map),
            "rows": sum(len(v.get("rows") or []) for v in self.video_data_map.values()),
            "frames": 0,
            "uploads": 0,
        }

        self.clear_videos()
        self.site_name = ""
        self.pipe_condition = ""
        self.project_meta = dict(PROJECT_META_FIELDS)

        if purge_files:
            for root, key in ((self.frames_root, "frames"), (self.uploads_root, "uploads")):
                if not root.exists():
                    continue
                for child in root.iterdir():
                    try:
                        if child.is_dir():
                            summary[key] += sum(1 for _ in child.rglob("*") if _.is_file())
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink()
                            summary[key] += 1
                    except OSError:
                        pass   # 재생 중이라 잠긴 파일은 건너뛴다 — 나머지는 계속 지운다
        return summary

    def get_path_by_name(self, name: str) -> Optional[Path]:
        return next((p for p in self.video_queue if p.name == name), None)


state = AppState()
