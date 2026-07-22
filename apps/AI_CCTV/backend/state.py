import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional


class AppState:
    """앱 전역 상태를 담는 싱글톤. 단일 사용자 로컬 데스크톱 앱이므로 DB 없이 메모리로 관리."""

    def __init__(self):
        self.video_queue: List[Path] = []
        self.video_data_map: Dict[str, dict] = {}  # name -> {path, pipe_id, dia, rows}
        self.site_name: str = ""
        self.analyzing: bool = False

        self.temp_dir = Path(tempfile.mkdtemp(prefix="ai_cctv_"))
        self.frames_root = self.temp_dir / "frames"
        self.frames_root.mkdir(parents=True, exist_ok=True)

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
            }

    def clear_videos(self):
        self.video_queue.clear()
        self.video_data_map.clear()

    def get_path_by_name(self, name: str) -> Optional[Path]:
        return next((p for p in self.video_queue if p.name == name), None)


state = AppState()
