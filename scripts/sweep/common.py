"""스윕 자동화 공용 유틸: 설정/상태 파일 로드, 상태 저장, 로깅."""
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        Path(tmp_name).replace(path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


class StateStore:
    """run별 진행 상태를 JSON 파일 하나에 원자적으로 기록한다.

    프로세스가 중간에 죽거나 워크스테이션이 재부팅되어도, 이미 status="done"인
    run은 다시 실행하지 않고 건너뛸 수 있게 해서 스윕을 이어서 진행한다.
    """

    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {"runs": {}}
            self._flush()

    def get(self, run_name: str) -> dict:
        return self._data["runs"].get(run_name, {"status": "pending"})

    def set(self, run_name: str, **fields) -> None:
        entry = self._data["runs"].get(run_name, {})
        entry.update(fields)
        self._data["runs"][run_name] = entry
        self._flush()

    def all_runs(self) -> dict:
        return dict(self._data["runs"])

    def _flush(self) -> None:
        atomic_write_json(self.path, self._data)


def setup_logger(name: str, log_path) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
