"""분석 결과를 디스크에 저장했다가 서버를 다시 켤 때 복원한다.

state는 메모리에만 있어서, 코드를 고쳐 서버를 재시작하면 몇 십 분 걸린 분석
결과가 통째로 사라졌다. 결과는 작은 JSON이라 매 변경마다 저장해도 부담이 없다.
(프레임 이미지·업로드 영상은 config.DATA_DIR 아래 고정 폴더에 있어 함께 살아남는다.)

※ 단일 사용자 기준이다. 여러 명이 동시에 쓰는 상황은 state 자체가 공용이라
   이 파일이 아니라 state 구조부터 바꿔야 한다.
"""
import json
import tempfile
from pathlib import Path
from typing import Optional

from .config import SESSION_FILE
from .state import PIPE_META_FIELDS, PROJECT_META_FIELDS, state

# 2: 보고서 정보를 현장공통(project_meta) + 관로별(videos[].meta)로 나눔
SCHEMA_VERSION = 2

# 행에서 저장할 항목. boxes_norm(오버레이 좌표)까지 담아야 복원 후에도 박스가 보인다.
_ROW_KEYS = ("time", "dist", "defects", "note", "direction", "fp", "grade",
             "boxes", "boxes_norm")
_PATH_KEYS = ("frame_path", "frame_annot_path")


def _row_to_json(row: dict) -> dict:
    out = {k: row.get(k) for k in _ROW_KEYS}
    for k in _PATH_KEYS:
        v = row.get(k)
        out[k] = str(v) if v else None
    return out


def _row_from_json(d: dict) -> dict:
    row = {k: d.get(k) for k in _ROW_KEYS}
    row["defects"] = list(row.get("defects") or [])
    row["boxes"] = list(row.get("boxes") or [])
    row["boxes_norm"] = list(row.get("boxes_norm") or [])
    row["fp"] = bool(row.get("fp"))
    row["grade"] = row.get("grade") or "중"
    for k in _PATH_KEYS:
        v = d.get(k)
        row[k] = Path(v) if v else None
    return row


def save() -> Optional[str]:
    """현재 상태를 저장한다. 실패해도 앱은 계속 돌아야 하므로 에러를 반환만 한다."""
    try:
        with state.lock:
            data = {
                "version": SCHEMA_VERSION,
                "site_name": state.site_name,
                "pipe_condition": state.pipe_condition,
                "project_meta": dict(state.project_meta),
                "remote_yolo_url": state.remote_yolo_url,
                "videos": [
                    {
                        "name": name,
                        "path": str(v["path"]),
                        "pipe_id": v.get("pipe_id", ""),
                        "dia": v.get("dia", ""),
                        "meta": dict(v.get("meta") or {}),
                        "rows": [_row_to_json(r) for r in v.get("rows", [])],
                    }
                    for name, v in state.video_data_map.items()
                ],
            }
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        # 저장 도중 죽어도 기존 파일이 깨지지 않도록 임시파일에 쓰고 교체한다
        fd, tmp = tempfile.mkstemp(dir=str(SESSION_FILE.parent), suffix=".tmp")
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        Path(tmp).replace(SESSION_FILE)
        return None
    except Exception as e:
        return f"세션 저장 실패: {e}"


def load() -> Optional[str]:
    """저장된 상태를 복원한다. 파일이 없으면 아무것도 하지 않는다."""
    if not SESSION_FILE.exists():
        return None
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return f"세션 파일을 읽지 못했습니다({e}). 새로 시작합니다."

    if data.get("version") != SCHEMA_VERSION:
        return f"세션 형식이 달라 건너뜁니다(version={data.get('version')})."

    state.site_name = data.get("site_name", "")
    state.pipe_condition = data.get("pipe_condition", "")
    saved = data.get("project_meta") or {}
    state.project_meta = {k: saved.get(k, v) for k, v in PROJECT_META_FIELDS.items()}
    if data.get("remote_yolo_url"):
        state.remote_yolo_url = data["remote_yolo_url"]

    state.video_queue.clear()
    state.video_data_map.clear()
    missing = []
    for v in data.get("videos", []):
        path = Path(v["path"])
        # 영상 파일이 사라졌어도 결함 행은 살린다(표는 보이고 재생만 안 된다).
        if not path.exists():
            missing.append(v["name"])
        state.add_video(path)
        d = state.video_data_map[path.name]
        d["pipe_id"] = v.get("pipe_id", "")
        d["dia"] = v.get("dia", "")
        saved_pipe = v.get("meta") or {}
        d["meta"] = {k: saved_pipe.get(k, dv) for k, dv in PIPE_META_FIELDS.items()}
        d["rows"] = [_row_from_json(r) for r in v.get("rows", [])]

    if missing:
        return f"영상 파일을 찾지 못했습니다(결함 행은 유지): {', '.join(missing)}"
    return None


def summary() -> str:
    n_rows = sum(len(v.get("rows", [])) for v in state.video_data_map.values())
    return f"영상 {len(state.video_data_map)}개 / 결함 {n_rows}행"
