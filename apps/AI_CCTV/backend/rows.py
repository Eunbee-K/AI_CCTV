from pathlib import Path
from typing import List, Optional

import cv2

from .config import EXTRACT_JPEG_QUALITY, defect_korean
from .frames import seconds_to_mmss
from .ocr import normalize_diameter_text
from .state import state


def mark_dist_conflicts(rows: List[dict]) -> None:
    """
    같은 거리(dist)가 2번 이상 나온 경우,
    그 거리의 모든 row.note 에 '확인필요'를 한 번만 붙인다.
    (실제 rows는 삭제하지 않음)
    """
    dist_counts = {}
    for r in rows:
        dist = (r.get("dist") or "").strip()
        if not dist:
            continue
        dist_counts[dist] = dist_counts.get(dist, 0) + 1

    for r in rows:
        dist = (r.get("dist") or "").strip()
        if not dist:
            continue
        if dist_counts.get(dist, 0) > 1:
            note = (r.get("note") or "").strip()
            if "확인필요" not in note:
                r["note"] = (note + " 확인필요").strip() if note else "확인필요"


def _row_json(seq, row: dict, fname: str, v_data: dict) -> dict:
    defects = row.get("defects", [])
    return {
        "seq": seq,
        "time_s": row["time"],
        "time_str": seconds_to_mmss(row["time"]),
        "pipe_id": v_data["pipe_id"],
        "dia": v_data["dia"],
        "dist": row.get("dist", ""),
        "defects": defects,
        # 코드(BK)와 짝이 되는 한글명(파손). 표에 별도 열로 보여준다.
        "defects_ko": [defect_korean(d) for d in defects],
        "note": row.get("note", ""),
        "direction": row.get("direction", ""),
        "boxes": row.get("boxes_norm", []),  # 오버레이용 정규화(0~1) 박스
        "fp": bool(row.get("fp", False)),    # 오탐(false positive) 마킹
        "filename": fname,
    }


def build_results_view() -> List[dict]:
    """refresh_tree()의 그룹핑 로직(구분선/거리 없는 행/거리 그룹)을 순수 함수로 재구현."""
    display = []

    for fname, v_data in state.video_data_map.items():
        rows = v_data["rows"]
        rows.sort(key=lambda x: x["time"])

        display.append({"type": "separator", "filename": fname})

        no_dist_rows = []
        dist_groups: dict = {}
        for r in rows:
            dist = (r.get("dist") or "").strip()
            if not dist:
                no_dist_rows.append(r)
            else:
                dist_groups.setdefault(dist, []).append(r)

        seq = 1

        for row in no_dist_rows:
            display.append({
                "type": "row",
                **_row_json(seq, row, fname, v_data),
            })
            seq += 1

        sorted_groups = sorted(
            dist_groups.items(),
            key=lambda kv: min(r["time"] for r in kv[1])
        )

        for dist, group_rows in sorted_groups:
            group_rows.sort(key=lambda r: r["time"])

            if len(group_rows) == 1:
                display.append({
                    "type": "row",
                    **_row_json(seq, group_rows[0], fname, v_data),
                })
                seq += 1
            else:
                children = []
                for child_idx, row in enumerate(group_rows, 1):
                    children.append({
                        "type": "row",
                        **_row_json(f"{seq}-{child_idx}", row, fname, v_data),
                    })
                display.append({
                    "type": "group",
                    "seq": seq,
                    "time_str": seconds_to_mmss(group_rows[0]["time"]),
                    "pipe_id": v_data["pipe_id"],
                    "dia": v_data["dia"],
                    "dist": dist,
                    "defects_summary": f"{len(group_rows)}개 결함",
                    "note": "확인필요",
                    "filename": fname,
                    "children": children,
                })
                seq += 1

    return display


def edit_row(video: str, time_s: int, field: str, value) -> Optional[str]:
    v_data = state.video_data_map.get(video)
    if not v_data:
        return f"Unknown video: {video}"

    if field == "pipe_id":
        v_data["pipe_id"] = str(value)
    elif field == "dia":
        v_data["dia"] = normalize_diameter_text(str(value))
    else:
        row = next((r for r in v_data["rows"] if r.get("time") == time_s), None)
        if row is None:
            return f"Row not found: time_s={time_s}"
        if field == "dist":
            row["dist"] = str(value)
        elif field == "defects":
            if isinstance(value, list):
                row["defects"] = [str(d).strip() for d in value if str(d).strip()]
            else:
                row["defects"] = [d.strip() for d in str(value).split(",") if d.strip()]
        elif field == "note":
            row["note"] = str(value)
        elif field == "fp":
            row["fp"] = value if isinstance(value, bool) else str(value).lower() in ("true", "1", "y")
        else:
            return f"Unknown field: {field}"

    mark_dist_conflicts(v_data["rows"])
    return None


def delete_row(video: str, time_s: int) -> Optional[str]:
    v_data = state.video_data_map.get(video)
    if not v_data:
        return f"Unknown video: {video}"
    v_data["rows"] = [r for r in v_data["rows"] if r.get("time") != time_s]
    mark_dist_conflicts(v_data["rows"])
    return None


def delete_group(video: str, dist: str) -> Optional[str]:
    v_data = state.video_data_map.get(video)
    if not v_data:
        return f"Unknown video: {video}"
    dist = (dist or "").strip()
    v_data["rows"] = [r for r in v_data["rows"] if (r.get("dist") or "").strip() != dist]
    mark_dist_conflicts(v_data["rows"])
    return None


def add_manual_row(video: str, time_s: int) -> Optional[str]:
    v_data = state.video_data_map.get(video)
    if not v_data:
        return f"Unknown video: {video}"

    # 행은 (영상, 시각)으로 구분한다. 같은 시각에 두 개가 생기면 이후 편집/삭제가
    # 엉뚱한 행에 걸리므로 미리 막는다.
    if any(r.get("time") == time_s for r in v_data["rows"]):
        return f"{seconds_to_mmss(time_s)} 위치에는 이미 행이 있습니다. 다른 시점으로 이동해서 추가하세요."

    path: Path = v_data["path"]
    dist = ""
    if v_data["rows"]:
        closest = min(v_data["rows"], key=lambda x: abs(x["time"] - time_s))
        dist = closest["dist"]

    fp = state.frames_root / path.stem / f"{time_s:06d}.jpg"
    fp.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_s * fps))
        ok, frame = cap.read()
        if ok:
            cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), EXTRACT_JPEG_QUALITY]
            )[1].tofile(str(fp))
    finally:
        cap.release()

    v_data["rows"].append({
        "time": time_s, "dist": dist, "defects": [],
        "note": "수동", "frame_path": fp, "direction": "",
        "boxes": [], "boxes_norm": [], "fp": False,
    })
    mark_dist_conflicts(v_data["rows"])
    return None
