from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from .config import DEFECT_GRADES, EXTRACT_JPEG_QUALITY, defect_korean
from .frames import seconds_to_mmss
from .ocr import normalize_diameter_text, ocr_distance_from_frame
from .state import state


def mark_dist_conflicts(rows: List[dict]) -> None:
    """같은 거리가 2번 이상 나온 행을 **그룹으로 묶을 수 있게 표시만** 한다.

    예전에는 비고에 "확인필요"를 붙였는데, 거리 중복이 흔해서 거의 모든 행에
    붙었고 그러면 아무것도 강조하지 못한다. 비고는 검수자가 직접 쓰는 칸으로
    비워두고, 그룹 접기는 `dist_group` 플래그로 처리한다.

    손으로 넣은 행은 제외한다 — 검수하려고 일부러 찍은 지점이다.
    """
    dist_counts = {}
    for r in rows:
        dist = (r.get("dist") or "").strip()
        if not dist or r.get("manual"):
            continue
        dist_counts[dist] = dist_counts.get(dist, 0) + 1

    for r in rows:
        dist = (r.get("dist") or "").strip()
        r["dist_group"] = bool(
            dist and not r.get("manual") and dist_counts.get(dist, 0) > 1)


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
        # 결함 등급(소/중/대). 파이프에셋 야장 캡션에 들어간다. 기본 "중".
        "grade": row.get("grade") or "중",
        "note": row.get("note", ""),
        "direction": row.get("direction", ""),
        "boxes": row.get("boxes_norm", []),  # 오버레이용 정규화(0~1) 박스
        "fp": bool(row.get("fp", False)),    # 오탐(false positive) 마킹
        "filename": fname,
    }


def build_results_view(only_video: Optional[str] = None) -> List[dict]:
    """결과표에 뿌릴 목록(구분선/단독 행/거리 그룹)을 만든다.

    only_video를 주면 그 영상만 담는다. 여러 영상을 분석하면 결과가 한 줄로 쭉
    이어져서 어느 관로 것인지 헷갈리므로, 화면에서는 선택한 영상만 본다.
    """
    display = []

    items = state.video_data_map.items()
    if only_video:
        v = state.video_data_map.get(only_video)
        items = [(only_video, v)] if v else []

    for fname, v_data in items:
        rows = v_data["rows"]
        rows.sort(key=lambda x: x["time"])

        display.append({"type": "separator", "filename": fname})

        no_dist_rows = []
        dist_groups: dict = {}
        for r in rows:
            dist = (r.get("dist") or "").strip()
            # 손으로 넣은 행은 거리가 같아도 묶지 않는다 — 검수하려고 따로 찍은
            # 지점이라 남의 그룹 안에 접혀 들어가면 찾을 수가 없다.
            if not dist or r.get("manual"):
                no_dist_rows.append(r)
            else:
                dist_groups.setdefault(dist, []).append(r)

        seq = 1

        # 거리 없는 단독 행과 거리 그룹을 한 줄로 세워 '시각' 순으로 정렬한다.
        # 예전에는 거리 없는 행을 전부 앞에 몰아넣어서, 수동으로 추가한 행이
        # 엉뚱하게 맨 위에 붙는 것처럼 보였다.
        entries = [(r["time"], "row", r) for r in no_dist_rows]
        for dist, group_rows in dist_groups.items():
            group_rows.sort(key=lambda r: r["time"])
            entries.append((group_rows[0]["time"], "group", (dist, group_rows)))
        entries.sort(key=lambda e: e[0])

        for _, kind, payload in entries:
            if kind == "row":
                display.append({
                    "type": "row",
                    **_row_json(seq, payload, fname, v_data),
                })
                seq += 1
                continue

            dist, group_rows = payload
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
        elif field == "grade":
            v = str(value).strip()
            if v not in DEFECT_GRADES:
                return f"등급은 {'/'.join(DEFECT_GRADES)} 중 하나여야 합니다: {v}"
            row["grade"] = v
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


def add_manual_row(video: str, time_s: int) -> Tuple[Optional[str], int]:
    """수동 행을 추가하고 (오류메시지, 실제로 넣은 시각)을 돌려준다.

    요청한 초가 이미 차 있으면 가장 가까운 빈 초로 옮기므로, 어디에 들어갔는지
    호출한 쪽에 알려줘야 화면에서 그 행을 잡아줄 수 있다.
    """
    v_data = state.video_data_map.get(video)
    if not v_data:
        return f"Unknown video: {video}", time_s

    # 행은 (영상, 시각)으로 구분하므로 같은 초에 두 개가 있으면 안 된다.
    # 예전에는 여기서 그냥 실패했는데, 결함이 몰린 구간에서는 버튼이 계속 안 먹는
    # 것처럼 보였다. 대신 비어 있는 가장 가까운 초를 찾아 거기에 넣는다.
    taken = {r.get("time") for r in v_data["rows"]}
    if time_s in taken:
        duration = int(v_data.get("duration") or 0)
        for step in range(1, 61):
            for cand in (time_s + step, time_s - step):
                if cand >= 0 and cand not in taken and (not duration or cand <= duration):
                    time_s = cand
                    break
            else:
                continue
            break
        else:
            return (f"{seconds_to_mmss(time_s)} 근처가 모두 차 있습니다. "
                    f"다른 구간으로 이동해서 추가하세요."), time_s

    path: Path = v_data["path"]
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

    # 거리는 방금 뽑은 프레임의 자막에서 읽는다. 예전에는 시간상 가까운 행의
    # 거리를 그대로 물려받았는데, 그러면 남의 거리 그룹에 끌려 들어가 버렸다.
    dist = ocr_distance_from_frame(fp) if fp.exists() else ""

    v_data["rows"].append({
        "time": time_s, "dist": dist, "defects": [],
        "note": "", "frame_path": fp, "direction": "",
        "boxes": [], "boxes_norm": [], "fp": False, "grade": "중",
        # 손으로 넣은 행은 거리가 같아도 묶지 않고 단독으로 보여준다
        "manual": True,
    })
    v_data["rows"].sort(key=lambda r: r["time"])
    mark_dist_conflicts(v_data["rows"])
    return None, time_s
