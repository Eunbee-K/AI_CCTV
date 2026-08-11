from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import session_store
from ..rows import add_manual_row, build_results_view, delete_group, delete_row, edit_row
from ..state import state
from ..ws_manager import result_update

router = APIRouter(prefix="/api/results", tags=["results"])


class EditRowBody(BaseModel):
    video: str
    time_s: int
    field: str
    value: Union[str, list]


class ManualRowBody(BaseModel):
    video: str
    time_s: int


class DeleteRowBody(BaseModel):
    video: str
    time_s: int


class DeleteGroupBody(BaseModel):
    video: str
    dist: str


class SiteNameBody(BaseModel):
    value: str
    video: Optional[str] = None   # 주면 그 관로만, 없으면 현장 전체 기본값


class PipeConditionBody(BaseModel):
    value: str


@router.get("")
def get_results(video: Optional[str] = None):
    """video를 주면 그 영상의 결과만 담아 돌려준다(화면 표시용).

    videos 목록과 통계는 항상 전체를 준다 — 목록에서 다른 영상으로 넘어가야 하고,
    통계 탭은 현장 전체를 봐야 하기 때문이다.
    """
    return {
        # 현장명은 관로마다 다를 수 있어 영상별로 들고 있다. 선택된 영상 것을 준다.
        "site_name": state.site_name_of(video) if video else state.site_name,
        "pipe_condition": state.pipe_condition,
        "analyzing": state.analyzing,
        "selected_video": video or "",
        "videos": [
            {"name": name, "pipe_id": v["pipe_id"], "dia": v["dia"],
             "row_count": len(v["rows"]), "site_name": state.site_name_of(name)}
            for name, v in state.video_data_map.items()
        ],
        "display": build_results_view(video),
        # 통계 탭은 전체를 봐야 하므로 따로 전부 담아준다
        "display_all": build_results_view(),
    }


@router.post("/reset")
def reset_all():
    """[초기화] — 영상·결함·현장정보를 모두 비우고 프레임/업로드 파일도 지운다.

    세션이 디스크에 남아 서버를 껐다 켜도 지난 작업이 계속 복원되므로,
    새 현장을 시작할 때 손으로 지울 방법이 필요하다.
    """
    if state.analyzing:
        raise HTTPException(400, "분석이 진행 중입니다. 끝난 뒤에 초기화하세요.")
    summary = state.reset_all()
    session_store.save()
    return {"status": "ok", **summary}


@router.post("/site_name")
def set_site_name(body: SiteNameBody):
    # video를 주면 그 관로만, 안 주면 현장 전체 기본값을 바꾼다
    if body.video:
        state.set_site_name_of(body.video, body.value)
    else:
        state.site_name = body.value
    session_store.save()
    return {"status": "ok", "value": body.value}


@router.post("/pipe_condition")
def set_pipe_condition(body: PipeConditionBody):
    value = (body.value or "").strip()
    if value not in ("", "신설", "노후"):
        raise HTTPException(400, f"관로 구분은 '신설' 또는 '노후'만 가능합니다: {value}")
    state.pipe_condition = value
    session_store.save()
    return {"status": "ok", "value": state.pipe_condition}


@router.patch("/row")
def patch_row(body: EditRowBody):
    err = edit_row(body.video, body.time_s, body.field, body.value)
    if err:
        raise HTTPException(400, err)
    result_update(body.video)
    session_store.save()
    return {"status": "ok"}


@router.post("/manual_row")
def post_manual_row(body: ManualRowBody):
    err, used_time = add_manual_row(body.video, body.time_s)
    if err:
        raise HTTPException(400, err)
    result_update(body.video)
    session_store.save()
    # 요청한 초가 차 있으면 가까운 빈 초로 옮겨 넣으므로 실제 시각을 알려준다
    return {"status": "ok", "time_s": used_time, "moved": used_time != body.time_s}


@router.delete("/row")
def delete_single_row(body: DeleteRowBody):
    err = delete_row(body.video, body.time_s)
    if err:
        raise HTTPException(400, err)
    result_update(body.video)
    session_store.save()
    return {"status": "ok"}


@router.delete("/group")
def delete_row_group(body: DeleteGroupBody):
    err = delete_group(body.video, body.dist)
    if err:
        raise HTTPException(400, err)
    result_update(body.video)
    session_store.save()
    return {"status": "ok"}


@router.get("/frame")
def get_detection_frame(video: str, time_s: int):
    """결함 행의 프레임 이미지를 반환. bbox가 그려진 버전이 있으면 그걸 우선."""
    v_data = state.video_data_map.get(video)
    if not v_data:
        raise HTTPException(404, "video not found")
    row = next((r for r in v_data["rows"] if r["time"] == time_s), None)
    if not row:
        raise HTTPException(404, "row not found")
    for key in ("frame_annot_path", "frame_path"):
        fp = row.get(key)
        if fp and Path(fp).exists():
            return FileResponse(str(fp), media_type="image/jpeg")
    raise HTTPException(404, "frame image not found")
