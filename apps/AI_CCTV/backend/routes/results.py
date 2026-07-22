from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

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


@router.get("")
def get_results():
    return {
        "site_name": state.site_name,
        "analyzing": state.analyzing,
        "videos": [
            {"name": name, "pipe_id": v["pipe_id"], "dia": v["dia"], "row_count": len(v["rows"])}
            for name, v in state.video_data_map.items()
        ],
        "display": build_results_view(),
    }


@router.post("/site_name")
def set_site_name(body: SiteNameBody):
    state.site_name = body.value
    return {"status": "ok"}


@router.patch("/row")
def patch_row(body: EditRowBody):
    err = edit_row(body.video, body.time_s, body.field, body.value)
    if err:
        raise HTTPException(400, err)
    result_update(body.video)
    return {"status": "ok"}


@router.post("/manual_row")
def post_manual_row(body: ManualRowBody):
    err = add_manual_row(body.video, body.time_s)
    if err:
        raise HTTPException(400, err)
    result_update(body.video)
    return {"status": "ok"}


@router.delete("/row")
def delete_single_row(body: DeleteRowBody):
    err = delete_row(body.video, body.time_s)
    if err:
        raise HTTPException(400, err)
    result_update(body.video)
    return {"status": "ok"}


@router.delete("/group")
def delete_row_group(body: DeleteGroupBody):
    err = delete_group(body.video, body.dist)
    if err:
        raise HTTPException(400, err)
    result_update(body.video)
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
