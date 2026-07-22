from fastapi import APIRouter
from pydantic import BaseModel

from ..state import state

router = APIRouter(prefix="/api/config", tags=["config"])


class RemoteUrlBody(BaseModel):
    url: str


@router.get("/remote_yolo_url")
def get_remote_url():
    return {"url": state.remote_yolo_url}


@router.post("/remote_yolo_url")
def set_remote_url(body: RemoteUrlBody):
    state.remote_yolo_url = body.url.strip()
    return {"status": "ok", "url": state.remote_yolo_url}
