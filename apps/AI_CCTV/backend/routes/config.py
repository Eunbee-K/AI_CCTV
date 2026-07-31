from fastapi import APIRouter
from pydantic import BaseModel

from ..state import state
from ..yolo_remote import normalize_remote_url, probe_remote

router = APIRouter(prefix="/api/config", tags=["config"])


class RemoteUrlBody(BaseModel):
    url: str


@router.get("/remote_yolo_url")
def get_remote_url():
    return {"url": state.remote_yolo_url}


@router.post("/remote_yolo_url")
def set_remote_url(body: RemoteUrlBody):
    """주소를 저장하면서 실제로 서버가 응답하는지까지 확인한다.

    저장만 하고 '연결됨'으로 표시하면, 주소가 틀렸어도 분석을 돌릴 때가 되어서야
    실패를 알게 된다. 연결 시점에 확인해서 원인을 바로 알려준다.
    """
    url = normalize_remote_url(body.url)
    state.remote_yolo_url = url
    if not url:
        return {"status": "ok", "url": "", "reachable": False, "detail": "연결 해제 (로컬 추론 사용)"}

    reachable, detail = probe_remote(url)
    return {"status": "ok", "url": url, "reachable": reachable, "detail": detail}
