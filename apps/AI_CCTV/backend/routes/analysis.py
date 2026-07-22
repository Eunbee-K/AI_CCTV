from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..analysis import start_analysis
from ..ws_manager import manager

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
ws_router = APIRouter()


@router.post("/run")
def run_analysis():
    err = start_analysis()
    if err:
        raise HTTPException(400, err)
    return {"status": "started"}


@ws_router.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 서버 -> 클라이언트 단방향 채널. 연결 유지를 위해 수신만 대기.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
