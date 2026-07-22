import asyncio
from typing import List

from fastapi import WebSocket

from .state import state


class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


def push(message: dict):
    """분석 스레드(동기)에서 안전하게 WS로 메시지를 브로드캐스트."""
    loop = state.loop
    if loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(manager.broadcast(message), loop)
    except Exception:
        pass


def log(msg: str, level: str = "INFO"):
    push({"type": "log", "level": level, "msg": msg})


def progress(video: str, index: int, total: int, stage: str):
    push({"type": "progress", "video": video, "index": index, "total": total, "stage": stage})


def result_update(video: str):
    push({"type": "result_update", "video": video})


def batch_done(stats: dict, errors: list):
    push({"type": "batch_done", "stats": stats, "errors": errors})
