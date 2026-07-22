import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
import webview

from backend.config import HOST
from backend.pywebview_api import Api
from backend.server import create_app


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    port = _free_port()
    app = create_app()
    config = uvicorn.Config(app, host=HOST, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 5.0
    while not server.started and time.time() < deadline:
        time.sleep(0.05)

    webview.create_window(
        "AI CCTV 결함 분석기",
        url=f"http://{HOST}:{port}/",
        js_api=Api(),
        width=1400,
        height=900,
        min_size=(1000, 700),
    )
    webview.start()

    # pywebview 창이 닫히면 torch/paddleocr 등이 붙여둔 비-데몬 스레드가
    # 남아있어도 즉시 프로세스를 종료한다.
    os._exit(0)


if __name__ == "__main__":
    main()
