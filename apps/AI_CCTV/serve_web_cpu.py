"""웹 배포용 실행 — 이 PC의 CPU로 직접 추론 (Colab 없이).

`serve_web.py`는 그대로 두고 이 파일만 따로 쓴다. 차이는 추론 위치 하나뿐이다:

  serve_web.py      → Colab(ngrok) 원격 추론.  UI에 연결 패널이 있다.
  serve_web_cpu.py  → 이 PC의 CPU로 로컬 추론.  `assets/best.pt`가 있어야 한다.

로컬 추론에 GPU는 필요 없다. yolo11l·imgsz 960 기준 프레임당 약 0.44초(22코어)라
6분짜리 영상이 1~2분이면 끝난다. 학습은 얘기가 다르다 — CPU로는 비현실적이니
GPU에서 돌려야 한다.

사용 예:
  AUTH_USERS="admin:1234" python serve_web_cpu.py --port 8001

모델을 바꾸려면 YOLO_MODEL_PATH 환경변수를 쓰거나 assets/best.pt를 교체한다.
"""
import argparse
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# backend를 import하기 전에 정해야 한다 — config가 import 시점에 한 번만 읽는다.
os.environ["REMOTE_YOLO_URL"] = ""        # 원격 비활성화 = 로컬 CPU 추론
os.environ["AUTH_ENABLED"] = "1"
os.environ.setdefault("AUTH_USERS", "admin:admin1234")

import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute

from backend.config import YOLO_CHUNK_SIZE, YOLO_CONF, YOLO_IMGSZ, YOLO_MODEL_PATH, resource_path
from backend.server import create_app
from backend.state import state

# 이 실행판에는 원격 추론이 없으므로 Colab 연결 패널을 감춘다.
# DOM에서 지우지 않고 CSS로만 감추는 이유: ui/js/app.js가 #btnConnectRemote 등에
# addEventListener를 걸기 때문에, 요소가 사라지면 거기서 예외가 나서 그 뒤의
# 초기화(스플리터·탭 전환 등)가 통째로 죽는다. 요소는 남기고 화면에서만 뺀다.
HIDE_COLAB_CSS = (
    "<style>#colabCard{display:none !important}</style>\n</head>"
)


def _index_html() -> str:
    html = (resource_path("ui") / "index.html").read_text(encoding="utf-8")
    return html.replace("</head>", HIDE_COLAB_CSS, 1)


def _attach_index_override(app):
    """'/' 와 '/index.html'을 가로채 패치된 HTML을 준다.

    create_app()이 마지막에 StaticFiles를 '/'에 mount하는데, Starlette은 등록
    순서대로 매칭하므로 그 뒤에 라우트를 더해도 mount가 먼저 잡아먹는다.
    그래서 라우트 목록 맨 앞에 끼워 넣는다.
    """
    async def index(_request=None):
        return HTMLResponse(_index_html())

    for path in ("/index.html", "/"):
        app.router.routes.insert(0, APIRoute(path, index, methods=["GET"]))
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    if not YOLO_MODEL_PATH.exists():
        print(f"[cpu] 모델 파일이 없습니다: {YOLO_MODEL_PATH}")
        print("[cpu] 학습된 .pt를 그 경로에 두거나 YOLO_MODEL_PATH로 지정하세요.")
        return 1

    app = _attach_index_override(create_app())

    # 지난 세션에 Colab 주소가 남아 있으면 session_store.load()가 되살린다.
    # 이 실행판은 로컬 전용이므로 복원 뒤에 한 번 더 지운다.
    if state.remote_yolo_url:
        print(f"[cpu] 세션에 남아있던 원격 주소를 무시합니다: {state.remote_yolo_url}")
        state.remote_yolo_url = ""

    print(f"[cpu] 로컬 CPU 추론 · 모델: {YOLO_MODEL_PATH.name}")
    print(f"[cpu] conf={YOLO_CONF} imgsz={YOLO_IMGSZ} chunk={YOLO_CHUNK_SIZE}장")
    print(f"[cpu] 로그인 계정: {os.environ['AUTH_USERS']}")
    print(f"[cpu] 접속(이 PC):      http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        try:
            print(f"[cpu] 접속(같은 네트워크): http://{socket.gethostbyname(socket.gethostname())}:{args.port}")
        except OSError:
            pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
