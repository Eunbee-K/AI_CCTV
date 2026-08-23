"""공모전 제출용 실행 — 배치파일 하나로 뜨는 로컬 전용 데모.

`serve_web_cpu.py`와 같은 로컬 CPU 추론이지만 **로그인이 없다.** 심사위원이
배치파일만 눌러 바로 화면을 보게 하는 것이 목적이라, 계정 입력 단계를 뺐다.

  serve_web.py       → Colab(ngrok) 원격 추론 + 로그인
  serve_web_cpu.py   → 로컬 CPU 추론 + 로그인 (사내 배포용)
  serve_demo.py      → 로컬 CPU 추론, 로그인 없음 (이 파일, 제출/시연용)

**외부 통신을 하지 않는다.** 원격 추론 주소를 비우고 LLM 판독도 꺼둔 채 뜬다.
두 모델(assets/best.pt, assets/classifier.onnx)만으로 이 PC에서 전부 돈다.

사용:
  python serve_demo.py --port 8080
"""
import argparse
import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# backend를 import하기 전에 정해야 한다 — config가 import 시점에 한 번만 읽는다.
os.environ["REMOTE_YOLO_URL"] = ""   # 원격 비활성화 = 로컬 CPU 추론
os.environ["AUTH_ENABLED"] = "0"     # 로그인 없음 (제출/시연용)

import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute

from backend import defect_classifier
from backend.config import (CLASSIFIER_MODEL_PATH, YOLO_CONF, YOLO_IMGSZ,
                            YOLO_MODEL_PATH, resource_path)
from backend.server import create_app
from backend.state import state

# 원격 추론이 없으므로 Colab 연결 패널을 감춘다. DOM에서 지우면 app.js가
# 거기에 addEventListener를 걸다 예외가 나서 초기화가 통째로 죽는다.
# 요소는 남기고 화면에서만 뺀다(serve_web_cpu.py와 같은 이유).
HIDE_COLAB_CSS = "<style>#colabCard{display:none !important}</style>\n</head>"


def _index_html() -> str:
    html = (resource_path("ui") / "index.html").read_text(encoding="utf-8")
    return html.replace("</head>", HIDE_COLAB_CSS, 1)


def _attach_index_override(app):
    """'/' 와 '/index.html'을 가로채 패치된 HTML을 준다.

    create_app()이 마지막에 StaticFiles를 '/'에 mount하는데 Starlette은 등록
    순서대로 매칭하므로, 라우트 목록 맨 앞에 끼워 넣어야 mount보다 먼저 잡힌다.
    """
    async def index(_request=None):
        return HTMLResponse(_index_html())

    for path in ("/index.html", "/"):
        app.router.routes.insert(0, APIRoute(path, index, methods=["GET"]))
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true", help="브라우저를 열지 않는다")
    ap.add_argument("--keep-session", action="store_true",
                    help="지난 작업을 이어서 연다 (기본은 빈 화면으로 시작)")
    args = ap.parse_args()

    missing = []
    if not YOLO_MODEL_PATH.exists():
        missing.append(f"판독기 모델: {YOLO_MODEL_PATH}")
    if not CLASSIFIER_MODEL_PATH.exists():
        missing.append(f"분류기 모델: {CLASSIFIER_MODEL_PATH}")
    if missing:
        print("  [오류] 모델 파일이 없습니다:")
        for m in missing:
            print(f"         {m}")
        print()
        print("  assets/ 폴더에 모델 파일을 두고 다시 실행하세요.")
        return 1

    # create_app()이 세션 복원·로그인 관련 줄을 찍는데, 이 실행판에는 로그인이
    # 없고 지난 작업도 열지 않으므로 사용자에게는 혼란만 준다. 그 구간의 출력만
    # 삼킨다(오류는 예외로 올라오므로 조용히 묻히지 않는다).
    import contextlib
    import io as _io

    with contextlib.redirect_stdout(_io.StringIO()):
        app = _attach_index_override(create_app())

    # 지난 세션에 Colab 주소가 남아 있으면 session_store.load()가 되살린다.
    # 이 실행판은 로컬 전용이므로 복원 뒤에 한 번 더 지운다.
    if state.remote_yolo_url:
        state.remote_yolo_url = ""
    # 외부 API를 부르는 LLM 판독도 꺼둔 채 시작한다(화면에서 켤 수는 있다).
    state.llm_enabled = False

    # 시연용은 **빈 화면에서 시작한다.** create_app()이 지난 세션을 복원하는데,
    # 남의 PC에서 처음 열었을 때 모르는 영상과 결과표가 이미 떠 있으면
    # 무엇을 보고 있는 것인지 알 수 없다. 파일은 지우지 않고 메모리만 비운다.
    if not args.keep_session:
        state.video_queue.clear()
        state.video_data_map.clear()

    print(f"  판독기 : {YOLO_MODEL_PATH.name}")
    ok, why = defect_classifier.availability()
    if ok:
        print(f"  분류기 : {CLASSIFIER_MODEL_PATH.name} "
              f"({len(defect_classifier._classes)}종)")
    else:
        print(f"  분류기 : 사용 불가 → 옛 이진 필터로 동작 ({why})")
    print("  추론   : 이 PC의 CPU (외부 통신 없음)")
    print()

    if not args.no_browser:
        # 서버가 뜨기 전에 열면 빈 화면이 나온다. uvicorn이 포트를 잡은 뒤
        # 열도록 기동 훅에 건다.
        import threading

        def _open():
            webbrowser.open(f"http://localhost:{args.port}")

        threading.Timer(1.5, _open).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
