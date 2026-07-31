"""웹 배포용 헤드리스 실행 (pywebview 창 없이 브라우저 접속).

기존 exe(main.py)는 로컬 단일 사용자라 로그인이 없지만, 이 스크립트로 띄우면
로그인 게이트가 켜진다(AUTH_ENABLED=1). 승인 계정은 AUTH_USERS로 관리.

사용 예:
  AUTH_USERS="admin:1234,kim:qwer" python serve_web.py --port 8000
접속:
  같은 네트워크: http://<이 PC의 IP>:8000
  외부: Cloudflare Tunnel / ngrok 로 이 포트를 노출 (예: cloudflared tunnel --url http://localhost:8000)

추론(YOLO)은 앱 화면의 '추론 서버(Colab)' 패널에서 ngrok 주소를 넣어 연결한다.
"""
import argparse
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["AUTH_ENABLED"] = "1"          # 웹 실행은 로그인 필수
os.environ.setdefault("AUTH_USERS", "admin:admin1234")  # 데모 기본 계정

import uvicorn

from backend.server import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"[web] 로그인 활성화됨. 승인 계정: {os.environ['AUTH_USERS']}")
    print(f"[web] 접속(이 PC):      http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
            print(f"[web] 접속(같은 네트워크): http://{lan_ip}:{args.port}")
        except OSError:
            pass
    print("[web] ※ 반드시 http:// 로 접속 (https:// 로 열면 Invalid HTTP request 경고)")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
