"""데모용 로그인 게이트.

- 웹 배포(`serve_web.py`)에서만 활성화된다(`AUTH_ENABLED=1`). 기존 exe(pywebview)는
  로컬 단일 사용자라 로그인 없이 그대로 동작한다.
- 승인 계정은 환경변수 `AUTH_USERS`("아이디:비번,아이디2:비번2")로 관리. 관리자가
  이 값을 설정해 접속 권한을 부여한다.
- ※ 데모용: 비밀번호 평문 비교. 상용화 시 반드시 해시(bcrypt) + 사용자 DB로 교체할 것.
- 구글 OAuth는 이번 데모 범위 밖(아이디/비번만). 추후 Authlib로 추가 예정. [[web-demo-plan]]
"""
import os

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .config import resource_path

# 로그인 없이 접근 허용하는 경로(로그인 페이지/처리 엔드포인트)
PUBLIC_PREFIXES = ("/login", "/api/auth/login")


def auth_enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "0") == "1"


def _load_users() -> dict:
    raw = os.getenv("AUTH_USERS", "admin:admin1234")  # 데모 기본 계정
    users = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            u, p = pair.split(":", 1)
            if u.strip():
                users[u.strip()] = p
    return users


router = APIRouter(tags=["auth"])


@router.get("/login", response_class=HTMLResponse)
def login_page():
    return (resource_path("ui") / "login.html").read_text(encoding="utf-8")


@router.post("/api/auth/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if _load_users().get(username) == password:
        request.session["user"] = username
        return JSONResponse({"status": "ok", "user": username})
    return JSONResponse({"detail": "아이디 또는 비밀번호가 올바르지 않습니다."}, status_code=401)


@router.post("/api/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return JSONResponse({"status": "ok"})


@router.get("/api/auth/me")
async def me(request: Request):
    # 인증 비활성(exe/로컬) 상태에서는 게이트가 없으므로 항상 통과로 표시
    return {"user": request.session.get("user"), "auth_enabled": auth_enabled()}


class AuthGateMiddleware:
    """AUTH_ENABLED=1 일 때만, 미로그인 요청을 로그인 페이지/401로 돌린다.

    순수 ASGI 미들웨어로 구현한다. BaseHTTPMiddleware(함수형)는 응답을 버퍼링해
    MJPEG 영상 스트림(무한 응답)을 막을 수 있어 사용하지 않는다.
    SessionMiddleware가 바깥(먼저)에 있어야 scope["session"]을 읽을 수 있다.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not auth_enabled():
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        session = scope.get("session") or {}
        if session.get("user") or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return await self.app(scope, receive, send)

        if path.startswith("/api/"):
            resp = JSONResponse({"detail": "unauthorized"}, status_code=401)
        else:
            resp = RedirectResponse("/login")
        await resp(scope, receive, send)
