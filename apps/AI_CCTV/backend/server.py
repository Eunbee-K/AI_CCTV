import asyncio
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from . import auth as auth_mod
from .config import resource_path
from .routes import analysis as analysis_routes
from .routes import config as config_routes
from .routes import export as export_routes
from .routes import results as results_routes
from .routes import videos as videos_routes
from .state import state


class NoCacheStaticFiles(StaticFiles):
    """UI 파일(html/css/js)을 브라우저가 캐시하지 않게 한다.

    캐시가 남으면 서버 코드를 고쳐도 화면은 예전 그대로라, 매번 Ctrl+Shift+R을
    눌러야 하고 "고쳤는데 안 된다"로 이어진다. UI는 작아서 캐시 이득이 거의 없다.
    (MJPEG 스트림 등은 라우터가 처리하므로 여기 영향을 받지 않는다.)
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp


def create_app() -> FastAPI:
    app = FastAPI(title="AI CCTV Inspector")

    # 안쪽=인증 게이트, 바깥=세션. add_middleware는 나중에 add한 것이 바깥(먼저 실행)
    # 이므로 SessionMiddleware를 나중에 add해야 게이트에서 scope["session"]을 읽는다.
    app.add_middleware(auth_mod.AuthGateMiddleware)
    app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "ai-cctv-demo-secret-change-me"))

    @app.on_event("startup")
    async def _capture_loop():
        state.loop = asyncio.get_running_loop()

    app.include_router(auth_mod.router)
    app.include_router(videos_routes.router)
    app.include_router(videos_routes.preview_router)
    app.include_router(analysis_routes.router)
    app.include_router(analysis_routes.ws_router)
    app.include_router(results_routes.router)
    app.include_router(export_routes.router)
    app.include_router(config_routes.router)

    ui_dir = resource_path("ui")
    app.mount("/", NoCacheStaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app
