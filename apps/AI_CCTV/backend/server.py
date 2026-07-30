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
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app
