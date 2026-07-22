import asyncio

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import resource_path
from .routes import analysis as analysis_routes
from .routes import config as config_routes
from .routes import export as export_routes
from .routes import results as results_routes
from .routes import videos as videos_routes
from .state import state


def create_app() -> FastAPI:
    app = FastAPI(title="AI CCTV Inspector")

    @app.on_event("startup")
    async def _capture_loop():
        state.loop = asyncio.get_running_loop()

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
