"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes.health import router as health_router
from api.routes.query import router as query_router
from api.routes.upload import router as upload_router
from core.config import settings


def create_app() -> FastAPI:
    """Create and configure the VaultMind API application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Agentic document intelligence API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(upload_router)
    app.include_router(query_router)

    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", tags=["root"], response_model=None)
    def root():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        return {"name": settings.app_name, "version": settings.app_version, "docs": "/docs"}

    return app


app = create_app()
