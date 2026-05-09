"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    @app.get("/", tags=["root"])
    def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
