from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import (
    archive,
    chat,
    health,
    items,
    layers,
    metrics,
    orchestration,
    profiles,
    summaries,
    windows,
)
from app.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan hook for startup and shutdown logic."""
    logger.info("Starting Context Window Management Service")
    yield
    logger.info("Shutting down Context Window Management Service")


app = FastAPI(
    title="Context Window Management Service",
    description="Python backend service for unified Agent context management.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: allow the frontend demo to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(items.router)
app.include_router(windows.router)
app.include_router(layers.router)
app.include_router(metrics.router)
app.include_router(archive.router)
app.include_router(chat.router)
app.include_router(profiles.router)
app.include_router(summaries.router)
app.include_router(orchestration.router)

# Serve the frontend demo if a static directory exists.
# Resolution order: STATIC_DIR env override -> repo layout (XContext/frontend)
# -> container layout (/app/frontend, see Dockerfile).
_static_candidates = [
    Path(p)
    for p in (os.getenv("STATIC_DIR", ""),)
    if p
]
_static_candidates += [
    Path(__file__).resolve().parent.parent.parent / "frontend",
    Path(__file__).resolve().parent.parent / "frontend",
]
_static_dir = next((p for p in _static_candidates if p.is_dir()), None)
if _static_dir is not None:
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
