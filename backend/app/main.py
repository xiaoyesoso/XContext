from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import archive, chat, health, items, layers, metrics, profiles, windows
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

# Serve the frontend demo if the static directory exists.
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
