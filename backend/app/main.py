from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api import archive, health, items, layers, metrics, windows
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

app.include_router(health.router)
app.include_router(items.router)
app.include_router(windows.router)
app.include_router(layers.router)
app.include_router(metrics.router)
app.include_router(archive.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
