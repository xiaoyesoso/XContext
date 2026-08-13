import os

from app.core.layers import LayerManager
from app.core.llm import OpenAISummarizer
from app.core.metrics import MetricsCollector
from app.core.summarizer import MockSummarizer
from app.repositories.archive import LocalArchiveRepository
from app.repositories.base import ContextRepository
from app.repositories.memory import InMemoryContextRepository
from app.services.context_service import ContextService

# Global singleton dependencies for the in-memory baseline.
# Replace with factory/dependency-injection wiring for production persistence.
_repository = InMemoryContextRepository()
_metrics_collector = MetricsCollector()

# Use the real LLM summarizer unless SUMMARIZER_MODE is set to "mock".
_summarizer = (
    MockSummarizer()
    if os.getenv("SUMMARIZER_MODE", "").lower() == "mock"
    else OpenAISummarizer()
)

_context_service = ContextService(
    _repository,
    metrics_collector=_metrics_collector,
    summarizer=_summarizer,
)
_layer_manager = LayerManager()
_archive_repository = LocalArchiveRepository(base_path="archive")


def get_context_service() -> ContextService:
    """Return the application context service instance."""
    return _context_service


def get_repository() -> InMemoryContextRepository:
    """Return the repository instance."""
    return _repository


def get_layer_manager() -> LayerManager:
    """Return the layer manager instance."""
    return _layer_manager


def get_metrics_collector() -> MetricsCollector:
    """Return the metrics collector instance."""
    return _metrics_collector


def get_archive_repository() -> ContextRepository:
    """Return the archive repository instance."""
    return _archive_repository
