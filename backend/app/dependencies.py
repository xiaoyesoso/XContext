import os

from app.core.key_facts import KeyFactExtractor, MockKeyFactExtractor
from app.core.layers import LayerManager
from app.core.llm import OpenAISummarizer
from app.core.metrics import MetricsCollector
from app.core.model_readable import MockModelReadableCompressor, ModelReadableCompressor
from app.core.summarizer import MockSummarizer
from app.core.user_profile import MockUserProfileExtractor, UserProfileExtractor
from app.repositories.archive import LocalArchiveRepository
from app.repositories.base import ContextRepository
from app.repositories.memory import InMemoryContextRepository
from app.services.context_service import ContextService
from app.services.profile_service import UserProfileService
from app.services.summary_service import SummaryService

# Global singleton dependencies for the in-memory baseline.
# Replace with factory/dependency-injection wiring for production persistence.
_repository = InMemoryContextRepository()
_metrics_collector = MetricsCollector()

# Mock mode switch shared by chat, summarizer, and all LLM-backed extractors.
_mock_mode = os.getenv("SUMMARIZER_MODE", "").lower() == "mock"

# Use the real LLM summarizer unless SUMMARIZER_MODE is set to "mock".
_summarizer = MockSummarizer() if _mock_mode else OpenAISummarizer()

_context_service = ContextService(
    _repository,
    metrics_collector=_metrics_collector,
    summarizer=_summarizer,
)
_layer_manager = LayerManager()
_archive_repository = LocalArchiveRepository(base_path="archive")

# User-profile subsystem: mock extractor in mock mode, mirroring the
# summarizer convention for local dev without API access.
_profile_extractor = (
    MockUserProfileExtractor() if _mock_mode else UserProfileExtractor()
)
_user_profile_service = UserProfileService(
    extractor=_profile_extractor,
    metrics_collector=_metrics_collector,
)

# Summary & detail-recall subsystem: LLM-backed extractors unless in mock mode.
_summary_service = SummaryService(
    summarizer=_summarizer,
    key_fact_extractor=MockKeyFactExtractor() if _mock_mode else KeyFactExtractor(),
    model_readable_compressor=(
        MockModelReadableCompressor() if _mock_mode else ModelReadableCompressor()
    ),
    mock_mode=_mock_mode,
)

# Conversation orchestrator: async construction at end_of_turn + sync
# injection at the start of the next turn (summaries / profile / recall).
from app.services.chat_orchestrator import ChatOrchestrator  # noqa: E402

_chat_orchestrator = ChatOrchestrator(
    context_service=_context_service,
    summary_service=_summary_service,
    profile_service=_user_profile_service,
)


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


def get_user_profile_service() -> UserProfileService:
    """Return the user-profile service instance."""
    return _user_profile_service


def get_summary_service() -> SummaryService:
    """Return the summary & detail-recall service instance."""
    return _summary_service


def get_chat_orchestrator() -> "ChatOrchestrator":
    """Return the conversation orchestrator instance."""
    return _chat_orchestrator
