import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Force the mock summarizer during tests to avoid real LLM API calls.
os.environ["SUMMARIZER_MODE"] = "mock"


@pytest.fixture(autouse=True)
def clean_runtime_artifacts():
    """Remove runtime artifacts before each test to avoid state leakage."""
    archive_path = Path("archive")
    if archive_path.exists():
        shutil.rmtree(archive_path)
    yield


@pytest.fixture
def client():
    """Return a FastAPI test client."""
    return TestClient(app)
