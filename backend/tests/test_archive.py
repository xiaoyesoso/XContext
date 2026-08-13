import asyncio
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.models import ContextAuthority, ContextItem, ContextScope, ContextSource, ContextType
from app.repositories.archive import LocalArchiveRepository


def _make_item(content: str) -> ContextItem:
    return ContextItem(
        type=ContextType.USER_INPUT,
        content=content,
        source=ContextSource.USER,
        scope=ContextScope.CURRENT_SESSION,
        authority=ContextAuthority.CONFIRMED,
    )


@pytest.fixture
def archive_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LocalArchiveRepository(base_path=tmpdir)


def test_archive_add_and_get(archive_repo):
    item = _make_item("hello")
    asyncio.run(archive_repo.add("session-1", item))
    fetched = asyncio.run(archive_repo.get("session-1", item.id))

    assert fetched is not None
    assert fetched.content == "hello"


def test_archive_list_and_delete(archive_repo):
    item1 = _make_item("one")
    item2 = _make_item("two")
    asyncio.run(archive_repo.add("session-1", item1))
    asyncio.run(archive_repo.add("session-1", item2))

    items = asyncio.run(archive_repo.list("session-1"))
    assert len(items) == 2

    deleted = asyncio.run(archive_repo.delete("session-1", item1.id))
    assert deleted is True
    items = asyncio.run(archive_repo.list("session-1"))
    assert len(items) == 1


def test_archive_api(client: TestClient):
    session_id = "archive-api-session"
    create_response = client.post(
        f"/context/items?session_id={session_id}",
        json={
            "type": "user_input",
            "content": "to be archived",
            "source": "user",
            "scope": "current_session",
        },
    )
    item_id = create_response.json()["id"]

    archive_response = client.post(f"/context/archive/{item_id}?session_id={session_id}")
    assert archive_response.status_code == 200
    assert archive_response.json()["content"] == "to be archived"

    list_response = client.get(f"/context/archive?session_id={session_id}")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1
