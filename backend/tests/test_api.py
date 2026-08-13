from fastapi.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_list_items(client: TestClient):
    session_id = "test-session"
    payload = {
        "type": "user_input",
        "content": "hello",
        "source": "user",
        "scope": "current_session",
    }

    create_response = client.post(f"/context/items?session_id={session_id}", json=payload)
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["content"] == "hello"
    assert created["type"] == "user_input"

    list_response = client.get(f"/context/items?session_id={session_id}")
    assert list_response.status_code == 200
    items = list_response.json()
    assert len(items) == 1
    assert items[0]["content"] == "hello"


def test_compose_window_sliding(client: TestClient):
    session_id = "compose-session"
    for i in range(5):
        client.post(
            f"/context/items?session_id={session_id}",
            json={
                "type": "user_input",
                "content": f"message {i}",
                "source": "user",
                "scope": "current_session",
            },
        )

    response = client.post(
        "/context/windows/compose",
        json={
            "session_id": session_id,
            "strategy": "sliding",
            "max_tokens": 100,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["session_id"] == session_id
    assert result["strategy"] == "sliding"
    assert result["item_count"] > 0
    assert result["total_tokens"] <= 100
    assert "prompt_fragment" in result


def test_list_layers(client: TestClient):
    response = client.get("/context/layers")
    assert response.status_code == 200
    layers = response.json()
    assert len(layers) >= 4
    assert any(layer["name"] == "working" for layer in layers)


def test_get_metrics_after_compose(client: TestClient):
    session_id = "metrics-session"
    client.post(
        f"/context/items?session_id={session_id}",
        json={
            "type": "user_input",
            "content": "hello",
            "source": "user",
            "scope": "current_session",
        },
    )
    client.post(
        "/context/windows/compose",
        json={
            "session_id": session_id,
            "strategy": "sliding",
            "max_tokens": 100,
        },
    )

    response = client.get(f"/metrics/{session_id}")
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["session_id"] == session_id
    assert metrics["retrieved_count"] == 1
    assert metrics["window_tokens"] <= 100
