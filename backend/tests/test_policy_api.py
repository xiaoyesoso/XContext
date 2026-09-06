"""Integration tests for policy endpoints."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_list_policies_and_unknown_action_validation():
    response = client.get("/policies")
    assert response.status_code == 200
    assert any(p["policy_id"] == "default" for p in response.json())

    response = client.post(
        "/policies",
        json={
            "policy_id": "api-policy",
            "name": "API policy",
            "scenario": "api-test",
            "segments": [{"action_name": "missing-action"}],
        },
    )
    assert response.status_code == 422


def test_policy_crud_and_execution_observation():
    payload = {
        "policy_id": "api-policy-valid",
        "name": "API policy valid",
        "scenario": "api-test-valid",
        "segments": [{"action_name": "dialogue_history", "priority": 8}],
    }
    response = client.post("/policies", json=payload)
    assert response.status_code == 201

    response = client.get("/policies/api-policy-valid")
    assert response.status_code == 200
    assert response.json()["scenario"] == "api-test-valid"

    response = client.put(
        "/policies/api-policy-valid",
        json={**payload, "name": "Updated policy"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Updated policy"

    response = client.post("/policies/api-policy-valid/default")
    assert response.status_code == 200
    assert response.json()["default"] is True

    response = client.delete("/policies/api-policy-valid")
    assert response.status_code == 200

    response = client.get("/policies/execution/no-such-session")
    assert response.status_code == 404
