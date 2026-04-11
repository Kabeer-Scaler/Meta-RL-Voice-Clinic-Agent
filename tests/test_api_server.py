"""Integration tests for the OpenEnv-backed FastAPI server."""

from fastapi.testclient import TestClient

import app


client = TestClient(app.app)


def test_reset_returns_observation():
    response = client.post("/reset", json={"task_id": "easy_001", "seed": 42})

    assert response.status_code == 200
    payload = response.json()
    assert payload["episode_id"]
    assert "observation" in payload
    assert payload["observation"]["task_level"] == "easy"
    assert payload["done"] is False


def test_step_and_state_round_trip():
    reset_response = client.post("/reset", json={"task_id": "easy_001", "seed": 42})
    assert reset_response.status_code == 200

    step_response = client.post(
        "/step",
        json={
            "action": {
                "action_type": "ask_question",
                "payload": {
                    "question_type": "symptoms",
                    "question_text": "What symptoms are you experiencing?",
                },
            },
        },
    )

    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert "observation" in step_payload
    assert "reward" in step_payload

    state_response = client.get("/state")

    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert "episode_id" in state_payload
    assert state_payload["task_id"] == "easy_001"
