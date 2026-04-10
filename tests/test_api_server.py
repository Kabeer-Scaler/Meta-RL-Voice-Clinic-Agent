"""Integration tests for the explicit FastAPI server."""

from fastapi.testclient import TestClient

import app


client = TestClient(app.app)


def test_reset_returns_episode_id_and_observation():
    response = client.post("/reset", json={"task_id": "easy_001", "seed": 42})

    assert response.status_code == 200
    payload = response.json()
    assert payload["episode_id"]
    assert "observation" in payload
    assert payload["observation"]["task_level"] == "easy"


def test_step_and_state_round_trip():
    reset_response = client.post("/reset", json={"task_id": "easy_001", "seed": 42})
    episode_id = reset_response.json()["episode_id"]

    step_response = client.post(
        "/step",
        json={
            "episode_id": episode_id,
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

    state_response = client.get(f"/state/{episode_id}")

    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["episode_id"] == episode_id
    assert state_payload["task_id"] == "easy_001"
