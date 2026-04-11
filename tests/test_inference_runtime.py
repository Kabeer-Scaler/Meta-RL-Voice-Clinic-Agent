"""Focused tests for inference runtime helpers."""

import inference


def test_verify_proxy_call_returns_false_when_proxy_fails(monkeypatch):
    """Proxy failure should be logged and reported, not raised."""

    class FailingChatCompletions:
        def create(self, **kwargs):
            raise RuntimeError("proxy down")

    class FailingChat:
        completions = FailingChatCompletions()

    class FailingClient:
        chat = FailingChat()

    monkeypatch.setattr(inference, "_build_openai_client", lambda api_key, base_url: FailingClient())

    config = inference.RuntimeConfig(
        api_base_url="https://validator.example/v1",
        api_key="secret-key",
        model_name="gpt-4.1-mini",
        env_base_url="http://localhost:7860",
    )

    assert inference.verify_proxy_call(config, attempts=2) is False


def test_verify_proxy_call_returns_true_when_call_succeeds_without_text(monkeypatch):
    """A successful API call should count even if the SDK payload is sparse."""

    class SparseResponse:
        choices = []

    class ChatCompletions:
        def create(self, **kwargs):
            return SparseResponse()

    class Chat:
        completions = ChatCompletions()

    class Client:
        chat = Chat()

    monkeypatch.setattr(inference, "_build_openai_client", lambda api_key, base_url: Client())

    config = inference.RuntimeConfig(
        api_base_url="https://validator.example/v1",
        api_key="secret-key",
        model_name="gpt-4.1-mini",
        env_base_url="http://localhost:7860",
    )

    assert inference.verify_proxy_call(config, attempts=1) is True


def test_wait_for_environment_returns_true_after_retry(monkeypatch):
    """Environment wait should tolerate transient startup failures."""

    attempts = {"count": 0}

    class Response:
        status_code = 200

    def fake_get(url, timeout):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("starting")
        return Response()

    monkeypatch.setattr(inference.requests, "get", fake_get)

    assert inference.wait_for_environment("http://localhost:7860", attempts=3, timeout_s=1) is True


def test_run_episode_falls_back_to_direct_env_when_openenv_client_fails(monkeypatch):
    """If the client/server path breaks, inference should still complete via direct env."""

    class BrokenClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def sync(self):
            return self

        def __enter__(self):
            raise RuntimeError("socket upgrade failed")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(inference, "CLIENT_AVAILABLE", True)
    monkeypatch.setattr(inference, "VoiceClinicEnv", BrokenClient)
    monkeypatch.setattr(
        inference,
        "_run_episode_direct",
        lambda task_id, model_name, agent: (0.75, True, 2, [0.25, 0.5]),
    )

    score = inference.run_episode(
        task_id="easy_001",
        env_base="http://localhost:7860",
        api_base="https://validator.example/v1",
        api_key="secret-key",
        model_name="gpt-4.1-mini",
        agent_type="rule-based",
    )

    assert score == 0.75
