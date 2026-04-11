"""Regression tests for the OpenEnv compatibility shim."""

import src.voiceclinicagent.openenv_compat as compat


class DummyEnv:
    """Simple environment stand-in for compatibility tests."""


def test_create_fastapi_app_retries_with_shared_factory_when_library_requires_callable(monkeypatch):
    """Newer openenv-core variants should use a shared factory, not a fresh class."""

    calls = []
    env_instance = DummyEnv()

    def fake_create(env, action_model, observation_model, *args, **kwargs):
        calls.append(env)
        if len(calls) == 1:
            raise TypeError("env must be a callable (class or factory function)")
        return "ok"

    monkeypatch.setattr(compat, "_raw_create_fastapi_app", fake_create)

    result = compat.create_fastapi_app(env_instance, object, object)

    assert result == "ok"
    assert calls[0] is env_instance or callable(calls[0])
    assert callable(calls[1])
    assert calls[1]() is env_instance


def test_create_fastapi_app_retries_with_instance_when_library_requires_bound_method(monkeypatch):
    """Older openenv-core variants expect an instance and fail on a class."""

    calls = []

    def fake_create(env, action_model, observation_model, *args, **kwargs):
        calls.append(env)
        if len(calls) == 1:
            raise TypeError("VoiceClinicEnvironment.reset() missing 1 required positional argument: 'self'")
        return "ok"

    monkeypatch.setattr(compat, "_raw_create_fastapi_app", fake_create)

    result = compat.create_fastapi_app(DummyEnv, object, object)

    assert result == "ok"
    assert calls[0] is DummyEnv
    assert isinstance(calls[1], DummyEnv)
