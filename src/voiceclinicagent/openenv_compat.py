"""
Compatibility layer for openenv imports.

The openenv-core package structure differs between local and Docker environments:
- Local: provides both openenv.core and openenv_core
- Docker: only provides openenv_core

This module provides a consistent import interface.
"""

try:
    # Try the preferred import path first
    from openenv.core.env_server import Environment, Action, Observation, State, create_fastapi_app as _raw_create_fastapi_app
    _OPENENV_RUNTIME = "modern"
except ModuleNotFoundError:
    # Fall back to the underscore version (Docker environment)
    from openenv_core.env_server import Environment, Action, Observation, State, create_fastapi_app as _raw_create_fastapi_app
    _OPENENV_RUNTIME = "legacy"

# Client imports - these may not be available in all versions
try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
except (ModuleNotFoundError, ImportError):
    try:
        from openenv_core.env_client import EnvClient
        from openenv_core.client_types import StepResult
    except (ModuleNotFoundError, ImportError):
        # Client modules not available - define placeholders
        EnvClient = None
        StepResult = None

def create_fastapi_app(env, action_model, observation_model, *args, **kwargs):
    """
    Create a FastAPI app across old and new openenv-core variants.

    Some versions expect an environment instance, while newer variants expect
    a factory callable. For newer builds, returning one shared instance from a
    factory preserves episode state across HTTP reset/step/state requests.
    """
    shared_factory = None
    candidate = env

    if _OPENENV_RUNTIME == "modern" and not callable(env):
        shared_factory = lambda env=env: env
        candidate = shared_factory

    try:
        return _raw_create_fastapi_app(candidate, action_model, observation_model, *args, **kwargs)
    except TypeError as exc:
        message = str(exc)
        if "env must be a callable" in message:
            if shared_factory is None and not callable(env):
                shared_factory = lambda env=env: env
            if shared_factory is not None:
                return _raw_create_fastapi_app(shared_factory, action_model, observation_model, *args, **kwargs)
        if "missing 1 required positional argument: 'self'" in message and callable(env):
            return _raw_create_fastapi_app(env(), action_model, observation_model, *args, **kwargs)
        raise


__all__ = [
    "Environment",
    "Action",
    "Observation",
    "State",
    "create_fastapi_app",
    "EnvClient",
    "StepResult",
]
