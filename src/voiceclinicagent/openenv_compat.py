"""
Compatibility layer for openenv imports.

The openenv-core package structure differs between local and Docker environments:
- Local: provides both openenv.core and openenv_core
- Docker: only provides openenv_core

This module provides a consistent import interface.
"""

try:
    # Try the preferred import path first
    from openenv.core.env_server import Environment, Action, Observation, State, create_fastapi_app
except ModuleNotFoundError:
    # Fall back to the underscore version (Docker environment)
    from openenv_core.env_server import Environment, Action, Observation, State, create_fastapi_app

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

__all__ = [
    'Environment',
    'Action',
    'Observation',
    'State',
    'create_fastapi_app',
    'EnvClient',
    'StepResult',
]
