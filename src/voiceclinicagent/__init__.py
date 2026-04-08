"""VoiceClinicAgent - OpenEnv-compatible RL environment for clinic appointment booking."""

__version__ = "0.1.0"

from .env import VoiceClinicEnvironment
from .api_models import VoiceClinicAction, VoiceClinicObservation, VoiceClinicState

# Client is optional - only import if EnvClient is available
try:
    from .client import VoiceClinicEnv
    __all__ = [
        "VoiceClinicEnvironment",
        "VoiceClinicAction",
        "VoiceClinicObservation",
        "VoiceClinicState",
        "VoiceClinicEnv",
    ]
except (ImportError, TypeError):
    # EnvClient not available (server-only deployment)
    __all__ = [
        "VoiceClinicEnvironment",
        "VoiceClinicAction",
        "VoiceClinicObservation",
        "VoiceClinicState",
    ]
