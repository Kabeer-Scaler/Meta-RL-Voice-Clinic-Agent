"""FastAPI application using OpenEnv framework."""

from openenv.core.env_server import create_fastapi_app
from src.voiceclinicagent.api_models import VoiceClinicAction, VoiceClinicObservation
from src.voiceclinicagent.env import VoiceClinicEnvironment

# Create FastAPI app using OpenEnv framework
# This automatically creates all endpoints: /ws, /reset, /step, /state, /health, /web, /docs
app = create_fastapi_app(VoiceClinicEnvironment, VoiceClinicAction, VoiceClinicObservation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
