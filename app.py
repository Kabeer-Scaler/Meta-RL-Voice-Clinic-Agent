"""FastAPI application using OpenEnv framework."""

from src.voiceclinicagent.openenv_compat import create_fastapi_app
from src.voiceclinicagent.api_models import VoiceClinicAction, VoiceClinicObservation
from src.voiceclinicagent.env import VoiceClinicEnvironment
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

def build_app():
    """
    Create the FastAPI app across both old and new openenv-core variants.

    Older openenv_core versions expect an environment instance.
    Newer openenv/openenv-core versions expect the environment class/factory.
    """
    try:
        return create_fastapi_app(
            VoiceClinicEnvironment(),
            VoiceClinicAction,
            VoiceClinicObservation,
        )
    except TypeError as exc:
        # Newer openenv-core explicitly rejects instances and asks for a callable.
        if "env must be a callable" not in str(exc):
            raise
        return create_fastapi_app(
            VoiceClinicEnvironment,
            VoiceClinicAction,
            VoiceClinicObservation,
        )


app = build_app()

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve landing page at root
@app.get("/")
async def root():
    """Serve the landing page"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "VoiceClinicAgent API is running", "docs": "/docs", "health": "/health"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
