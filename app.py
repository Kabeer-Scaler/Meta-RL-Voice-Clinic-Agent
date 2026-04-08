"""FastAPI application using OpenEnv framework."""

from src.voiceclinicagent.openenv_compat import create_fastapi_app
from src.voiceclinicagent.api_models import VoiceClinicAction, VoiceClinicObservation
from src.voiceclinicagent.env import VoiceClinicEnvironment
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Create an instance of the environment
# OpenEnv framework expects an instance, not a class
env_instance = VoiceClinicEnvironment()

# Create FastAPI app using OpenEnv framework
# Pass the instance instead of the class
app = create_fastapi_app(env_instance, VoiceClinicAction, VoiceClinicObservation)

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
