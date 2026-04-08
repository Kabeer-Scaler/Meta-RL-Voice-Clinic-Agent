"""FastAPI application using OpenEnv framework."""

from src.voiceclinicagent.openenv_compat import create_fastapi_app
from src.voiceclinicagent.api_models import VoiceClinicAction, VoiceClinicObservation
from src.voiceclinicagent.env import VoiceClinicEnvironment
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Create FastAPI app using OpenEnv framework
# This automatically creates all endpoints: /ws, /reset, /step, /state, /health, /web, /docs
app = create_fastapi_app(VoiceClinicEnvironment, VoiceClinicAction, VoiceClinicObservation)

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
