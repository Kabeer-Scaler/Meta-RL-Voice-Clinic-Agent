"""FastAPI application using OpenEnv framework."""

from src.voiceclinicagent.openenv_compat import create_fastapi_app
from src.voiceclinicagent.api_models import VoiceClinicAction, VoiceClinicObservation
from src.voiceclinicagent.env import VoiceClinicEnvironment
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Create environment factory function
def create_env():
    """Factory function to create environment instances"""
    return VoiceClinicEnvironment()

# Create FastAPI app using OpenEnv framework
# Pass the factory function instead of the class
app = create_fastapi_app(create_env, VoiceClinicAction, VoiceClinicObservation)

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
