"""FastAPI application using a stable OpenEnv-backed HTTP surface."""

from typing import Optional

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
import os

from src.voiceclinicagent.openenv_compat import create_fastapi_app
from src.voiceclinicagent.api_models import VoiceClinicAction, VoiceClinicObservation
from src.voiceclinicagent.env import VoiceClinicEnvironment


class ResetRequest(BaseModel):
    """Stable reset payload accepted by the validator and local tests."""

    task_id: str = "easy_001"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    """Stable step payload that mirrors the OpenEnv action contract."""

    action: VoiceClinicAction
    timeout_s: Optional[float] = None


SHARED_ENV = VoiceClinicEnvironment()

# The compatibility shim handles both old "instance" and new "callable" APIs.
app = create_fastapi_app(
    SHARED_ENV,
    VoiceClinicAction,
    VoiceClinicObservation,
)


# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def _promote_route(path: str, method: str) -> None:
    """Move the most recently added matching route ahead of OpenEnv defaults."""
    for index in range(len(app.router.routes) - 1, -1, -1):
        route = app.router.routes[index]
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            app.router.routes.insert(0, app.router.routes.pop(index))
            break


@app.post("/reset")
async def reset_endpoint(payload: ResetRequest):
    """Reset the shared environment and expose the generated episode id."""
    observation = SHARED_ENV.reset(
        seed=payload.seed,
        episode_id=payload.episode_id,
        task_id=payload.task_id,
    )
    return {
        "episode_id": SHARED_ENV.state.episode_id,
        "observation": jsonable_encoder(observation),
        "reward": observation.reward,
        "done": observation.done,
    }


@app.post("/step")
async def step_endpoint(payload: StepRequest):
    """Advance the shared environment by one action."""
    observation = SHARED_ENV.step(payload.action, timeout_s=payload.timeout_s)
    return {
        "observation": jsonable_encoder(observation),
        "reward": observation.reward,
        "done": observation.done,
    }


@app.get("/state")
async def state_endpoint():
    """Return the full environment state, not just the OpenEnv base fields."""
    return jsonable_encoder(SHARED_ENV.state)


@app.get("/")
async def root():
    """Serve the landing page"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "VoiceClinicAgent API is running", "docs": "/docs", "health": "/health"}


@app.get("/robots.txt")
async def robots():
    """Return a tiny robots.txt so crawlers don't generate noisy 404s."""
    return PlainTextResponse("User-agent: *\nDisallow:\n")


_promote_route("/reset", "POST")
_promote_route("/step", "POST")
_promote_route("/state", "GET")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
