"""FastAPI application for VoiceClinicAgent.

This module serves a small, explicit HTTP API instead of relying on
openenv-core's server wrapper. That avoids version skew between the local
environment and Hugging Face Spaces while keeping the API surface compatible
with the endpoints the project already uses:

- GET  /health
- POST /reset
- POST /step
- GET  /state/{episode_id}
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.voiceclinicagent.api_models import VoiceClinicAction
from src.voiceclinicagent.env import VoiceClinicEnvironment


app = FastAPI(title="VoiceClinicAgent")
_EPISODES: Dict[str, VoiceClinicEnvironment] = {}
_EPISODES_LOCK = Lock()


class ResetRequest(BaseModel):
    """Request body for starting a new episode."""

    task_id: str = "easy_001"
    seed: Optional[int] = None


class StepActionRequest(BaseModel):
    """Action payload accepted by the step endpoint."""

    action_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    """Request body for advancing an episode by one action."""

    episode_id: str
    action: Optional[StepActionRequest] = None
    action_type: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    def to_action(self) -> VoiceClinicAction:
        """Normalize both supported request shapes into a VoiceClinicAction."""
        if self.action is not None:
            return VoiceClinicAction(
                action_type=self.action.action_type,
                payload=self.action.payload,
            )
        if self.action_type is not None:
            return VoiceClinicAction(
                action_type=self.action_type,
                payload=self.payload,
            )
        raise HTTPException(status_code=422, detail="Missing action payload")


def _serialize_observation(observation) -> Dict[str, Any]:
    """Convert a VoiceClinicObservation into plain JSON data."""
    return {
        "done": getattr(observation, "done", False),
        "reward": getattr(observation, "reward", None),
        "task_level": getattr(observation, "task_level", "easy"),
        "turn_idx": getattr(observation, "turn_idx", 0),
        "max_turns": getattr(observation, "max_turns", 20),
        "patient_message": getattr(observation, "patient_message", ""),
        "conversation_summary": getattr(observation, "conversation_summary", ""),
        "patient_flags": jsonable_encoder(getattr(observation, "patient_flags", {})),
        "clinic_state": jsonable_encoder(getattr(observation, "clinic_state", {})),
        "clinical_history": jsonable_encoder(getattr(observation, "clinical_history", {})),
        "reflection_token": jsonable_encoder(getattr(observation, "reflection_token", [0.0, 0.0, 0.0, 0.0])),
        "memory_vault_summary": jsonable_encoder(getattr(observation, "memory_vault_summary", {})),
        "privacy_risk_mask": jsonable_encoder(getattr(observation, "privacy_risk_mask", {})),
        "history_accessed_this_turn": getattr(observation, "history_accessed_this_turn", False),
    }


def _serialize_state(state) -> Dict[str, Any]:
    """Convert a VoiceClinicState into plain JSON data."""
    done = bool(
        getattr(state, "final_score", None) is not None
        or getattr(state, "termination_reason", None)
        or getattr(state, "booking_confirmed", False)
        or getattr(state, "escalated_urgent", False)
        or getattr(state, "escalated_human", False)
    )
    return {
        "episode_id": getattr(state, "episode_id", None),
        "step_count": getattr(state, "step_count", 0),
        "task_id": getattr(state, "task_id", ""),
        "max_turns": getattr(state, "max_turns", 20),
        "done": done,
        "truncated": False,
        "cumulative_reward": getattr(state, "cumulative_reward", 0.0),
        "termination_reason": getattr(state, "termination_reason", None),
        "final_score": getattr(state, "final_score", None),
        "booking_confirmed": getattr(state, "booking_confirmed", False),
        "escalated_urgent": getattr(state, "escalated_urgent", False),
        "escalated_human": getattr(state, "escalated_human", False),
        "grade_report": jsonable_encoder(getattr(state, "grade_report", None)),
    }


def _get_env(episode_id: str) -> VoiceClinicEnvironment:
    """Fetch a live episode environment or raise 404."""
    with _EPISODES_LOCK:
        env = _EPISODES.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown episode_id: {episode_id}")
    return env


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint used by Spaces and validators."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> Dict[str, Any]:
    """Create a fresh episode environment and return the initial observation."""
    env = VoiceClinicEnvironment()
    observation = env.reset(seed=request.seed, task_id=request.task_id)
    state = env.state
    episode_id = getattr(state, "episode_id", None)
    if not episode_id:
        raise HTTPException(status_code=500, detail="Environment did not produce an episode_id")
    with _EPISODES_LOCK:
        _EPISODES[episode_id] = env
    return {
        "episode_id": episode_id,
        "observation": _serialize_observation(observation),
        "reward": getattr(observation, "reward", None),
        "done": getattr(observation, "done", False),
    }


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """Advance an existing episode by one action."""
    env = _get_env(request.episode_id)
    observation = env.step(request.to_action())
    return {
        "episode_id": request.episode_id,
        "observation": _serialize_observation(observation),
        "reward": getattr(observation, "reward", None),
        "done": getattr(observation, "done", False),
        "truncated": False,
        "info": {},
    }


@app.get("/state/{episode_id}")
async def state(episode_id: str) -> Dict[str, Any]:
    """Return episode state metadata, including final score when available."""
    env = _get_env(episode_id)
    return _serialize_state(env.state)


# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Serve the landing page."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "VoiceClinicAgent API is running", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
