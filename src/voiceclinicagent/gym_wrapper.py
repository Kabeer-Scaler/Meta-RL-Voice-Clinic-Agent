"""
Gymnasium wrapper for VoiceClinicEnvironment.

Converts the Pydantic-based environment into a standard Gym interface
so any RL library (stable-baselines3, RLlib, etc.) can train on it.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from .env import VoiceClinicEnvironment
from .api_models import VoiceClinicAction

# All valid action types (must match ActionParser.REQUIRED_FIELDS)
ACTION_TYPES = [
    "ask_question",
    "query_availability",
    "offer_slot",
    "confirm_booking",
    "escalate_urgent",
    "escalate_human",
    "store_memory_safe",
    "recall_memory",
    "trigger_reflection",
    "end_call",
]

# Fixed payloads per action type so the agent only picks WHAT to do, not HOW
# The environment handles the details
FIXED_PAYLOADS = {
    "ask_question": {
        "question_type": "symptom_inquiry",
        "question_text": "Can you describe your symptoms in more detail?"
    },
    "query_availability": {"department": "general"},
    "offer_slot": {"slot_id": "SLOT_AUTO", "reason": "Best available slot"},
    "confirm_booking": {"slot_id": "SLOT_AUTO", "patient_confirmation": True},
    "escalate_urgent": {
        "urgency_reason": "Patient shows urgent symptoms",
        "symptoms": "Worsening condition detected"
    },
    "escalate_human": {"reason": "Complex case requiring human judgment"},
    "store_memory_safe": {"memory_key": "preferred_time", "value": "evening"},
    "recall_memory": {"memory_key": "preferred_time"},
    "trigger_reflection": {},
    "end_call": {"reason": "Call completed"},
}


class VoiceClinicGymEnv(gym.Env):
    """
    Standard Gymnasium wrapper around VoiceClinicEnvironment.

    Observation space: flat float32 vector encoding all relevant state.
    Action space:      Discrete(10) - one integer per action type.
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_dir: str = "scenarios", task_id: str = "easy_001"):
        super().__init__()
        self.task_id = task_id
        self._env = VoiceClinicEnvironment(scenario_dir)

        # --- Action space ---
        # Discrete: agent picks an integer 0-9, we map to action type
        self.action_space = spaces.Discrete(len(ACTION_TYPES))

        # --- Observation space ---
        # We encode the observation as a fixed-length float32 vector.
        # See _encode_obs() for what each dimension means.
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        obs = self._env.reset(task_id=self.task_id, seed=seed)
        return self._encode_obs(obs), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_type = ACTION_TYPES[action]
        payload = self._resolve_payload(action_type)

        clinic_action = VoiceClinicAction(action_type=action_type, payload=payload)
        obs = self._env.step(clinic_action)

        reward = float(obs.reward) if obs.reward is not None else 0.0
        terminated = bool(obs.done)
        truncated = False

        return self._encode_obs(obs), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _obs_dim(self) -> int:
        """Total number of features in the observation vector."""
        return (
            4   # turn progress, urgency hint, frustration proxy, queue pressure
            + 8  # symptom category flags
            + 5  # risk factor flags
            + 4  # reflection token
            + 4  # privacy risk mask
            + 3  # clinic state (delay, queue, slot count)
        )  # = 28

    def _encode_obs(self, obs) -> np.ndarray:
        """
        Encode a VoiceClinicObservation into a flat float32 vector.

        Every dimension is normalised to roughly [-1, 1] or [0, 1].
        The agent never sees raw text or disease names - only numeric signals.
        """
        vec = np.zeros(self._obs_dim(), dtype=np.float32)
        i = 0

        # --- Turn progress (0 → 1 as episode progresses) ---
        vec[i] = obs.turn_idx / max(obs.max_turns, 1);  i += 1

        # --- Urgency hint ---
        vec[i] = obs.patient_flags.get("urgency_score_hint", 0.5);  i += 1

        # --- Language mix (proxy for communication difficulty) ---
        vec[i] = obs.patient_flags.get("language_mix", 0.0);  i += 1

        # --- Urgent queue pressure ---
        queue = obs.clinic_state.get("urgent_queue_length", 0)
        vec[i] = min(queue / 10.0, 1.0);  i += 1

        # --- Symptom category flags (from patient message) ---
        msg = obs.patient_message.lower()
        symptom_keywords = {
            "respiratory": ["breath", "wheez", "cough", "chest tight"],
            "cardiac":     ["chest pain", "palpitat", "sweat", "heart"],
            "neurological":["headache", "dizz", "confus", "numb", "vision"],
            "gastro":      ["nausea", "vomit", "abdom", "diarr"],
            "dermatology": ["rash", "itch", "skin", "swell"],
            "infectious":  ["fever", "chill", "fatigue", "infect"],
            "pediatric":   ["child", "baby", "infant", "kid"],
            "deteriorating":["worse", "worsening", "getting worse", "more severe"],
        }
        for keywords in symptom_keywords.values():
            vec[i] = 1.0 if any(k in msg for k in keywords) else 0.0
            i += 1

        # --- Risk factor flags (from clinical history) ---
        hist = obs.clinical_history
        vec[i] = 1.0 if hist.high_risk_flag else 0.0;                    i += 1
        vec[i] = 1.0 if hist.prior_escalations else 0.0;                 i += 1
        vec[i] = 1.0 if hist.allergies else 0.0;                         i += 1
        vec[i] = min(len(hist.chronic_conditions) / 5.0, 1.0);           i += 1
        vec[i] = 1.0 if hist.frequent_visitor else 0.0;                  i += 1

        # --- Reflection token (4 floats, already in [0,1]) ---
        for v in obs.reflection_token:
            vec[i] = float(v);  i += 1

        # --- Privacy risk mask ---
        for v in obs.privacy_risk_mask.values():
            vec[i] = float(v);  i += 1

        # --- Clinic state ---
        delay = obs.clinic_state.get("doctor_delay_minutes", 0)
        vec[i] = min(delay / 60.0, 1.0);  i += 1

        vec[i] = min(queue / 10.0, 1.0);  i += 1  # queue again (explicit)

        total_slots = sum(
            len(v) for v in obs.clinic_state.get("available_slots", {}).values()
        )
        vec[i] = min(total_slots / 10.0, 1.0);  i += 1

        return vec

    # ------------------------------------------------------------------
    # Payload resolution
    # ------------------------------------------------------------------

    def _resolve_payload(self, action_type: str) -> Dict[str, Any]:
        """
        Build a valid payload for the chosen action type.
        For slot-based actions, pick the first available slot dynamically.
        """
        payload = dict(FIXED_PAYLOADS[action_type])

        if action_type in ("offer_slot", "confirm_booking"):
            # Pick first available slot from current clinic state
            available = self._env._available_slots
            for dept_slots in available.values():
                if dept_slots:
                    payload["slot_id"] = dept_slots[0]
                    break

        return payload
