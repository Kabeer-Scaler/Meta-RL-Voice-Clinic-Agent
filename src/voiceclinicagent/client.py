"""OpenEnv client for VoiceClinicAgent."""

from openenv_core.env_client import EnvClient
from openenv_core.client_types import StepResult
from .api_models import VoiceClinicAction, VoiceClinicObservation, VoiceClinicState


class VoiceClinicEnv(EnvClient[VoiceClinicAction, VoiceClinicObservation, VoiceClinicState]):
    """
    Client for interacting with VoiceClinicAgent environment.
    
    Usage:
        with VoiceClinicEnv(base_url="http://localhost:7860").sync() as env:
            result = env.reset(task_id="easy_001")
            result = env.step(VoiceClinicAction(
                action_type="ask_question",
                payload={"question_type": "symptoms", "question_text": "What symptoms?"}
            ))
            print(result.observation.patient_message)
    """
    
    def _step_payload(self, action: VoiceClinicAction) -> dict:
        """Convert action to JSON payload for step request."""
        return {
            "action_type": action.action_type,
            "payload": action.payload,
        }
    
    def _parse_result(self, payload: dict) -> StepResult:
        """Parse step response into StepResult."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=VoiceClinicObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                task_level=obs_data.get("task_level", "easy"),
                turn_idx=obs_data.get("turn_idx", 0),
                max_turns=obs_data.get("max_turns", 20),
                patient_message=obs_data.get("patient_message", ""),
                conversation_summary=obs_data.get("conversation_summary", ""),
                patient_flags=obs_data.get("patient_flags", {}),
                clinic_state=obs_data.get("clinic_state", {}),
                clinical_history=obs_data.get("clinical_history", {}),
                reflection_token=obs_data.get("reflection_token", [0.0, 0.0, 0.0, 0.0]),
                memory_vault_summary=obs_data.get("memory_vault_summary", {}),
                privacy_risk_mask=obs_data.get("privacy_risk_mask", {}),
                history_accessed_this_turn=obs_data.get("history_accessed_this_turn", False),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: dict) -> VoiceClinicState:
        """Parse state response into VoiceClinicState."""
        return VoiceClinicState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            max_turns=payload.get("max_turns", 20),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            termination_reason=payload.get("termination_reason"),
            final_score=payload.get("final_score"),
            booking_confirmed=payload.get("booking_confirmed", False),
            escalated_urgent=payload.get("escalated_urgent", False),
            escalated_human=payload.get("escalated_human", False),
        )
