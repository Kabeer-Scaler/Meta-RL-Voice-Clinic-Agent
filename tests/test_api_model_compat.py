"""Compatibility tests for API models across openenv-core variants."""

from src.voiceclinicagent import api_models
from src.voiceclinicagent.models import ClinicalHistory


def test_voice_clinic_action_supports_legacy_action_base(monkeypatch):
    """Older Action bases only accept metadata in __init__."""

    def legacy_action_init(self, metadata=None):
        object.__setattr__(self, "metadata", metadata or {})

    monkeypatch.setattr(api_models.Action, "__init__", legacy_action_init, raising=True)

    action = api_models.VoiceClinicAction(
        action_type="ask_question",
        payload={"question_type": "symptoms"},
    )

    assert action.action_type == "ask_question"
    assert action.payload == {"question_type": "symptoms"}


def test_voice_clinic_observation_supports_legacy_observation_base(monkeypatch):
    """Older Observation bases only accept done/reward/metadata."""

    def legacy_observation_init(self, done=False, reward=None, metadata=None):
        object.__setattr__(self, "done", done)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "metadata", metadata or {})

    monkeypatch.setattr(api_models.Observation, "__init__", legacy_observation_init, raising=True)

    observation = api_models.VoiceClinicObservation(
        done=False,
        reward=None,
        task_level="easy",
        turn_idx=0,
        max_turns=10,
        patient_message="hello",
        conversation_summary="start",
        patient_flags={},
        clinic_state={},
        reflection_token=[0.0, 0.0, 0.0, 0.0],
        memory_vault_summary={},
        privacy_risk_mask={},
        clinical_history=ClinicalHistory(patient_id="p1", has_history=False),
        history_accessed_this_turn=False,
    )

    assert observation.done is False
    assert observation.task_level == "easy"
    assert observation.patient_message == "hello"


def test_voice_clinic_state_supports_legacy_state_base(monkeypatch):
    """Older State bases only accept episode_id/step_count/metadata."""

    def legacy_state_init(self, episode_id=None, step_count=0, metadata=None):
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "step_count", step_count)
        object.__setattr__(self, "metadata", metadata or {})

    monkeypatch.setattr(api_models.State, "__init__", legacy_state_init, raising=True)

    state = api_models.VoiceClinicState(
        episode_id="ep1",
        step_count=2,
        task_id="easy_001",
        max_turns=15,
        cumulative_reward=1.0,
    )

    assert state.episode_id == "ep1"
    assert state.step_count == 2
    assert state.task_id == "easy_001"
