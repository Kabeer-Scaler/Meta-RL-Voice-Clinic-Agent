"""API request/response models using OpenEnv base classes."""

from typing import Dict, Any, Optional, Literal, List, Union
from .openenv_compat import Action, Observation, State
from pydantic import Field
from .models import ClinicalHistory


class VoiceClinicAction(Action):
    """Agent action with typed payload."""
    action_type: Literal[
        "ask_question",
        "offer_slot",
        "confirm_booking",
        "escalate_urgent",
        "escalate_human",
        "query_availability",
        "check_insurance",
        "trigger_reflection",
        "store_memory_safe",
        "recall_memory",
        "end_call",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        """Custom init to handle base Action class constraints."""
        # Extract our custom fields
        action_type = data.pop('action_type', None)
        payload = data.pop('payload', {})
        
        # Initialize base Action class with remaining data
        super().__init__(**data)
        
        # Set our custom fields after initialization
        if action_type is not None:
            object.__setattr__(self, 'action_type', action_type)
        object.__setattr__(self, 'payload', payload)


class PatientFlags(dict):
    """Flags indicating patient characteristics (as dict for JSON compatibility)."""
    pass


class ClinicState(dict):
    """Current state of the clinic (as dict for JSON compatibility)."""
    pass


class VoiceClinicObservation(Observation):
    """Structured observation returned to the agent.
    
    Note: done and reward are inherited from Observation base class.
    """
    task_level: Literal["easy", "medium", "hard"]
    turn_idx: int
    max_turns: int
    patient_message: str
    conversation_summary: str
    patient_flags: Dict[str, Union[int, float]]
    clinic_state: Dict[str, Any]
    reflection_token: List[float]
    memory_vault_summary: Dict[str, Union[int, float]]
    privacy_risk_mask: Dict[str, int]
    clinical_history: ClinicalHistory
    history_accessed_this_turn: bool = False


class VoiceClinicState(State):
    """Episode metadata.
    
    Note: episode_id and step_count are inherited from State base class.
    """
    task_id: str = ""
    max_turns: int = 20
    cumulative_reward: float = 0.0
    termination_reason: Optional[str] = None
    final_score: Optional[float] = None  # Always in [0.0, 1.0]
    booking_confirmed: bool = False
    escalated_urgent: bool = False
    escalated_human: bool = False
    grade_report: Optional[Dict[str, Any]] = None  # Full breakdown when done
