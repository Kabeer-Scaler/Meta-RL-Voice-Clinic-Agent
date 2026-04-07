"""Core Pydantic models for VoiceClinicAgent."""

from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator


class ClinicalHistory(BaseModel):
    """
    Confidential clinical history summary.
    
    Follows minimum-necessary principle: provides essential medical context
    without exposing full raw chart data.
    """
    patient_id: str
    has_history: bool = Field(description="False for new patients, True for returning patients")
    
    # Chronic conditions
    chronic_conditions: List[str] = Field(default_factory=list, description="e.g., ['diabetes', 'hypertension']")
    
    # Allergies
    allergies: List[str] = Field(default_factory=list, description="e.g., ['penicillin']")
    
    # Prior visits (last 6 months)
    prior_visits: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent visits with date, department, reason"
    )
    
    # Prior escalations
    prior_escalations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prior urgent/ER visits"
    )
    
    # Preferences from history
    language_preference: Optional[str] = None
    slot_preference_time: Optional[str] = None  # "morning", "evening"
    preferred_department: Optional[str] = None
    
    # Problem list
    active_problems: List[str] = Field(default_factory=list, description="Current active diagnoses")
    
    # Current medications (names only, no dosages)
    current_medications: List[str] = Field(default_factory=list, description="e.g., ['metformin', 'lisinopril']")
    
    # Risk flags
    high_risk_flag: bool = Field(default=False, description="True if chronic conditions + age > 60")
    frequent_visitor: bool = Field(default=False, description="True if > 3 visits in 6 months")


class PatientFlags(BaseModel):
    """Flags indicating patient characteristics."""
    parent_booking: int = Field(ge=0, le=1, description="1 if parent booking for child, 0 otherwise")
    language_mix: float = Field(ge=0.0, le=1.0, description="Proportion of Hindi in responses")
    vague_symptoms: int = Field(ge=0, le=1, description="1 if symptoms are vague, 0 otherwise")
    urgency_score_hint: float = Field(ge=0.0, le=1.0, description="Hint about urgency level")
    insurance_status_known: int = Field(ge=0, le=1, description="1 if insurance status is known, 0 otherwise")
    duplicate_info_flag: int = Field(ge=0, le=1, description="1 if duplicate booking detected, 0 otherwise")


class ClinicState(BaseModel):
    """Current state of the clinic."""
    available_slots: Dict[str, List[str]] = Field(description="Department -> list of slot_ids")
    doctor_delay_minutes: int = Field(ge=0, description="Current doctor delay in minutes")
    urgent_queue_length: int = Field(ge=0, description="Number of urgent cases in queue")


class Observation(BaseModel):
    """Structured observation returned to the agent."""
    task_level: Literal["easy", "medium", "hard"]
    turn_idx: int = Field(ge=0)
    max_turns: int = Field(gt=0)
    patient_message: str = Field(min_length=1)
    conversation_summary: str
    patient_flags: PatientFlags
    clinic_state: ClinicState
    
    # NEW: Clinical history summary
    clinical_history: ClinicalHistory
    
    reflection_token: List[float] = Field(min_length=4, max_length=4)
    memory_vault_summary: Dict[str, Union[int, float]]
    privacy_risk_mask: Dict[str, int]
    
    # NEW: History access tracking (for grading)
    history_accessed_this_turn: bool = Field(default=False, description="True if agent accessed history this turn")

    @field_validator("reflection_token")
    @classmethod
    def validate_reflection_token(cls, v):
        if len(v) != 4:
            raise ValueError("reflection_token must have exactly 4 elements")
        for val in v:
            if not (0.0 <= val <= 1.0):
                raise ValueError("All reflection_token values must be in [0.0, 1.0]")
        return v


class PatientProfile(BaseModel):
    """Patient profile from scenario."""
    symptoms: str
    symptoms_severity: float = Field(ge=0.0, le=1.0)
    duration_days: int = Field(ge=0)
    pain_level: float = Field(ge=0.0, le=1.0)
    urgency_level: float = Field(ge=0.0, le=1.0)
    language_mix: float = Field(ge=0.0, le=1.0)
    vagueness_level: float = Field(ge=0.0, le=1.0)
    parent_booking: bool = False
    preferences: Dict[str, Any] = Field(default_factory=dict)
    hidden_facts: Dict[str, Any] = Field(default_factory=dict)


class ClinicConfig(BaseModel):
    """Clinic configuration from scenario."""
    departments: List[str]
    initial_slots: Dict[str, List[Dict[str, str]]]  # dept -> [{slot_id, datetime, doctor}]
    doctor_delays: Dict[int, int] = Field(default_factory=dict)  # turn -> delay_minutes


class GroundTruth(BaseModel):
    """Ground truth for grading."""
    correct_department: str
    urgency_level: float = Field(ge=0.0, le=1.0)
    slot_preferences: Dict[str, Any] = Field(default_factory=dict)
    expected_outcome: str  # "booking_confirmed", "escalated_urgent", etc.


class TimedEvent(BaseModel):
    """Timed event that occurs at a specific turn."""
    turn: int = Field(ge=0)
    event_type: str  # "doctor_delay", "urgent_consume_slot", etc.
    params: Dict[str, Any] = Field(default_factory=dict)


class SuccessCriteria(BaseModel):
    """Success criteria for the scenario."""
    min_info_collected: List[str] = Field(default_factory=list)
    max_turns_for_perfect: int = Field(gt=0)
    acceptable_outcomes: List[str] = Field(default_factory=list)


class Scenario(BaseModel):
    """Complete scenario definition."""
    task_id: str
    task_level: Literal["easy", "medium", "hard"]
    patient_profile: PatientProfile
    clinical_history: Optional[ClinicalHistory] = None  # NEW: Clinical history for Phase 3.5
    hidden_facts: Dict[str, Any] = Field(default_factory=dict)  # NEW: Hidden facts including deterioration
    clinic_config: ClinicConfig
    ground_truth: GroundTruth
    max_turns: int = Field(gt=0, le=50)
    success_criteria: SuccessCriteria
    timed_events: List[TimedEvent] = Field(default_factory=list)
    acceptable_departments: List[str] = Field(default_factory=list)
    acceptable_outcomes: List[str] = Field(default_factory=list)
    required_info_to_collect: List[str] = Field(default_factory=list)
    forbidden_questions: List[str] = Field(default_factory=list)
    initial_clinic_state: ClinicState


class GradeReport(BaseModel):
    """Final grading report."""
    final_score: float = Field(ge=0.0, le=1.0)
    booking_score: float = Field(ge=0.0, le=1.0)
    privacy_score: float = Field(ge=0.0, le=1.0)
    escalation_score: float = Field(ge=0.0, le=1.0)
    coordination_score: float = Field(ge=0.0, le=1.0)
    reflection_score: float = Field(ge=0.0, le=1.0)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    violations: List[str] = Field(default_factory=list)


class EpisodeState(BaseModel):
    """Internal episode state."""
    episode_id: str
    task_id: str
    turn_idx: int = Field(ge=0)
    max_turns: int = Field(gt=0)
    done: bool = False
    truncated: bool = False
    cumulative_reward: float = 0.0
    termination_reason: Optional[str] = None
    final_score: Optional[float] = None
    grade_report: Optional[GradeReport] = None
    # Internal state (not exposed via API)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    observation_history: List[Observation] = Field(default_factory=list)
    patient_frustration: float = Field(ge=0.0, le=1.0, default=0.0)
    booking_confirmed: bool = False
    escalated_urgent: bool = False
    escalated_human: bool = False
    escalation_turn: Optional[int] = None
    booked_department: Optional[str] = None
    checked_duplicates: bool = False
    offered_alternative: bool = False
    plan_revised_after_reflection: bool = False
    privacy_violations: List[str] = Field(default_factory=list)
