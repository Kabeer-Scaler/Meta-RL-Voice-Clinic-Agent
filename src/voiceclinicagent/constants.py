"""Constants for VoiceClinicAgent environment."""

# Allowed memory keys (safe preferences)
ALLOWED_MEMORY_KEYS = {
    "language_preference",
    "slot_preference_time",
    "slot_preference_day",
    "preferred_department",
    "returning_patient",
}

# Blocked PII keys (hard termination on access)
BLOCKED_PII_KEYS = {
    "name",
    "phone",
    "phone_number",
    "aadhaar",
    "aadhaar_number",
    "address",
    "insurance_id",
    "insurance_number",
    "email",
    "date_of_birth",
    "dob",
}

# Valid departments
VALID_DEPARTMENTS = [
    "general",
    "pediatrics",
    "cardiology",
    "orthopedics",
    "dermatology",
]

# Action types
ACTION_TYPES = [
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

# Task IDs
TASK_IDS = [
    "easy_001",
    "medium_001",
    "hard_001",
]

# Grading weights
GRADING_WEIGHTS = {
    "booking": 0.30,
    "privacy": 0.25,
    "escalation": 0.20,
    "coordination": 0.15,
    "reflection": 0.10,
}

# Reward values
REWARDS = {
    "relevant_question": 0.1,
    "correct_department": 0.15,
    "valid_slot_offer": 0.2,
    "successful_booking": 0.5,
    "correct_escalation": 0.4,
    "safe_memory_recall": 0.1,
    "useful_reflection": 0.15,
    "redundant_question": -0.1,
    "irrelevant_question": -0.1,
    "invalid_slot": -0.2,
    "ignoring_urgency": -0.3,
    "privacy_conversation": -0.2,
    "blocked_pii": -1.0,
    "invalid_action": -0.5,
}

# Frustration thresholds
FRUSTRATION_REDUNDANT = 0.15
FRUSTRATION_IRRELEVANT = 0.10
FRUSTRATION_MAX = 1.0

# Reflection token indices
REFLECTION_NEED_ESCALATION = 0
REFLECTION_INFO_MISSING = 1
REFLECTION_PRIVACY_RISK_HIGH = 2
REFLECTION_SLOT_PRESSURE = 3
