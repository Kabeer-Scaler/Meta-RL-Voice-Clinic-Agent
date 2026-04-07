"""Action parser and validator for VoiceClinicAgent."""

from typing import Dict, Any, Optional
from .constants import ACTION_TYPES


class ParsedAction:
    """Result of action parsing."""
    
    def __init__(self, is_valid: bool, action: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.is_valid = is_valid
        self.action = action
        self.error = error


class ActionParser:
    """
    Parses and validates agent actions.
    
    Ensures actions have correct structure and required fields.
    """
    
    # Required fields for each action type
    REQUIRED_FIELDS = {
        "ask_question": ["question_type", "question_text"],
        "offer_slot": ["slot_id"],
        "confirm_booking": ["slot_id", "patient_confirmation"],
        "escalate_urgent": ["urgency_reason", "symptoms"],
        "escalate_human": [],
        "query_availability": ["department"],
        "check_insurance": [],
        "trigger_reflection": [],
        "store_memory_safe": ["memory_key", "value"],
        "recall_memory": ["memory_key"],
        "end_call": [],
    }
    
    def parse(self, action: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> ParsedAction:
        """
        Parse and validate action.
        
        Args:
            action: Action dict with action_type and payload
            state: Optional episode state for context-aware validation
            
        Returns:
            ParsedAction with is_valid flag and error message if invalid
        """
        # Check action_type exists
        action_type = action.get("action_type")
        if not action_type:
            return ParsedAction(
                is_valid=False,
                error="Missing action_type field"
            )
        
        # Check action_type is valid
        if action_type not in ACTION_TYPES:
            return ParsedAction(
                is_valid=False,
                error=f"Invalid action_type: {action_type}. Must be one of: {ACTION_TYPES}"
            )
        
        # Check payload exists
        payload = action.get("payload", {})
        if not isinstance(payload, dict):
            return ParsedAction(
                is_valid=False,
                error="Payload must be a dict"
            )
        
        # Check required fields for this action type
        required_fields = self.REQUIRED_FIELDS.get(action_type, [])
        missing_fields = [field for field in required_fields if field not in payload]
        
        if missing_fields:
            return ParsedAction(
                is_valid=False,
                error=f"Missing required fields for {action_type}: {missing_fields}"
            )
        
        # Validate specific action types
        validation_error = self._validate_action_specifics(action_type, payload, state)
        if validation_error:
            return ParsedAction(
                is_valid=False,
                error=validation_error
            )
        
        # Action is valid
        return ParsedAction(
            is_valid=True,
            action=action
        )
    
    def _validate_action_specifics(
        self,
        action_type: str,
        payload: Dict[str, Any],
        state: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Validate action-specific constraints.
        
        Args:
            action_type: Action type
            payload: Action payload
            state: Episode state
            
        Returns:
            Error message if invalid, None if valid
        """
        if action_type == "ask_question":
            question_type = payload.get("question_type", "")
            question_text = payload.get("question_text", "")
            
            if not question_type or not isinstance(question_type, str):
                return "question_type must be a non-empty string"
            
            if not question_text or not isinstance(question_text, str):
                return "question_text must be a non-empty string"
        
        elif action_type == "offer_slot":
            slot_id = payload.get("slot_id", "")
            
            if not slot_id or not isinstance(slot_id, str):
                return "slot_id must be a non-empty string"
        
        elif action_type == "confirm_booking":
            slot_id = payload.get("slot_id", "")
            patient_confirmation = payload.get("patient_confirmation")
            
            if not slot_id or not isinstance(slot_id, str):
                return "slot_id must be a non-empty string"
            
            if not isinstance(patient_confirmation, bool):
                return "patient_confirmation must be a boolean"
        
        elif action_type == "store_memory_safe":
            memory_key = payload.get("memory_key", "")
            
            if not memory_key or not isinstance(memory_key, str):
                return "memory_key must be a non-empty string"
            
            if "value" not in payload:
                return "value field is required"
        
        elif action_type == "recall_memory":
            memory_key = payload.get("memory_key", "")
            
            if not memory_key or not isinstance(memory_key, str):
                return "memory_key must be a non-empty string"
        
        elif action_type == "query_availability":
            department = payload.get("department", "")
            
            if not department or not isinstance(department, str):
                return "department must be a non-empty string"
        
        elif action_type == "escalate_urgent":
            urgency_reason = payload.get("urgency_reason", "")
            symptoms = payload.get("symptoms", "")
            
            if not urgency_reason or not isinstance(urgency_reason, str):
                return "urgency_reason must be a non-empty string"
            
            if not symptoms or not isinstance(symptoms, str):
                return "symptoms must be a non-empty string"
        
        return None
