"""Termination checker for VoiceClinicAgent."""

from typing import Dict, Any, Tuple, Optional


class TerminationChecker:
    """
    Checks episode termination conditions.
    
    Done conditions:
    - Booking confirmed successfully
    - Escalated to urgent queue
    - Escalated to human operator
    - Hard privacy violation (blocked PII access)
    
    Truncated conditions:
    - Maximum turns exceeded
    - Patient frustration >= 1.0 (hung up)
    """
    
    def check(
        self,
        action: Dict[str, Any],
        turn_idx: int,
        max_turns: int,
        frustration: float,
        privacy_violation_hard: bool = False,
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Check if episode should terminate.
        
        Args:
            action: Current action
            turn_idx: Current turn number
            max_turns: Maximum turns allowed
            frustration: Patient frustration level
            privacy_violation_hard: Whether hard privacy violation occurred
            
        Returns:
            Tuple of (done, truncated, termination_reason)
        """
        action_type = action.get("action_type", "")
        
        # Check done conditions
        if action_type == "confirm_booking":
            return True, False, "booking_confirmed"
        
        if action_type == "escalate_urgent":
            return True, False, "escalated_urgent"
        
        if action_type == "escalate_human":
            return True, False, "escalated_human"
        
        if privacy_violation_hard:
            return True, False, "privacy_violation_hard"
        
        # Check truncated conditions
        if turn_idx >= max_turns:
            return False, True, "max_turns_exceeded"
        
        if frustration >= 1.0:
            return False, True, "patient_hung_up"
        
        # Episode continues
        return False, False, None
