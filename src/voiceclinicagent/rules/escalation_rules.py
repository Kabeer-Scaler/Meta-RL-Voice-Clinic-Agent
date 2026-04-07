"""
Escalation validation rules for VoiceClinicAgent.

Determines when escalation is appropriate and evaluates escalation timing.
"""

from typing import Dict, Any


def should_escalate_urgent(
    urgency_score: float,
    queue_length: int,
    available_slots_count: int
) -> Dict[str, Any]:
    """
    Determine if urgent escalation is appropriate.
    
    Args:
        urgency_score: Patient urgency score [0.0, 1.0]
        queue_length: Current urgent queue length
        available_slots_count: Number of available slots
        
    Returns:
        Dict with:
            - should_escalate: bool
            - reason: str
            - urgency_level: str ("low", "medium", "high", "critical")
    """
    # Determine urgency level
    if urgency_score >= 0.9:
        urgency_level = "critical"
    elif urgency_score >= 0.7:
        urgency_level = "high"
    elif urgency_score >= 0.4:
        urgency_level = "medium"
    else:
        urgency_level = "low"
    
    # Critical cases should always escalate
    if urgency_score >= 0.9:
        return {
            "should_escalate": True,
            "reason": "Critical urgency requires immediate attention",
            "urgency_level": urgency_level
        }
    
    # High urgency with limited slots
    if urgency_score >= 0.7 and available_slots_count < 3:
        return {
            "should_escalate": True,
            "reason": "High urgency with limited slot availability",
            "urgency_level": urgency_level
        }
    
    # High urgency with long queue
    if urgency_score >= 0.7 and queue_length >= 5:
        return {
            "should_escalate": True,
            "reason": "High urgency with long urgent queue",
            "urgency_level": urgency_level
        }
    
    # Medium urgency generally doesn't need escalation
    return {
        "should_escalate": False,
        "reason": "Urgency level does not require escalation",
        "urgency_level": urgency_level
    }


def should_escalate_human(
    frustration_level: float,
    turn_idx: int,
    max_turns: int,
    repeated_failures: int
) -> Dict[str, Any]:
    """
    Determine if human escalation is appropriate due to conversation issues.
    
    Args:
        frustration_level: Patient frustration [0.0, 1.0]
        turn_idx: Current turn number
        max_turns: Maximum allowed turns
        repeated_failures: Count of consecutive failed actions
        
    Returns:
        Dict with:
            - should_escalate: bool
            - reason: str
    """
    # High frustration
    if frustration_level >= 0.8:
        return {
            "should_escalate": True,
            "reason": "Patient frustration level is very high"
        }
    
    # Multiple repeated failures
    if repeated_failures >= 3:
        return {
            "should_escalate": True,
            "reason": "Multiple consecutive failed actions"
        }
    
    # Near max turns with no progress
    if turn_idx >= int(0.8 * max_turns) and frustration_level >= 0.5:
        return {
            "should_escalate": True,
            "reason": "Approaching max turns with elevated frustration"
        }
    
    return {
        "should_escalate": False,
        "reason": "Conversation is progressing normally"
    }


def evaluate_escalation_timing(
    turn_escalated: int,
    urgency_score: float,
    max_turns: int
) -> float:
    """
    Evaluate if escalation happened at the right time.
    
    Args:
        turn_escalated: Turn when escalation occurred
        urgency_score: Patient urgency score
        max_turns: Maximum allowed turns
        
    Returns:
        Timing score in [0.0, 1.0]
    """
    # For critical urgency, should escalate very early
    if urgency_score >= 0.9:
        ideal_turn = min(3, max_turns)
        if turn_escalated <= ideal_turn:
            return 1.0
        # Penalty for late escalation of critical cases
        delay = turn_escalated - ideal_turn
        return max(0.0, 1.0 - (delay * 0.2))
    
    # For high urgency, should escalate reasonably early
    if urgency_score >= 0.7:
        ideal_turn = min(5, int(0.3 * max_turns))
        if turn_escalated <= ideal_turn:
            return 1.0
        delay = turn_escalated - ideal_turn
        return max(0.0, 1.0 - (delay * 0.1))
    
    # For medium/low urgency, escalation timing is less critical
    # But shouldn't escalate too early (wastes resources)
    if turn_escalated < 3:
        return 0.5  # Premature escalation
    
    return 0.8  # Reasonable timing


def check_escalation_appropriateness(
    action_type: str,
    urgency_score: float,
    frustration_level: float,
    available_slots_count: int
) -> Dict[str, Any]:
    """
    Check if the escalation action is appropriate for the situation.
    
    Args:
        action_type: "escalate_urgent" or "escalate_human"
        urgency_score: Patient urgency score
        frustration_level: Patient frustration level
        available_slots_count: Number of available slots
        
    Returns:
        Dict with:
            - is_appropriate: bool
            - reason: str
            - score: float [0.0, 1.0]
    """
    if action_type == "escalate_urgent":
        # Check if urgency justifies escalation
        if urgency_score >= 0.7:
            return {
                "is_appropriate": True,
                "reason": "Urgency level justifies urgent escalation",
                "score": 1.0
            }
        elif urgency_score >= 0.4 and available_slots_count < 2:
            return {
                "is_appropriate": True,
                "reason": "Moderate urgency with very limited slots",
                "score": 0.7
            }
        else:
            return {
                "is_appropriate": False,
                "reason": "Urgency level does not justify escalation",
                "score": 0.0
            }
    
    elif action_type == "escalate_human":
        # Check if conversation issues justify human escalation
        if frustration_level >= 0.7:
            return {
                "is_appropriate": True,
                "reason": "High frustration justifies human escalation",
                "score": 1.0
            }
        elif frustration_level >= 0.5:
            return {
                "is_appropriate": True,
                "reason": "Moderate frustration, human escalation reasonable",
                "score": 0.7
            }
        else:
            return {
                "is_appropriate": False,
                "reason": "Frustration level does not justify human escalation",
                "score": 0.3
            }
    
    return {
        "is_appropriate": False,
        "reason": "Unknown escalation type",
        "score": 0.0
    }
