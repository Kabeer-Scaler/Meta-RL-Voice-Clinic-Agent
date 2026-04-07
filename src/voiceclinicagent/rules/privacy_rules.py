"""Privacy rules for VoiceClinicAgent."""

from typing import Dict, Any, Tuple
from ..constants import BLOCKED_PII_KEYS, ALLOWED_MEMORY_KEYS
from ..utils.text import contains_pii_pattern


def check_conversation_pii(action: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if conversation contains PII patterns.
    
    Args:
        action: Action dict
        
    Returns:
        Tuple of (has_pii, violation_message)
    """
    if action.get("action_type") != "ask_question":
        return False, ""
    
    question_text = action.get("payload", {}).get("question_text", "")
    
    if contains_pii_pattern(question_text):
        return True, f"Question contains PII pattern: {question_text[:50]}..."
    
    return False, ""


def is_blocked_pii_key(key: str) -> bool:
    """
    Check if memory key is blocked PII.
    
    Args:
        key: Memory key to check
        
    Returns:
        True if key is blocked PII
    """
    normalized_key = key.lower().strip()
    return normalized_key in BLOCKED_PII_KEYS


def is_allowed_memory_key(key: str) -> bool:
    """
    Check if memory key is allowed.
    
    Args:
        key: Memory key to check
        
    Returns:
        True if key is allowed
    """
    normalized_key = key.lower().strip()
    return normalized_key in ALLOWED_MEMORY_KEYS


def validate_memory_access(
    action_type: str,
    memory_key: str,
) -> Tuple[bool, bool, str]:
    """
    Validate memory access attempt.
    
    Args:
        action_type: "store_memory_safe" or "recall_memory"
        memory_key: Memory key being accessed
        
    Returns:
        Tuple of (is_valid, is_hard_violation, error_message)
    """
    if action_type not in ["store_memory_safe", "recall_memory"]:
        return True, False, ""
    
    # Check if key is blocked PII (hard violation)
    if is_blocked_pii_key(memory_key):
        return False, True, f"BLOCKED: Cannot access PII key '{memory_key}'"
    
    # Check if key is allowed
    if not is_allowed_memory_key(memory_key):
        return False, False, f"Key '{memory_key}' not in allowed list"
    
    return True, False, ""
