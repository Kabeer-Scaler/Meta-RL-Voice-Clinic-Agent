"""Text processing utilities."""

import re
from typing import List, Dict, Any


def is_redundant_question(action: Dict[str, Any], history: List[Dict[str, Any]]) -> bool:
    """
    Check if a question has already been asked.
    
    Args:
        action: Current action dict with action_type and payload
        history: List of previous actions
        
    Returns:
        True if question_type was already asked
    """
    if action.get("action_type") != "ask_question":
        return False
    
    current_question_type = action.get("payload", {}).get("question_type", "")
    if not current_question_type:
        return False
    
    for past_action in history:
        if past_action.get("action_type") == "ask_question":
            past_question_type = past_action.get("payload", {}).get("question_type", "")
            if past_question_type == current_question_type:
                return True
    
    return False


def is_irrelevant_question(action: Dict[str, Any], patient_symptoms: str) -> bool:
    """
    Check if a question is irrelevant to patient symptoms.
    
    Args:
        action: Current action dict
        patient_symptoms: Patient's symptom description
        
    Returns:
        True if question seems unrelated to symptoms
        
    Note: This is a simple heuristic. In production, could use more sophisticated matching.
    """
    if action.get("action_type") != "ask_question":
        return False
    
    question_text = action.get("payload", {}).get("question_text", "").lower()
    symptoms_lower = patient_symptoms.lower()
    
    # Very basic relevance check
    # If question mentions completely unrelated topics, mark as irrelevant
    irrelevant_keywords = ["weather", "sports", "politics", "food preference", "hobby"]
    for keyword in irrelevant_keywords:
        if keyword in question_text:
            return True
    
    return False


def contains_pii_pattern(text: str) -> bool:
    """
    Check if text contains PII patterns (phone numbers, Aadhaar, etc.).
    
    Args:
        text: Text to check
        
    Returns:
        True if PII patterns detected
    """
    text_lower = text.lower()
    
    # Check for explicit PII requests
    pii_keywords = [
        "phone number", "mobile number", "contact number",
        "aadhaar", "aadhar", "aadhaar number",
        "insurance id", "insurance number",
        "email address", "email id",
        "full name", "your name",
        "home address", "residential address",
        "date of birth", "dob",
    ]
    
    for keyword in pii_keywords:
        if keyword in text_lower:
            return True
    
    # Check for phone number patterns (10 digits)
    if re.search(r'\b\d{10}\b', text):
        return True
    
    # Check for Aadhaar patterns (12 digits)
    if re.search(r'\b\d{12}\b', text):
        return True
    
    return False
