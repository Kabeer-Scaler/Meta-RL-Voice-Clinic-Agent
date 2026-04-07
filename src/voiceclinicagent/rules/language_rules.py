"""
Language handling rules for VoiceClinicAgent.

Detects language mixing and generates appropriate mixed-language responses.
"""

import random
from typing import Dict, List, Any, Optional


# Common Hindi-English mixed phrases for clinic context
HINDI_GREETINGS = [
    "Namaste",
    "Aap kaise hain",
    "Theek hai",
    "Dhanyavaad"
]

HINDI_MEDICAL_TERMS = [
    "bukhar",  # fever
    "dard",    # pain
    "sir dard", # headache
    "pet dard", # stomach ache
    "khasi",   # cough
    "doctor sahab",
    "dawai"    # medicine
]

HINDI_TIME_TERMS = [
    "kal",     # tomorrow
    "aaj",     # today
    "subah",   # morning
    "shaam",   # evening
    "abhi"     # now
]


def detect_language_mix(text: str) -> float:
    """
    Detect the level of language mixing in text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language mix score [0.0, 1.0] where 0.0 is pure English, 1.0 is heavy mixing
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    words = text_lower.split()
    
    if len(words) == 0:
        return 0.0
    
    # Count Hindi words
    hindi_word_count = 0
    all_hindi_terms = HINDI_GREETINGS + HINDI_MEDICAL_TERMS + HINDI_TIME_TERMS
    
    for word in words:
        for hindi_term in all_hindi_terms:
            if hindi_term.lower() in word:
                hindi_word_count += 1
                break
    
    # Calculate mix ratio
    mix_ratio = hindi_word_count / len(words)
    return min(1.0, mix_ratio * 2.0)  # Scale up for visibility


def get_mixed_response(
    template: str,
    language_mix: float,
    rng: random.Random
) -> str:
    """
    Generate a response with appropriate language mixing.
    
    Args:
        template: Base English response template
        language_mix: Desired language mix level [0.0, 1.0]
        rng: Seeded random number generator
        
    Returns:
        Response with language mixing applied
    """
    if language_mix < 0.3:
        # Low mixing - just add occasional Hindi greeting
        if rng.random() < 0.3:
            greeting = rng.choice(HINDI_GREETINGS)
            return f"{greeting}. {template}"
        return template
    
    elif language_mix < 0.6:
        # Medium mixing - replace some medical/time terms
        response = template
        
        # Replace some medical terms
        replacements = {
            "fever": "bukhar",
            "pain": "dard",
            "headache": "sir dard",
            "cough": "khasi",
            "doctor": "doctor sahab"
        }
        
        for eng, hindi in replacements.items():
            if eng in response.lower() and rng.random() < 0.5:
                response = response.replace(eng, hindi)
                response = response.replace(eng.capitalize(), hindi.capitalize())
        
        return response
    
    else:
        # High mixing - more aggressive replacement
        response = template
        
        # Add Hindi greeting
        if rng.random() < 0.7:
            greeting = rng.choice(HINDI_GREETINGS)
            response = f"{greeting}. {response}"
        
        # Replace medical terms
        replacements = {
            "fever": "bukhar",
            "pain": "dard",
            "headache": "sir dard",
            "stomach ache": "pet dard",
            "cough": "khasi",
            "doctor": "doctor sahab",
            "medicine": "dawai",
            "tomorrow": "kal",
            "today": "aaj",
            "morning": "subah",
            "evening": "shaam",
            "now": "abhi"
        }
        
        for eng, hindi in replacements.items():
            if eng in response.lower() and rng.random() < 0.7:
                response = response.replace(eng, hindi)
                response = response.replace(eng.capitalize(), hindi.capitalize())
        
        return response


def should_use_mixed_language(
    task_difficulty: str,
    language_mix_config: float
) -> bool:
    """
    Determine if mixed language should be used based on task difficulty.
    
    Args:
        task_difficulty: "easy", "medium", or "hard"
        language_mix_config: Language mix level from scenario config
        
    Returns:
        True if mixed language should be used
    """
    if task_difficulty == "easy":
        return False
    
    if task_difficulty == "medium":
        return language_mix_config > 0.0
    
    if task_difficulty == "hard":
        return language_mix_config > 0.0
    
    return False


def normalize_mixed_text(text: str) -> str:
    """
    Normalize mixed-language text for comparison.
    
    Args:
        text: Mixed language text
        
    Returns:
        Normalized text with common Hindi terms translated to English
    """
    normalized = text.lower()
    
    # Reverse mapping for normalization
    hindi_to_english = {
        "bukhar": "fever",
        "dard": "pain",
        "sir dard": "headache",
        "pet dard": "stomach ache",
        "khasi": "cough",
        "doctor sahab": "doctor",
        "dawai": "medicine",
        "kal": "tomorrow",
        "aaj": "today",
        "subah": "morning",
        "shaam": "evening",
        "abhi": "now",
        "namaste": "hello",
        "dhanyavaad": "thank you",
        "theek hai": "okay"
    }
    
    for hindi, english in hindi_to_english.items():
        normalized = normalized.replace(hindi, english)
    
    return normalized


def extract_intent_from_mixed(text: str) -> Dict[str, Any]:
    """
    Extract intent from mixed-language text.
    
    Args:
        text: Mixed language text
        
    Returns:
        Dict with extracted intent information
    """
    normalized = normalize_mixed_text(text)
    
    # Simple keyword-based intent extraction
    intent = {
        "has_symptoms": False,
        "symptoms": [],
        "has_time_preference": False,
        "time_preference": None,
        "has_urgency": False
    }
    
    # Check for symptoms
    symptom_keywords = ["fever", "pain", "headache", "stomach ache", "cough"]
    for symptom in symptom_keywords:
        if symptom in normalized:
            intent["has_symptoms"] = True
            intent["symptoms"].append(symptom)
    
    # Check for time preferences
    time_keywords = ["tomorrow", "today", "morning", "evening", "now"]
    for time_word in time_keywords:
        if time_word in normalized:
            intent["has_time_preference"] = True
            intent["time_preference"] = time_word
            break
    
    # Check for urgency
    urgency_keywords = ["urgent", "emergency", "immediately", "now", "abhi", "serious"]
    for urgency_word in urgency_keywords:
        if urgency_word in normalized:
            intent["has_urgency"] = True
            break
    
    return intent
