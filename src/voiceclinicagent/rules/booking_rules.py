"""
Booking validation rules for VoiceClinicAgent.

Validates slot offers, checks for duplicate bookings, and ensures booking consistency.
"""

from typing import Dict, List, Any, Optional
from ..models import ClinicState


def validate_slot_offer(
    slot_id: str,
    clinic_state: ClinicState
) -> Dict[str, Any]:
    """
    Validate that a slot offer is valid.
    
    Args:
        slot_id: The slot identifier being offered
        clinic_state: Current clinic state with available slots
        
    Returns:
        Dict with:
            - is_valid: bool
            - reason: str (if invalid)
            - slot_details: Optional[Dict] (if valid)
    """
    # Check if slot exists in available slots
    available_slots = clinic_state.available_slots
    
    slot_found = None
    for slot in available_slots:
        if slot.get("slot_id") == slot_id:
            slot_found = slot
            break
    
    if not slot_found:
        return {
            "is_valid": False,
            "reason": f"Slot {slot_id} not found in available slots",
            "slot_details": None
        }
    
    # Check if slot is already reserved
    if slot_found.get("reserved", False):
        return {
            "is_valid": False,
            "reason": f"Slot {slot_id} is already reserved",
            "slot_details": None
        }
    
    return {
        "is_valid": True,
        "reason": "",
        "slot_details": slot_found
    }


def check_duplicate_booking(
    patient_id: str,
    history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Check if patient already has a confirmed booking.
    
    Args:
        patient_id: Patient identifier
        history: Action history for the episode
        
    Returns:
        Dict with:
            - has_duplicate: bool
            - existing_booking: Optional[Dict] (if duplicate found)
    """
    for entry in history:
        action = entry.get("action", {})
        response = entry.get("response", {})
        
        # Check for confirmed bookings
        if action.get("action_type") == "confirm_booking":
            if response.get("status") == "confirmed":
                return {
                    "has_duplicate": True,
                    "existing_booking": {
                        "slot_id": action.get("payload", {}).get("slot_id"),
                        "turn": entry.get("turn_idx")
                    }
                }
    
    return {
        "has_duplicate": False,
        "existing_booking": None
    }


def is_correct_department(
    offered_department: str,
    patient_symptoms: List[str],
    correct_department: str
) -> bool:
    """
    Check if the offered department matches the correct department for symptoms.
    
    Args:
        offered_department: Department being offered
        patient_symptoms: List of patient symptoms
        correct_department: The correct department from scenario
        
    Returns:
        True if department is correct
    """
    return offered_department.lower() == correct_department.lower()


def calculate_booking_efficiency(
    turn_confirmed: int,
    max_turns: int
) -> float:
    """
    Calculate booking efficiency bonus based on how quickly booking was confirmed.
    
    Args:
        turn_confirmed: Turn number when booking was confirmed
        max_turns: Maximum turns allowed
        
    Returns:
        Efficiency score in [0.0, 1.0]
    """
    if turn_confirmed <= 0 or max_turns <= 0:
        return 0.0
    
    # Earlier booking = higher efficiency
    # Perfect efficiency (1.0) if booked in first 20% of turns
    # Linear decay to 0.0 at max_turns
    efficiency_threshold = max(1, int(0.2 * max_turns))
    
    if turn_confirmed <= efficiency_threshold:
        return 1.0
    
    remaining_turns = max_turns - efficiency_threshold
    turns_used = turn_confirmed - efficiency_threshold
    
    efficiency = max(0.0, 1.0 - (turns_used / remaining_turns))
    return efficiency
