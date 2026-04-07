"""Privacy-safe memory vault for VoiceClinicAgent."""

from typing import Dict, Any, Optional, Tuple
from .constants import BLOCKED_PII_KEYS


class PrivacySafeMemoryVault:
    """
    Privacy-safe memory storage with key enforcement.
    
    Allowed keys: preferences, symptoms_summary, booking_notes, follow_up_needed
    Blocked keys: name, phone, aadhaar, insurance_id, address, email, dob
    
    Attempting to store/recall blocked keys triggers hard termination.
    """
    
    # Allowed memory keys (safe to store)
    ALLOWED_KEYS = {
        "preferences",
        "symptoms_summary",
        "booking_notes",
        "follow_up_needed",
        "preferred_time",
        "preferred_department",
        "urgency_notes",
        "special_requests",
    }
    
    def __init__(self):
        """Initialize empty memory vault."""
        self._storage: Dict[str, Dict[str, Any]] = {}  # patient_id -> {key: value}
        self._access_log: list = []
    
    def store(
        self,
        key: str,
        value: Any,
        patient_id: str,
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Store a key-value pair for a patient.
        
        Args:
            key: Memory key to store
            value: Value to store
            patient_id: Patient identifier
            
        Returns:
            Tuple of (success, terminate_episode, error_message)
            - success: True if stored successfully
            - terminate_episode: True if blocked PII key was attempted
            - error_message: Error description if failed
        """
        key_lower = key.lower().strip()
        
        # Check if key is blocked PII
        if key_lower in BLOCKED_PII_KEYS:
            self._access_log.append({
                "action": "store_blocked",
                "key": key_lower,
                "patient_id": patient_id,
            })
            return False, True, f"BLOCKED: Cannot store PII key '{key}'"
        
        # Check if key is allowed
        if key_lower not in self.ALLOWED_KEYS:
            self._access_log.append({
                "action": "store_invalid",
                "key": key_lower,
                "patient_id": patient_id,
            })
            return False, False, f"Invalid memory key '{key}'. Allowed keys: {', '.join(self.ALLOWED_KEYS)}"
        
        # Store the value
        if patient_id not in self._storage:
            self._storage[patient_id] = {}
        
        self._storage[patient_id][key_lower] = value
        
        self._access_log.append({
            "action": "store_success",
            "key": key_lower,
            "patient_id": patient_id,
        })
        
        return True, False, None
    
    def recall(
        self,
        key: str,
        patient_id: str,
    ) -> Tuple[bool, bool, Optional[Any], Optional[str]]:
        """
        Recall a stored value for a patient.
        
        Args:
            key: Memory key to recall
            patient_id: Patient identifier
            
        Returns:
            Tuple of (success, terminate_episode, value, error_message)
            - success: True if recalled successfully
            - terminate_episode: True if blocked PII key was attempted
            - value: Stored value if found, None otherwise
            - error_message: Error description if failed
        """
        key_lower = key.lower().strip()
        
        # Check if key is blocked PII
        if key_lower in BLOCKED_PII_KEYS:
            self._access_log.append({
                "action": "recall_blocked",
                "key": key_lower,
                "patient_id": patient_id,
            })
            return False, True, None, f"BLOCKED: Cannot recall PII key '{key}'"
        
        # Check if key is allowed
        if key_lower not in self.ALLOWED_KEYS:
            self._access_log.append({
                "action": "recall_invalid",
                "key": key_lower,
                "patient_id": patient_id,
            })
            return False, False, None, f"Invalid memory key '{key}'. Allowed keys: {', '.join(self.ALLOWED_KEYS)}"
        
        # Recall the value
        if patient_id not in self._storage:
            self._access_log.append({
                "action": "recall_not_found",
                "key": key_lower,
                "patient_id": patient_id,
            })
            return False, False, None, f"No memory stored for patient {patient_id}"
        
        if key_lower not in self._storage[patient_id]:
            self._access_log.append({
                "action": "recall_key_not_found",
                "key": key_lower,
                "patient_id": patient_id,
            })
            return False, False, None, f"Key '{key}' not found in memory"
        
        value = self._storage[patient_id][key_lower]
        
        self._access_log.append({
            "action": "recall_success",
            "key": key_lower,
            "patient_id": patient_id,
        })
        
        return True, False, value, None
    
    def get_summary(self, patient_id: str) -> Dict[str, int]:
        """
        Get summary of stored keys for a patient (presence flags only).
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict with key names as keys and 1/0 as values (1 = stored, 0 = not stored)
        """
        summary = {key: 0 for key in self.ALLOWED_KEYS}
        
        if patient_id in self._storage:
            for key in self._storage[patient_id]:
                if key in summary:
                    summary[key] = 1
        
        return summary
    
    def clear_patient(self, patient_id: str) -> None:
        """
        Clear all memory for a patient.
        
        Args:
            patient_id: Patient identifier
        """
        if patient_id in self._storage:
            del self._storage[patient_id]
    
    def get_access_log(self) -> list:
        """
        Get access log for debugging.
        
        Returns:
            List of access log entries
        """
        return self._access_log.copy()
