"""Observation builder for VoiceClinicAgent."""

from typing import Dict, Any, List, Optional
from .api_models import VoiceClinicObservation
from .user_simulator import PatientResponse
from .models import ClinicalHistory


class ObservationBuilder:
    """
    Builds structured observations from episode state.
    
    Assembles all observation fields from various components:
    - Patient simulator state
    - Clinic state (slots, delays, queue)
    - Reflection token
    - Memory vault summary
    - Privacy risk mask
    """
    
    def build(
        self,
        task_level: str,
        turn_idx: int,
        max_turns: int,
        patient_response: PatientResponse,
        conversation_summary: str,
        patient_flags: Dict[str, Any],
        clinic_state: Dict[str, Any],
        clinical_history: ClinicalHistory,
        reflection_token: List[float],
        memory_vault_summary: Dict[str, Any],
        privacy_risk_mask: Dict[str, int],
        done: bool,
        reward: float,
        history_accessed_this_turn: bool = False,
    ) -> VoiceClinicObservation:
        """
        Build complete observation.
        
        Args:
            task_level: "easy", "medium", or "hard"
            turn_idx: Current turn number
            max_turns: Maximum turns for episode
            patient_response: Patient's response object
            conversation_summary: Brief conversation summary
            patient_flags: Patient characteristic flags
            clinic_state: Current clinic state
            clinical_history: Patient clinical history summary
            reflection_token: 4-element reflection vector
            memory_vault_summary: Stored preferences summary
            privacy_risk_mask: PII risk flags
            done: Episode done flag
            reward: Reward for this step
            history_accessed_this_turn: Whether agent accessed history this turn
            
        Returns:
            VoiceClinicObservation
        """
        return VoiceClinicObservation(
            done=done,
            reward=reward,
            task_level=task_level,
            turn_idx=turn_idx,
            max_turns=max_turns,
            patient_message=patient_response.text,
            conversation_summary=conversation_summary,
            patient_flags=patient_flags,
            clinic_state=clinic_state,
            clinical_history=clinical_history,
            reflection_token=reflection_token,
            memory_vault_summary=memory_vault_summary,
            privacy_risk_mask=privacy_risk_mask,
            history_accessed_this_turn=history_accessed_this_turn,
        )
    
    @staticmethod
    def build_patient_flags(
        patient_profile: Any,
        frustration: float,
        revealed_facts: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build patient flags dict from patient state.
        
        Args:
            patient_profile: PatientProfile object
            frustration: Current frustration level
            revealed_facts: Facts revealed so far
            
        Returns:
            Patient flags dict
        """
        return {
            "parent_booking": 1 if patient_profile.parent_booking else 0,
            "language_mix": patient_profile.language_mix,
            "vague_symptoms": 1 if patient_profile.vagueness_level > 0.5 else 0,
            "urgency_score_hint": patient_profile.urgency_level,
            "insurance_status_known": 1 if "insurance" in revealed_facts else 0,
            "duplicate_info_flag": 0,  # Will be set by receptionist agent
        }
    
    @staticmethod
    def build_clinic_state(
        available_slots: Dict[str, List[str]],
        doctor_delay_minutes: int,
        urgent_queue_length: int,
    ) -> Dict[str, Any]:
        """
        Build clinic state dict.
        
        Args:
            available_slots: Department -> list of slot_ids
            doctor_delay_minutes: Current delay
            urgent_queue_length: Queue length
            
        Returns:
            Clinic state dict
        """
        return {
            "available_slots": available_slots,
            "doctor_delay_minutes": doctor_delay_minutes,
            "urgent_queue_length": urgent_queue_length,
        }
    
    @staticmethod
    def build_default_reflection_token() -> List[float]:
        """
        Build default reflection token (all zeros).
        
        Returns:
            [0.0, 0.0, 0.0, 0.0]
        """
        return [0.0, 0.0, 0.0, 0.0]
    
    @staticmethod
    def build_privacy_risk_mask(action_history: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Build privacy risk mask based on conversation.
        
        Args:
            action_history: List of actions taken
            
        Returns:
            Dict with PII field names and risk flags (0 or 1)
        """
        # Simple heuristic: check if any questions asked about PII
        risk_mask = {
            "name": 0,
            "phone": 0,
            "aadhaar": 0,
            "insurance_id": 0,
        }
        
        for action in action_history:
            if action.get("action_type") == "ask_question":
                question_text = action.get("payload", {}).get("question_text", "").lower()
                
                if "name" in question_text:
                    risk_mask["name"] = 1
                if "phone" in question_text or "mobile" in question_text:
                    risk_mask["phone"] = 1
                if "aadhaar" in question_text or "aadhar" in question_text:
                    risk_mask["aadhaar"] = 1
                if "insurance" in question_text:
                    risk_mask["insurance_id"] = 1
        
        return risk_mask
    
    @staticmethod
    def build_clinical_history_from_scenario(scenario: Any) -> ClinicalHistory:
        """
        Build clinical history from scenario definition.
        
        Args:
            scenario: Scenario object
            
        Returns:
            ClinicalHistory object
        """
        # Check if scenario has clinical_history defined
        if hasattr(scenario, 'clinical_history'):
            history_data = scenario.clinical_history
            # Handle both dict and object access
            if isinstance(history_data, dict):
                return ClinicalHistory(
                    patient_id=history_data.get("patient_id", "unknown"),
                    has_history=history_data.get("has_history", False),
                    chronic_conditions=history_data.get("chronic_conditions", []),
                    allergies=history_data.get("allergies", []),
                    prior_visits=history_data.get("prior_visits", []),
                    prior_escalations=history_data.get("prior_escalations", []),
                    language_preference=history_data.get("language_preference"),
                    slot_preference_time=history_data.get("slot_preference_time"),
                    preferred_department=history_data.get("preferred_department"),
                    active_problems=history_data.get("active_problems", []),
                    current_medications=history_data.get("current_medications", []),
                    high_risk_flag=history_data.get("high_risk_flag", False),
                    frequent_visitor=history_data.get("frequent_visitor", False),
                )
            else:
                # Already a ClinicalHistory object
                return history_data
        
        # No history - new patient
        patient_id = getattr(scenario.patient_profile, 'patient_id', 'unknown') if hasattr(scenario, 'patient_profile') else 'unknown'
        return ClinicalHistory(
            patient_id=patient_id,
            has_history=False,
        )
