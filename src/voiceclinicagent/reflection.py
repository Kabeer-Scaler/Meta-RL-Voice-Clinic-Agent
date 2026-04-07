"""Reflection engine for VoiceClinicAgent."""

from typing import List, Dict, Any, Optional


class ReflectionEngine:
    """
    Computes reflection token for agent self-awareness.
    
    Reflection token is a 4-element vector [0.0, 1.0]^4:
    - [0]: need_escalation (urgency + deterioration + high_risk)
    - [1]: info_missing (required facts not yet collected)
    - [2]: privacy_risk_high (PII questions asked)
    - [3]: slot_pressure (urgent queue vs available slots)
    """
    
    def __init__(self):
        """Initialize reflection engine."""
        pass
    
    def compute_reflection_token(
        self,
        scenario: Any,
        action_history: List[Dict[str, Any]],
        revealed_facts: Dict[str, str],
        available_slots: Dict[str, List[str]],
        urgent_queue_length: int,
        turn_idx: int,
        max_turns: int,
    ) -> List[float]:
        """
        Compute reflection token from current episode state.
        
        Args:
            scenario: Current scenario object
            action_history: List of actions taken so far
            revealed_facts: Facts revealed by patient
            available_slots: Available slots by department
            urgent_queue_length: Current urgent queue length
            turn_idx: Current turn number
            max_turns: Maximum turns allowed
            
        Returns:
            4-element list of floats in [0.0, 1.0]
        """
        # [0] need_escalation: based on urgency, deterioration, high_risk
        need_escalation = self._compute_need_escalation(
            scenario,
            turn_idx,
            max_turns,
        )
        
        # [1] info_missing: based on required info vs revealed facts
        info_missing = self._compute_info_missing(
            scenario,
            revealed_facts,
        )
        
        # [2] privacy_risk_high: based on PII questions asked
        privacy_risk_high = self._compute_privacy_risk(
            action_history,
        )
        
        # [3] slot_pressure: based on urgent queue vs available slots
        slot_pressure = self._compute_slot_pressure(
            available_slots,
            urgent_queue_length,
        )
        
        return [
            need_escalation,
            info_missing,
            privacy_risk_high,
            slot_pressure,
        ]
    
    def _compute_need_escalation(
        self,
        scenario: Any,
        turn_idx: int,
        max_turns: int,
    ) -> float:
        """
        Compute need_escalation signal.
        
        Factors:
        - Patient urgency level (from scenario)
        - Deterioration rate (hard scenarios)
        - High risk flag (from clinical history)
        - Time pressure (turns remaining)
        
        Returns:
            Float in [0.0, 1.0]
        """
        urgency = scenario.patient_profile.urgency_level
        
        # Check for deterioration (hard scenarios)
        deterioration_rate = 0.0
        if hasattr(scenario, 'deterioration_dynamics') and scenario.deterioration_dynamics:
            deterioration_rate = scenario.deterioration_dynamics.get('rate', 0.0)
        
        # Check high risk flag from clinical history
        high_risk = 0.0
        if hasattr(scenario, 'clinical_history'):
            high_risk = 1.0 if scenario.clinical_history.high_risk_flag else 0.0
        
        # Time pressure: increases as we approach max_turns
        time_pressure = turn_idx / max_turns if max_turns > 0 else 0.0
        
        # Weighted combination
        score = (
            0.5 * urgency +
            0.3 * min(deterioration_rate * 10, 1.0) +  # Scale deterioration rate
            0.1 * high_risk +
            0.1 * time_pressure
        )
        
        return min(max(score, 0.0), 1.0)
    
    def _compute_info_missing(
        self,
        scenario: Any,
        revealed_facts: Dict[str, str],
    ) -> float:
        """
        Compute info_missing signal.
        
        Checks which required facts haven't been collected yet.
        
        Returns:
            Float in [0.0, 1.0] (1.0 = all info missing, 0.0 = all collected)
        """
        # Get required info from ground truth
        if not hasattr(scenario, 'ground_truth'):
            return 0.0
        
        ground_truth = scenario.ground_truth
        required_keys = set()
        
        # Collect required keys from ground truth
        if hasattr(ground_truth, 'symptoms') and ground_truth.symptoms:
            required_keys.add('symptoms')
        if hasattr(ground_truth, 'preferred_department') and ground_truth.preferred_department:
            required_keys.add('department')
        if hasattr(ground_truth, 'urgency_level'):
            required_keys.add('urgency')
        
        # Check what's been revealed
        if not required_keys:
            return 0.0
        
        revealed_keys = set(revealed_facts.keys())
        missing_count = len(required_keys - revealed_keys)
        
        return missing_count / len(required_keys)
    
    def _compute_privacy_risk(
        self,
        action_history: List[Dict[str, Any]],
    ) -> float:
        """
        Compute privacy_risk_high signal.
        
        Checks if agent has asked about PII fields.
        
        Returns:
            Float in [0.0, 1.0]
        """
        pii_keywords = {
            'name', 'phone', 'mobile', 'aadhaar', 'aadhar',
            'insurance_id', 'insurance number', 'address', 'email',
        }
        
        pii_question_count = 0
        total_questions = 0
        
        for action in action_history:
            if action.get('action_type') == 'ask_question':
                total_questions += 1
                question_text = action.get('payload', {}).get('question_text', '').lower()
                
                # Check if question contains PII keywords
                if any(keyword in question_text for keyword in pii_keywords):
                    pii_question_count += 1
        
        if total_questions == 0:
            return 0.0
        
        # Risk increases with proportion of PII questions
        risk_ratio = pii_question_count / total_questions
        
        return min(risk_ratio * 2.0, 1.0)  # Scale up for sensitivity
    
    def _compute_slot_pressure(
        self,
        available_slots: Dict[str, List[str]],
        urgent_queue_length: int,
    ) -> float:
        """
        Compute slot_pressure signal.
        
        Measures pressure from urgent queue vs available capacity.
        
        Returns:
            Float in [0.0, 1.0]
        """
        # Count total available slots
        total_slots = sum(len(slots) for slots in available_slots.values())
        
        if total_slots == 0:
            return 1.0  # Maximum pressure if no slots
        
        # Pressure increases with queue length relative to capacity
        pressure = urgent_queue_length / (total_slots + urgent_queue_length)
        
        return min(max(pressure, 0.0), 1.0)
