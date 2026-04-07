"""Enhanced order-aware reward system for VoiceClinicAgent."""

from typing import Dict, Any, List, Set
from .constants import REWARDS
from .utils.text import is_redundant_question, is_irrelevant_question, contains_pii_pattern


class ConversationState:
    """Tracks conversation state for order-aware rewards."""
    
    def __init__(self):
        self.asked_symptoms = False
        self.asked_duration = False
        self.asked_severity = False
        self.asked_preferred_time = False
        self.queried_availability = False
        self.offered_slot = False
        self.collected_info: Set[str] = set()
        self.turn_count = 0
    
    def update(self, action: Dict[str, Any], revealed_facts: List[str]):
        """Update state based on action and response."""
        self.turn_count += 1
        action_type = action.get("action_type", "")
        
        if action_type == "ask_question":
            q_text = action.get("payload", {}).get("question_text", "").lower()
            q_type = action.get("payload", {}).get("question_type", "").lower()
            
            if "symptom" in q_text or "symptom" in q_type:
                self.asked_symptoms = True
                self.collected_info.add("symptoms")
            elif "duration" in q_text or "how long" in q_text:
                self.asked_duration = True
                self.collected_info.add("duration")
            elif "severe" in q_text or "pain" in q_text or "scale" in q_text:
                self.asked_severity = True
                self.collected_info.add("severity")
            elif "time" in q_text or "when" in q_text or "prefer" in q_text:
                self.asked_preferred_time = True
                self.collected_info.add("preferred_time")
        
        elif action_type == "query_availability":
            self.queried_availability = True
        
        elif action_type == "offer_slot":
            self.offered_slot = True
        
        # Track revealed facts
        for fact in revealed_facts:
            self.collected_info.add(fact)


class RewardCalculator:
    """
    Computes dense step-wise rewards with order-aware and requirement-aware logic.
    
    Positive rewards:
    - Required question (in required_info_to_collect): +0.1
    - Useful question (relevant but not required): +0.05
    - Correct department: +0.15
    - Valid slot offer: +0.2
    - Successful booking: +0.5
    - Correct escalation (urgency >= 0.7): +0.4
    - Appropriate human escalation (urgency >= 0.5): +0.2
    - Safe memory recall: +0.1
    - Useful reflection: +0.15
    
    Negative rewards:
    - Redundant question: -0.1
    - Irrelevant question: -0.1
    - Unnecessary question (not required/useful): -0.02
    - Out-of-order question (booking before symptoms): -0.15
    - Offer slot without querying availability: -0.15
    - Unnecessary escalation (urgency < 0.5): -0.15
    - Invalid slot: -0.2
    - Ignoring urgency: -0.3
    - Privacy conversation: -0.2
    - Blocked PII: -1.0
    - Invalid action: -0.5
    """
    
    def __init__(self):
        self.conversation_state = ConversationState()
    
    def reset(self):
        """Reset conversation state for new episode."""
        self.conversation_state = ConversationState()
    
    def compute_step_reward(
        self,
        action: Dict[str, Any],
        patient_response: Any,
        action_history: List[Dict[str, Any]],
        scenario: Any,
        available_slots: Dict[str, List[str]],
    ) -> float:
        """
        Compute reward for a single step with order-aware and requirement-aware logic.
        
        Args:
            action: Current action dict
            patient_response: PatientResponse object
            action_history: Previous actions
            scenario: Current scenario
            available_slots: Current available slots
            
        Returns:
            Reward value (can be negative)
        """
        action_type = action.get("action_type", "")
        payload = action.get("payload", {})
        
        reward = 0.0
        
        # Get required info from scenario
        required_info = getattr(scenario, 'required_info_to_collect', [])
        
        # Ask question rewards with order-awareness
        if action_type == "ask_question":
            q_text = payload.get("question_text", "").lower()
            q_type = payload.get("question_type", "").lower()
            
            # Check for redundancy first
            if is_redundant_question(action, action_history):
                reward += REWARDS["redundant_question"]
            # Check for PII in question
            elif contains_pii_pattern(q_text):
                reward += REWARDS["privacy_conversation"]
            # Check for out-of-order questions (booking questions before symptoms)
            elif self._is_booking_question(q_text, q_type) and not self.conversation_state.asked_symptoms:
                # Penalize asking about booking/time before gathering symptoms
                reward -= 0.15
            # Check if question is in required_info_to_collect
            elif self._is_required_question(q_text, q_type, required_info):
                # Required question - good!
                reward += 0.1
                # Bonus if it revealed facts
                if patient_response.revealed_facts:
                    reward += 0.05
            # Check if question is useful but not required
            elif self._is_useful_question(q_text, q_type, scenario):
                reward += 0.05
            # Check for irrelevance
            elif is_irrelevant_question(action, scenario.patient_profile.symptoms):
                reward += REWARDS["irrelevant_question"]
            # Optional/unnecessary question - slight penalty for inefficiency
            else:
                reward -= 0.02  # Penalize unnecessary questions to encourage efficiency
            
            # Update conversation state
            self.conversation_state.update(action, patient_response.revealed_facts if hasattr(patient_response, 'revealed_facts') else [])
        
        # Query availability rewards
        elif action_type == "query_availability":
            department = payload.get("department", "")
            # Correct department
            if department == scenario.ground_truth.correct_department:
                reward += REWARDS["correct_department"]
        
        # Offer slot rewards
        elif action_type == "offer_slot":
            slot_id = payload.get("slot_id", "")
            
            # Check if availability was queried first (proper workflow)
            queried_availability = any(
                a.get("action_type") == "query_availability"
                for a in action_history
            )
            
            if not queried_availability:
                reward -= 0.15  # Penalty for offering without checking availability first
            
            # Check if slot is valid (exists in available slots)
            is_valid = any(slot_id in slots for slots in available_slots.values())
            
            if is_valid:
                reward += REWARDS["valid_slot_offer"]
                # Bonus if patient accepts (high slot_acceptance)
                if patient_response.slot_acceptance and patient_response.slot_acceptance > 0.7:
                    reward += 0.1  # Bonus for good match
            else:
                reward += REWARDS["invalid_slot"]
        
        # Confirm booking rewards
        elif action_type == "confirm_booking":
            reward += REWARDS["successful_booking"]
        
        # Escalation rewards
        elif action_type == "escalate_urgent":
            # Check if escalation was appropriate
            if scenario.ground_truth.urgency_level >= 0.7:
                reward += REWARDS["correct_escalation"]
            else:
                # Unnecessary escalation
                reward -= 0.15
        
        elif action_type == "escalate_human":
            # Check if escalation is appropriate based on urgency
            if scenario.ground_truth.urgency_level >= 0.5:
                reward += 0.2  # Appropriate escalation for medium/high urgency
            else:
                reward -= 0.15  # Unnecessary escalation for low urgency cases
        
        # Memory rewards (will be enhanced when memory vault is implemented)
        elif action_type == "recall_memory":
            reward += REWARDS["safe_memory_recall"]
        
        # Reflection rewards (will be enhanced when reflection is implemented)
        elif action_type == "trigger_reflection":
            reward += REWARDS["useful_reflection"]
        
        return reward
    
    def _is_booking_question(self, q_text: str, q_type: str) -> bool:
        """Check if question is about booking/scheduling before symptoms are gathered."""
        booking_keywords = [
            "time", "when", "prefer", "available", "slot", "appointment",
            "schedule", "book", "morning", "evening", "afternoon"
        ]
        return any(kw in q_text for kw in booking_keywords) or "time" in q_type or "availability" in q_type
    
    def _is_required_question(self, q_text: str, q_type: str, required_info: List[str]) -> bool:
        """Check if question asks about required information."""
        for req in required_info:
            req_lower = req.lower()
            if req_lower in q_text or req_lower in q_type:
                return True
            
            # Map common variations
            if req_lower == "symptoms" and any(kw in q_text for kw in ["symptom", "problem", "issue", "concern"]):
                return True
            if req_lower == "duration" and any(kw in q_text for kw in ["how long", "when did", "started"]):
                return True
            if req_lower == "severity" and any(kw in q_text for kw in ["severe", "pain", "scale", "bad"]):
                return True
            if req_lower == "preferred_time" and any(kw in q_text for kw in ["time", "when", "prefer"]):
                return True
        
        return False
    
    def _is_useful_question(self, q_text: str, q_type: str, scenario: Any) -> bool:
        """Check if question is useful but not required (e.g., clarifying questions)."""
        useful_keywords = [
            "clarify", "explain", "tell me more", "describe", "detail",
            "history", "previous", "medication", "allergy", "chronic"
        ]
        return any(kw in q_text for kw in useful_keywords)
    
    def compute_ignoring_urgency_penalty(
        self,
        turn_idx: int,
        urgency_level: float,
        escalated: bool,
    ) -> float:
        """
        Compute penalty for ignoring urgent cases.
        
        Args:
            turn_idx: Current turn number
            urgency_level: Patient urgency level
            escalated: Whether escalation has occurred
            
        Returns:
            Penalty (negative value)
        """
        if urgency_level >= 0.7 and not escalated and turn_idx >= 10:
            return REWARDS["ignoring_urgency"]
        return 0.0
