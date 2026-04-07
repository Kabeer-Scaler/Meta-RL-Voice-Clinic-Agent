"""Synthetic patient simulator for VoiceClinicAgent."""

import random
from typing import Dict, Any, List, Optional
from .models import Scenario, PatientProfile
from .constants import FRUSTRATION_REDUNDANT, FRUSTRATION_IRRELEVANT, FRUSTRATION_MAX
from .deterioration import DeteriorationEngine


class PatientResponse:
    """Patient response to agent action."""
    
    def __init__(
        self,
        text: str,
        revealed_facts: List[str],
        frustration_level: float,
        slot_acceptance: Optional[float] = None,
    ):
        self.text = text
        self.revealed_facts = revealed_facts
        self.frustration_level = frustration_level
        self.slot_acceptance = slot_acceptance


class SyntheticPatientSimulator:
    """
    Simulates realistic patient behavior.
    
    - Reveals information gradually based on questions
    - Tracks frustration from redundant/irrelevant questions
    - Supports English and Hindi-English mixed responses
    - Uses seeded RNG for deterministic behavior
    """
    
    def __init__(self):
        self.profile: Optional[PatientProfile] = None
        self.hidden_facts: Dict[str, str] = {}
        self.revealed_facts: Dict[str, str] = {}
        self.frustration: float = 0.0
        self.episode_rng: Optional[random.Random] = None
        self.deterioration_engine: Optional[DeteriorationEngine] = None
        self.current_turn: int = 0
        self.scenario: Optional[Scenario] = None
    
    def initialize(self, scenario: Scenario, episode_rng: random.Random) -> None:
        """
        Initialize patient simulator from scenario.
        
        Args:
            scenario: Scenario definition
            episode_rng: Seeded RNG for deterministic behavior
        """
        self.scenario = scenario
        self.profile = scenario.patient_profile
        self.hidden_facts = dict(self.profile.hidden_facts)
        self.revealed_facts = {}
        self.frustration = 0.0
        self.episode_rng = episode_rng
        self.current_turn = 0
        
        # Initialize deterioration engine if enabled
        hidden_facts_dict = scenario.hidden_facts if hasattr(scenario, 'hidden_facts') else {}
        deterioration_rate = hidden_facts_dict.get("deterioration_rate", 0.0)
        
        if deterioration_rate > 0.0:
            self.deterioration_engine = DeteriorationEngine(scenario, episode_rng)
        else:
            self.deterioration_engine = None
    
    def respond_to_action(
        self,
        action: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
    ) -> PatientResponse:
        """
        Generate patient response to agent action.
        
        Args:
            action: Agent action dict
            conversation_history: Previous actions
            
        Returns:
            PatientResponse with text and metadata
        """
        # Update turn counter
        self.current_turn = len(conversation_history) + 1
        
        # Update deterioration state if enabled
        deterioration_state = None
        if self.deterioration_engine:
            deterioration_state = self.deterioration_engine.update(self.current_turn)
        
        action_type = action.get("action_type", "")
        payload = action.get("payload", {})
        
        # Handle different action types
        if action_type == "ask_question":
            response = self._respond_to_question(payload, conversation_history)
        elif action_type == "offer_slot":
            response = self._respond_to_slot_offer(payload)
        elif action_type == "confirm_booking":
            response = self._respond_to_booking_confirmation(payload)
        elif action_type == "query_availability":
            response = PatientResponse(
                text="Okay, please check what's available.",
                revealed_facts=[],
                frustration_level=self.frustration,
            )
        elif action_type == "escalate_urgent":
            response = PatientResponse(
                text="Thank you for prioritizing my case.",
                revealed_facts=[],
                frustration_level=self.frustration,
            )
        else:
            response = PatientResponse(
                text="I understand.",
                revealed_facts=[],
                frustration_level=self.frustration,
            )
        
        # Add deterioration cues to response if applicable
        if deterioration_state and deterioration_state["new_symptoms"]:
            response = self._add_deterioration_cues(response, deterioration_state)
        
        return response
    
    def _respond_to_question(
        self,
        payload: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
    ) -> PatientResponse:
        """Handle ask_question action."""
        question_type = payload.get("question_type", "")
        question_text = payload.get("question_text", "")
        
        # Check for redundancy
        is_redundant = self._is_redundant(question_type, conversation_history)
        if is_redundant:
            self.frustration = min(FRUSTRATION_MAX, self.frustration + FRUSTRATION_REDUNDANT)
            return PatientResponse(
                text="I already told you that. Please pay attention.",
                revealed_facts=[],
                frustration_level=self.frustration,
            )
        
        # Check for irrelevance
        is_irrelevant = self._is_irrelevant(question_text)
        if is_irrelevant:
            self.frustration = min(FRUSTRATION_MAX, self.frustration + FRUSTRATION_IRRELEVANT)
            return PatientResponse(
                text="I don't think that's relevant to my appointment.",
                revealed_facts=[],
                frustration_level=self.frustration,
            )
        
        # Reveal fact if available
        if question_type in self.hidden_facts:
            fact_value = self.hidden_facts[question_type]
            
            # Compute revelation probability based on vagueness and frustration
            revelation_prob = self._compute_revelation_probability(len(conversation_history))
            
            if self.episode_rng.random() < revelation_prob:
                # Reveal the fact
                self.revealed_facts[question_type] = fact_value
                del self.hidden_facts[question_type]
                
                # Generate response with language mixing if applicable
                response_text = self._generate_response_text(fact_value)
                
                return PatientResponse(
                    text=response_text,
                    revealed_facts=[question_type],
                    frustration_level=self.frustration,
                )
            else:
                # Give vague response
                return PatientResponse(
                    text=self._generate_vague_response(question_type),
                    revealed_facts=[],
                    frustration_level=self.frustration,
                )
        else:
            # Question type not in hidden facts - check clinical history
            response_text = self._check_clinical_history_for_answer(question_type, question_text)
            
            return PatientResponse(
                text=response_text,
                revealed_facts=[],
                frustration_level=self.frustration,
            )
    
    def _respond_to_slot_offer(self, payload: Dict[str, Any]) -> PatientResponse:
        """Handle offer_slot action."""
        slot_id = payload.get("slot_id", "")
        
        # Simple preference matching based on time of day
        preference_match = self._evaluate_slot_preference(slot_id)
        
        if preference_match > 0.7:
            response_text = "Yes, that time works perfectly for me!"
        elif preference_match > 0.4:
            response_text = "That could work, but do you have anything earlier?"
        else:
            response_text = "That time doesn't work for me. Do you have other options?"
        
        return PatientResponse(
            text=response_text,
            revealed_facts=[],
            frustration_level=self.frustration,
            slot_acceptance=preference_match,
        )
    
    def _respond_to_booking_confirmation(self, payload: Dict[str, Any]) -> PatientResponse:
        """Handle confirm_booking action."""
        return PatientResponse(
            text="Thank you! I've noted down the appointment details.",
            revealed_facts=[],
            frustration_level=self.frustration,
        )
    
    def _is_redundant(self, question_type: str, history: List[Dict[str, Any]]) -> bool:
        """Check if question type was already asked."""
        for past_action in history:
            if past_action.get("action_type") == "ask_question":
                past_question_type = past_action.get("payload", {}).get("question_type", "")
                if past_question_type == question_type:
                    return True
        return False
    
    def _is_irrelevant(self, question_text: str) -> bool:
        """Check if question is irrelevant."""
        irrelevant_keywords = ["weather", "sports", "politics", "food", "hobby"]
        question_lower = question_text.lower()
        return any(keyword in question_lower for keyword in irrelevant_keywords)
    
    def _compute_revelation_probability(self, turn_count: int) -> float:
        """
        Compute probability of revealing information.
        
        Higher probability as conversation progresses.
        Lower probability with higher vagueness or frustration.
        """
        base_prob = 0.7
        
        # Increase with turn count
        turn_bonus = min(0.2, turn_count * 0.05)
        
        # Decrease with vagueness
        vagueness_penalty = self.profile.vagueness_level * 0.3
        
        # Decrease with frustration
        frustration_penalty = self.frustration * 0.2
        
        prob = base_prob + turn_bonus - vagueness_penalty - frustration_penalty
        return max(0.1, min(1.0, prob))
    
    def _generate_response_text(self, fact_value: str) -> str:
        """Generate response text with optional language mixing."""
        if self.profile.language_mix > 0.0 and self.episode_rng.random() < self.profile.language_mix:
            # Add some Hindi mixing (simple simulation)
            return self._add_hindi_mixing(fact_value)
        return fact_value
    
    def _add_hindi_mixing(self, text: str) -> str:
        """Add simple Hindi-English mixing."""
        # Simple simulation: add Hindi words at start/end
        hindi_prefixes = ["Haan", "Theek hai", "Achha"]
        hindi_suffixes = ["hai", "hota hai", "ho raha hai"]
        
        prefix = self.episode_rng.choice(hindi_prefixes)
        suffix = self.episode_rng.choice(hindi_suffixes)
        
        return f"{prefix}, {text} {suffix}."
    
    def _generate_vague_response(self, question_type: str) -> str:
        """Generate vague response when not revealing fact."""
        vague_responses = [
            "I'm not exactly sure...",
            "It's hard to say...",
            "Maybe, I think...",
            "I don't remember exactly...",
        ]
        return self.episode_rng.choice(vague_responses)
    
    def _evaluate_slot_preference(self, slot_id: str) -> float:
        """
        Evaluate how well slot matches patient preferences.
        
        Returns:
            Match score in [0.0, 1.0]
        """
        # Extract time from slot_id (format: dept_date_time_doctor)
        parts = slot_id.split("_")
        if len(parts) < 3:
            return 0.5  # Default neutral
        
        time_str = parts[2]  # e.g., "09:00"
        hour = int(time_str.split(":")[0])
        
        preferred_time = self.profile.preferences.get("preferred_time", "")
        
        if preferred_time == "morning" and 6 <= hour < 12:
            return 0.9
        elif preferred_time == "afternoon" and 12 <= hour < 17:
            return 0.9
        elif preferred_time == "evening" and 17 <= hour < 21:
            return 0.9
        else:
            return 0.5  # Acceptable but not preferred
    
    def get_hidden_facts(self) -> Dict[str, str]:
        """Get remaining hidden facts."""
        return dict(self.hidden_facts)
    
    def get_revealed_facts(self) -> Dict[str, str]:
        """Get revealed facts."""
        return dict(self.revealed_facts)
    
    def _add_deterioration_cues(
        self,
        response: PatientResponse,
        deterioration_state: Dict[str, Any]
    ) -> PatientResponse:
        """
        Add deterioration cues to patient response.
        
        Args:
            response: Original patient response
            deterioration_state: Current deterioration state
            
        Returns:
            Modified response with deterioration cues
        """
        new_symptoms = deterioration_state.get("new_symptoms", [])
        stage = deterioration_state.get("deterioration_stage", "stable")
        
        if not new_symptoms:
            return response
        
        # Add symptom mentions to response
        symptom_text = ", ".join(new_symptoms)
        
        # Create deterioration message based on stage
        if stage == "mild_worsening":
            cue = f"Actually, I'm starting to feel {symptom_text}."
        elif stage == "moderate_worsening":
            cue = f"I need to mention - I'm now experiencing {symptom_text}. It's getting worse."
        elif stage == "critical":
            cue = f"I'm really worried - I have {symptom_text} now. This is getting serious."
        else:
            cue = f"By the way, I'm also feeling {symptom_text}."
        
        # Append to original response
        modified_text = f"{response.text} {cue}"
        
        return PatientResponse(
            text=modified_text,
            revealed_facts=response.revealed_facts + new_symptoms,
            frustration_level=response.frustration_level,
            slot_acceptance=response.slot_acceptance,
        )
    
    def get_deterioration_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current deterioration state.
        
        Returns:
            Deterioration state dict or None if not enabled
        """
        if self.deterioration_engine:
            return self.deterioration_engine.get_current_state()
        return None

    
    def _check_clinical_history_for_answer(self, question_type: str, question_text: str) -> str:
        """
        Check clinical history for answers to questions not in hidden_facts.
        
        Args:
            question_type: Type of question asked
            question_text: Full question text
            
        Returns:
            Appropriate response based on clinical history
        """
        if not self.scenario or not hasattr(self.scenario, 'clinical_history'):
            return "I'm not sure about that."
        
        clinical_history = self.scenario.clinical_history
        question_lower = question_text.lower()
        
        # Check for medication questions
        if "medication" in question_lower or "medicine" in question_lower or question_type == "medications":
            if clinical_history.current_medications:
                meds = ", ".join(clinical_history.current_medications)
                return f"Yes, I'm currently taking {meds}."
            else:
                return "No, I'm not taking any medications currently."
        
        # Check for allergy questions
        if "allerg" in question_lower or question_type == "allergies":
            if clinical_history.allergies:
                allergies = ", ".join(clinical_history.allergies)
                return f"Yes, I'm allergic to {allergies}."
            else:
                return "No, I don't have any known allergies."
        
        # Check for chronic condition questions
        if "chronic" in question_lower or "condition" in question_lower or "disease" in question_lower:
            if clinical_history.chronic_conditions:
                conditions = ", ".join(clinical_history.chronic_conditions)
                return f"I have {conditions}."
            else:
                return "No, I don't have any chronic conditions."
        
        # Check for prior visit questions
        if "previous" in question_lower or "before" in question_lower or "last visit" in question_lower or question_type == "history":
            if clinical_history.prior_visits:
                last_visit = clinical_history.prior_visits[-1]
                return f"Yes, I visited {last_visit.get('department', 'the clinic')} on {last_visit.get('date', 'recently')} for {last_visit.get('reason', 'treatment')}."
            else:
                return "No, this is my first visit."
        
        # Check for age questions
        if "age" in question_lower or "old" in question_lower or question_type == "age":
            # Age not typically in clinical history, but patient should know
            return "I'm 35 years old."  # Default reasonable age
        
        # Check for booking-for questions (parent booking)
        if "booking for" in question_lower or "yourself" in question_lower or question_type == "booking_for":
            if hasattr(self.profile, 'parent_booking') and self.profile.parent_booking:
                return "I'm booking for my child."
            else:
                return "I'm booking for myself."
        
        # Default: genuinely don't know
        return "I'm not sure about that."
