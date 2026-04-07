"""
Golden-hour deterioration layer for VoiceClinicAgent.

Simulates patient deterioration over time for high-risk scenarios.
"""

import random
from typing import Dict, Any, List, Optional


class DeteriorationEngine:
    """
    Simulates patient deterioration over time.
    
    For high-risk scenarios (Task 3), symptoms worsen gradually:
    - Turn 1-3: Baseline symptoms
    - Turn 4-6: Mild worsening (if not escalated)
    - Turn 7-10: Moderate worsening
    - Turn 10+: Severe worsening (critical)
    
    This creates a "golden hour" dynamic where the agent must detect
    worsening symptoms and escalate appropriately.
    """
    
    def __init__(self, scenario: Any, episode_rng: random.Random):
        """
        Initialize deterioration engine.
        
        Args:
            scenario: Scenario object with hidden_facts
            episode_rng: Seeded random number generator
        """
        self.scenario = scenario
        self.rng = episode_rng
        
        # Deterioration parameters from scenario
        hidden_facts = scenario.hidden_facts if hasattr(scenario, 'hidden_facts') else {}
        self.baseline_urgency = hidden_facts.get("urgency_score", 0.3)
        self.deterioration_rate = hidden_facts.get("deterioration_rate", 0.0)
        self.deterioration_enabled = self.deterioration_rate > 0.0
        
        # Symptom progression map (turn -> new symptoms)
        self.symptom_progression = hidden_facts.get("symptom_progression", {})
        
        # Current state
        self.current_urgency = self.baseline_urgency
        self.turn_count = 0
        self.symptom_severity = 0.5
        self.accumulated_symptoms = []
    
    def update(self, turn_idx: int) -> Dict[str, Any]:
        """
        Update deterioration state for current turn.
        
        Args:
            turn_idx: Current turn number
            
        Returns:
            Dict with:
                - urgency: Current urgency score [0.0, 1.0]
                - severity: Current symptom severity [0.0, 1.0]
                - new_symptoms: List of new symptoms appearing this turn
                - deterioration_stage: "stable", "mild_worsening", "moderate_worsening", "critical"
                - should_auto_escalate: True if deterioration is critical
        """
        if not self.deterioration_enabled:
            return {
                "urgency": self.baseline_urgency,
                "severity": self.symptom_severity,
                "new_symptoms": [],
                "deterioration_stage": "stable",
                "should_auto_escalate": False
            }
        
        self.turn_count = turn_idx
        
        # Increase urgency over time
        self.current_urgency = min(1.0, self.baseline_urgency + (self.deterioration_rate * turn_idx))
        
        # Increase symptom severity
        self.symptom_severity = min(1.0, 0.5 + (0.05 * turn_idx))
        
        # Check for new symptoms based on turn
        new_symptoms = self._get_new_symptoms(turn_idx)
        
        return {
            "urgency": self.current_urgency,
            "severity": self.symptom_severity,
            "new_symptoms": new_symptoms,
            "deterioration_stage": self._get_stage(),
            "should_auto_escalate": self.should_auto_escalate()
        }
    
    def _get_new_symptoms(self, turn_idx: int) -> List[str]:
        """
        Get new symptoms that appear at this turn.
        
        Args:
            turn_idx: Current turn number
            
        Returns:
            List of new symptom descriptions
        """
        new_symptoms = []
        
        # Check symptom progression map
        turn_key = f"turn_{turn_idx}"
        if turn_key in self.symptom_progression:
            symptoms = self.symptom_progression[turn_key]
            if isinstance(symptoms, list):
                new_symptoms.extend(symptoms)
            else:
                new_symptoms.append(symptoms)
        
        # Default progression based on urgency thresholds
        if not new_symptoms:
            if turn_idx == 5 and self.current_urgency >= 0.8:
                new_symptoms.append("increased sweating")
            elif turn_idx == 8 and self.current_urgency >= 0.9:
                new_symptoms.append("difficulty breathing")
            elif turn_idx == 10 and self.current_urgency >= 0.95:
                new_symptoms.append("confusion")
        
        # Track accumulated symptoms
        self.accumulated_symptoms.extend(new_symptoms)
        
        return new_symptoms
    
    def _get_stage(self) -> str:
        """
        Get current deterioration stage.
        
        Returns:
            Stage name: "stable", "mild_worsening", "moderate_worsening", "critical"
        """
        if self.current_urgency < 0.6:
            return "stable"
        elif self.current_urgency < 0.8:
            return "mild_worsening"
        elif self.current_urgency < 0.95:
            return "moderate_worsening"
        else:
            return "critical"
    
    def should_auto_escalate(self) -> bool:
        """
        Check if deterioration is so severe that auto-escalation should occur.
        
        Returns:
            True if patient condition is critical and requires immediate escalation
        """
        return self.current_urgency >= 0.95 and self.turn_count >= 10
    
    def get_deterioration_cues(self) -> List[str]:
        """
        Get verbal cues that indicate deterioration for patient responses.
        
        Returns:
            List of phrases to add to patient responses
        """
        stage = self._get_stage()
        
        cues = {
            "stable": [],
            "mild_worsening": [
                "It's getting a bit worse",
                "I'm feeling more uncomfortable now"
            ],
            "moderate_worsening": [
                "The pain is getting worse",
                "I'm having trouble breathing",
                "I feel very weak"
            ],
            "critical": [
                "I can barely breathe",
                "The pain is unbearable",
                "I feel like I'm going to pass out"
            ]
        }
        
        return cues.get(stage, [])
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current deterioration state summary.
        
        Returns:
            Dict with all current state information
        """
        return {
            "enabled": self.deterioration_enabled,
            "turn": self.turn_count,
            "baseline_urgency": self.baseline_urgency,
            "current_urgency": self.current_urgency,
            "deterioration_rate": self.deterioration_rate,
            "severity": self.symptom_severity,
            "stage": self._get_stage(),
            "accumulated_symptoms": self.accumulated_symptoms.copy(),
            "should_auto_escalate": self.should_auto_escalate()
        }
