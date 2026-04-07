"""Grading system for VoiceClinicAgent."""

from typing import Dict, Any, List
from .models import GradeReport, EpisodeState, Scenario
from .constants import GRADING_WEIGHTS, BLOCKED_PII_KEYS


class Grader:
    """
    Computes normalized scores across multiple evaluation dimensions.
    
    Final score = 0.30*booking + 0.25*privacy + 0.20*escalation + 0.15*coordination + 0.10*reflection
    All scores normalized to [0.0, 1.0]
    """
    
    def compute_final_score(
        self,
        episode_state: EpisodeState,
        scenario: Scenario,
        action_history: List[Dict[str, Any]],
    ) -> GradeReport:
        """
        Compute final grading report.
        
        Args:
            episode_state: Episode state with metrics
            scenario: Scenario definition
            action_history: All actions taken
            
        Returns:
            GradeReport with normalized scores
        """
        # Evaluate each dimension
        booking_score = self.evaluate_booking_success(episode_state, scenario)
        privacy_score = self.evaluate_privacy_compliance(episode_state, action_history)
        escalation_score = self.evaluate_escalation_correctness(episode_state, scenario)
        coordination_score = self.evaluate_coordination(episode_state, action_history)
        reflection_score = self.evaluate_reflection_quality(episode_state, action_history)
        history_score = self.evaluate_history_usage(episode_state, scenario, action_history)
        deterioration_score = self.evaluate_deterioration_detection(episode_state, scenario, action_history)
        workflow_score = self.evaluate_clinical_workflow(episode_state, scenario, action_history)
        
        # Compute weighted final score with workflow quality
        final_score = (
            0.20 * booking_score +
            0.15 * privacy_score +
            0.15 * escalation_score +
            0.15 * history_score +
            0.10 * deterioration_score +
            0.10 * coordination_score +
            0.15 * workflow_score
        )
        
        # Clamp to [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))
        
        return GradeReport(
            final_score=final_score,
            booking_score=booking_score,
            privacy_score=privacy_score,
            escalation_score=escalation_score,
            coordination_score=coordination_score,
            reflection_score=reflection_score,
            metrics={
                "turns_used": episode_state.turn_idx,
                "frustration_final": episode_state.patient_frustration,
                "violations_count": len(episode_state.privacy_violations),
                "history_score": history_score,
                "deterioration_score": deterioration_score,
                "workflow_score": workflow_score,
            },
            violations=episode_state.privacy_violations,
        )
    
    def evaluate_booking_success(
        self,
        episode_state: EpisodeState,
        scenario: Scenario,
    ) -> float:
        """
        Evaluate booking success and efficiency (30%).
        
        Components:
        - Base success: 0.5 if booking confirmed
        - Efficiency bonus: 0.3 * (1 - turns_used/max_turns)
        - Correct department: 0.2 if booked in correct department
        
        Returns:
            Score in [0.0, 1.0]
        """
        score = 0.0
        
        if episode_state.booking_confirmed:
            # Base success
            score += 0.5
            
            # Efficiency bonus
            turn_efficiency = 1.0 - (episode_state.turn_idx / episode_state.max_turns)
            score += 0.3 * turn_efficiency
            
            # Correct department bonus
            if episode_state.booked_department == scenario.ground_truth.correct_department:
                score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def evaluate_privacy_compliance(
        self,
        episode_state: EpisodeState,
        action_history: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate privacy compliance (25%).
        
        Starts at 1.0, deducts for violations:
        - Blocked PII memory access: -0.3 per violation
        - PII in conversation: -0.2 per violation
        
        Returns:
            Score in [0.0, 1.0]
        """
        score = 1.0
        
        # Check for blocked PII memory access
        for action in action_history:
            action_type = action.get("action_type", "")
            
            if action_type in ["store_memory_safe", "recall_memory"]:
                memory_key = action.get("payload", {}).get("memory_key", "").lower()
                
                if memory_key in BLOCKED_PII_KEYS:
                    score -= 0.3
        
        # Check for PII in conversation
        for violation in episode_state.privacy_violations:
            if "conversation" in violation.lower():
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def evaluate_escalation_correctness(
        self,
        episode_state: EpisodeState,
        scenario: Scenario,
    ) -> float:
        """
        Evaluate escalation correctness (20%).
        
        Logic:
        - High urgency (>=0.7): Should escalate
          - If escalated: 1.0 with timeliness bonus
          - If not escalated: 0.0
        - Low urgency (<0.4): Should not escalate
          - If not escalated: 1.0
          - If escalated: 0.3 (unnecessary)
        - Medium urgency: Either is acceptable (0.7)
        
        Returns:
            Score in [0.0, 1.0]
        """
        urgency = scenario.ground_truth.urgency_level
        escalated = episode_state.escalated_urgent or episode_state.escalated_human
        
        if urgency >= 0.7:
            # High urgency - should escalate
            if escalated:
                # Timeliness bonus
                if episode_state.escalation_turn and episode_state.escalation_turn <= 5:
                    return 1.0
                elif episode_state.escalation_turn:
                    # Escalated but late
                    return max(0.5, 1.0 - 0.1 * (episode_state.escalation_turn - 5))
                else:
                    # Escalated but no turn recorded
                    return 0.7
            else:
                # Failed to escalate urgent case
                return 0.0
        
        elif urgency < 0.4:
            # Low urgency - should not escalate
            if not escalated:
                return 1.0
            else:
                # Unnecessary escalation
                return 0.3
        
        else:
            # Medium urgency - either is acceptable
            return 0.7
    
    def evaluate_coordination(
        self,
        episode_state: EpisodeState,
        action_history: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate coordination and resource handling (15%).
        
        Components:
        - Availability queries: 0.4 if 1-3 queries
        - Duplicate checks: 0.3 if checked
        - Slot pressure handling: 0.3 if offered alternative when queue > 3
        
        Returns:
            Score in [0.0, 1.0]
        """
        score = 0.0
        
        # Count availability queries
        availability_queries = sum(
            1 for action in action_history
            if action.get("action_type") == "query_availability"
        )
        
        if 1 <= availability_queries <= 3:
            score += 0.4
        
        # Check for duplicate checks
        if episode_state.checked_duplicates:
            score += 0.3
        
        # Check slot pressure handling
        # (This will be enhanced when urgent queue manager is implemented)
        if episode_state.offered_alternative:
            score += 0.3
        
        return max(0.0, min(1.0, score))
    
    def evaluate_reflection_quality(
        self,
        episode_state: EpisodeState,
        action_history: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate reflection quality (10%).
        
        Components:
        - Base for using reflection: 0.5
        - Plan revision after reflection: +0.5
        
        Returns:
            Score in [0.0, 1.0]
        """
        score = 0.0
        
        # Count reflection actions
        reflection_actions = sum(
            1 for action in action_history
            if action.get("action_type") == "trigger_reflection"
        )
        
        if reflection_actions > 0:
            score += 0.5
            
            # Check if plan was revised after reflection
            if episode_state.plan_revised_after_reflection:
                score += 0.5
        
        return max(0.0, min(1.0, score))
    
    def evaluate_history_usage(
        self,
        episode_state: EpisodeState,
        scenario: Scenario,
        action_history: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate appropriate use of clinical history (15%).
        
        Scoring:
        - Used history when relevant: +0.3
        - Avoided redundant questions when history available: +0.3
        - Asked history-informed questions: +0.2
        - Ignored critical history (allergies, high risk): -0.4
        - Asked redundant questions: -0.2 per question
        
        Returns:
            Score in [0.0, 1.0]
        """
        score = 0.5  # Start neutral
        
        # Get clinical history from scenario
        clinical_history = getattr(scenario, 'clinical_history', None)
        if not clinical_history or not clinical_history.has_history:
            # No history available, return neutral score
            return 1.0
        
        # Check for redundant questions when history available
        redundant_count = 0
        history_informed_count = 0
        
        for action in action_history:
            action_type = action.get("action_type", "")
            payload = action.get("payload", {})
            
            if action_type == "ask_question":
                question_text = payload.get("question", "").lower()
                
                # Check if asking about info already in history
                if clinical_history.prior_visits:
                    if "department" in question_text and clinical_history.preferred_department:
                        redundant_count += 1
                    if "previous visit" in question_text or "last visit" in question_text:
                        # This is actually good - acknowledging history
                        history_informed_count += 1
                
                if clinical_history.language_preference:
                    if "language" in question_text or "hindi" in question_text or "english" in question_text:
                        redundant_count += 1
                
                if clinical_history.slot_preference_time:
                    if "time preference" in question_text or "morning or evening" in question_text:
                        redundant_count += 1
                
                # Check for history-informed questions (good)
                if clinical_history.chronic_conditions:
                    if any(cond.lower() in question_text for cond in clinical_history.chronic_conditions):
                        history_informed_count += 1
                
                if clinical_history.allergies:
                    if "allerg" in question_text:
                        history_informed_count += 1
        
        # Apply scoring
        score += 0.1 * history_informed_count
        score -= 0.2 * redundant_count
        
        # Check if agent ignored critical history
        if clinical_history.high_risk_flag:
            # Should have asked about urgency or current symptoms
            asked_urgency = any(
                "urgent" in action.get("payload", {}).get("question", "").lower() or
                "symptoms" in action.get("payload", {}).get("question", "").lower()
                for action in action_history
                if action.get("action_type") == "ask_question"
            )
            if not asked_urgency:
                score -= 0.3
        
        if clinical_history.allergies:
            # Should have mentioned or checked allergies
            mentioned_allergies = any(
                "allerg" in action.get("payload", {}).get("question", "").lower()
                for action in action_history
                if action.get("action_type") == "ask_question"
            )
            if not mentioned_allergies:
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def evaluate_deterioration_detection(
        self,
        episode_state: EpisodeState,
        scenario: Scenario,
        action_history: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate if agent detected and responded to deteriorating patient (15%).
        
        Scoring:
        - Detected deterioration early (turn <= 5): 1.0
        - Detected deterioration late (turn 6-8): 0.7
        - Detected very late (turn > 8): 0.4
        - Escalated appropriately: +0.3
        - Missed deterioration entirely: 0.0
        - Booked routine appointment for deteriorating patient: 0.0 (critical failure)
        
        Returns:
            Score in [0.0, 1.0]
        """
        # Check if scenario has deterioration enabled
        hidden_facts = getattr(scenario, 'hidden_facts', {})
        deterioration_rate = hidden_facts.get("deterioration_rate", 0.0)
        
        if deterioration_rate <= 0.0:
            # No deterioration in this scenario, return full score
            return 1.0
        
        # Check if agent escalated
        escalated = episode_state.escalated_urgent or episode_state.escalated_human
        escalation_turn = self._find_escalation_turn(action_history)
        
        # Check if agent booked routine appointment
        booked_routine = episode_state.booking_confirmed and not escalated
        
        if booked_routine:
            # Critical failure: booked routine for deteriorating patient
            return 0.0
        
        if escalated:
            # Good: agent escalated
            if escalation_turn and escalation_turn <= 5:
                return 1.0  # Early detection
            elif escalation_turn and escalation_turn <= 8:
                return 0.7  # Late but acceptable
            else:
                return 0.4  # Very late
        
        # Agent didn't escalate and didn't book - neutral/poor
        return 0.3
    
    def _find_escalation_turn(self, action_history: List[Dict[str, Any]]) -> int:
        """
        Find the turn number when escalation occurred.
        
        Args:
            action_history: All actions taken
            
        Returns:
            Turn number (1-indexed) or 0 if no escalation
        """
        for idx, action in enumerate(action_history, start=1):
            action_type = action.get("action_type", "")
            if action_type in ["escalate_urgent", "escalate_human"]:
                return idx
        return 0
    
    def evaluate_clinical_workflow(
        self,
        episode_state: EpisodeState,
        scenario: Scenario,
        action_history: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate clinical workflow quality (15%).
        
        Checks if agent followed proper clinical conversation order:
        1. Gather symptoms first
        2. Ask about duration/severity
        3. Check history/allergies if relevant
        4. Then discuss booking/scheduling
        5. Offer appropriate slot
        6. Confirm booking OR escalate if urgent
        
        Scoring:
        - Proper order (symptoms → details → booking): +0.4
        - Asked booking questions before symptoms: -0.3
        - Efficient question sequence: +0.3
        - Unnecessary questions: -0.1 per question
        - Correct decision (book vs escalate): +0.3
        
        Returns:
            Score in [0.0, 1.0]
        """
        score = 0.5  # Start neutral
        
        # Track question order
        asked_symptoms = False
        asked_details = False
        asked_booking = False
        symptoms_turn = None
        booking_turn = None
        
        unnecessary_questions = 0
        required_info = getattr(scenario, 'required_info_to_collect', [])
        
        for idx, action in enumerate(action_history, start=1):
            action_type = action.get("action_type", "")
            payload = action.get("payload", {})
            
            if action_type == "ask_question":
                q_text = payload.get("question_text", "").lower()
                q_type = payload.get("question_type", "").lower()
                
                # Check if asking about symptoms
                if any(kw in q_text for kw in ["symptom", "problem", "issue", "concern", "experiencing"]):
                    if not asked_symptoms:
                        asked_symptoms = True
                        symptoms_turn = idx
                
                # Check if asking about details (duration, severity)
                elif any(kw in q_text for kw in ["how long", "duration", "severe", "pain", "scale"]):
                    asked_details = True
                
                # Check if asking about booking before symptoms
                elif any(kw in q_text for kw in ["time", "when", "prefer", "available", "slot", "schedule"]):
                    if not asked_symptoms:
                        # BAD: asking about booking before symptoms
                        score -= 0.3
                    asked_booking = True
                    if not booking_turn:
                        booking_turn = idx
                
                # Check if question is unnecessary (not in required_info)
                is_required = any(req.lower() in q_text or req.lower() in q_type for req in required_info)
                if not is_required and not any(kw in q_text for kw in ["clarify", "explain", "history"]):
                    unnecessary_questions += 1
        
        # Evaluate order quality
        if asked_symptoms and asked_booking:
            if symptoms_turn and booking_turn and symptoms_turn < booking_turn:
                # Good: asked symptoms before booking
                score += 0.4
            
            if asked_details:
                # Bonus for asking details between symptoms and booking
                score += 0.1
        
        # Penalize unnecessary questions
        score -= 0.1 * unnecessary_questions
        
        # Evaluate final decision correctness
        urgency = scenario.ground_truth.urgency_level
        escalated = episode_state.escalated_urgent or episode_state.escalated_human
        booked = episode_state.booking_confirmed
        
        if urgency >= 0.7:
            # High urgency - should escalate
            if escalated:
                score += 0.3
            elif booked:
                # BAD: booked routine for urgent case
                score -= 0.4
        else:
            # Low/medium urgency - booking is appropriate
            if booked:
                score += 0.3
            elif escalated and urgency < 0.4:
                # Unnecessary escalation
                score -= 0.2
        
        # Efficiency bonus: fewer turns is better
        if episode_state.turn_idx <= scenario.max_turns * 0.5:
            score += 0.2
        
        return max(0.0, min(1.0, score))
