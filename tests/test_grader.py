"""Tests for grading system."""

import pytest
from src.voiceclinicagent.env import VoiceClinicEnvironment
from src.voiceclinicagent.api_models import VoiceClinicAction


def test_final_score_in_range():
    """Test that final_score is always in [0.0, 1.0]."""
    env = VoiceClinicEnvironment()
    
    task_ids = ["easy_001", "medium_001", "hard_001"]
    
    for task_id in task_ids:
        obs = env.reset(seed=42, task_id=task_id)
        
        # Run episode to completion
        max_steps = 30
        for _ in range(max_steps):
            if obs.done:
                break
            
            # Simple action to end episode
            action = VoiceClinicAction(
                action_type="end_call",
                payload={}
            )
            obs = env.step(action)
        
        # Get final state
        state = env.state
        
        # Final score must be in [0.0, 1.0]
        if state.final_score is not None:
            assert 0.0 <= state.final_score <= 1.0, f"Task {task_id} final_score out of range: {state.final_score}"


def test_component_scores_in_range():
    """Test that all component scores are in [0.0, 1.0]."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Run to completion
    max_steps = 20
    for _ in range(max_steps):
        if obs.done:
            break
        
        action = VoiceClinicAction(
            action_type="end_call",
            payload={}
        )
        obs = env.step(action)
    
    # Get state with grade report
    state = env.state
    
    if state.grade_report:
        grade_report = state.grade_report
        
        # Check component scores
        components = [
            "booking_success",
            "privacy_compliance",
            "escalation_correctness",
            "coordination_quality",
            "reflection_quality",
        ]
        
        for component in components:
            if component in grade_report:
                score = grade_report[component]
                assert 0.0 <= score <= 1.0, f"{component} out of range: {score}"


def test_weighted_sum_correct():
    """Test that final_score is correct weighted sum of components."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Complete episode
    action = VoiceClinicAction(
        action_type="end_call",
        payload={}
    )
    obs = env.step(action)
    
    state = env.state
    
    if state.grade_report and state.final_score is not None:
        report = state.grade_report
        
        # Compute expected weighted sum
        expected = (
            0.30 * report.get("booking_success", 0.0) +
            0.25 * report.get("privacy_compliance", 0.0) +
            0.20 * report.get("escalation_correctness", 0.0) +
            0.15 * report.get("coordination_quality", 0.0) +
            0.10 * report.get("reflection_quality", 0.0)
        )
        
        # Should match final_score (within floating point tolerance)
        assert abs(state.final_score - expected) < 0.001


def test_successful_booking_gives_high_score():
    """Test that successful booking gives a reasonable score."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Simulate successful booking flow
    actions = [
        VoiceClinicAction(
            action_type="ask_question",
            payload={"question_type": "symptoms", "question_text": "What symptoms?"}
        ),
        VoiceClinicAction(
            action_type="query_availability",
            payload={"department": "dermatology"}
        ),
        VoiceClinicAction(
            action_type="offer_slot",
            payload={"slot_id": "DERM_2026_04_03_17_00"}
        ),
        VoiceClinicAction(
            action_type="confirm_booking",
            payload={"slot_id": "DERM_2026_04_03_17_00", "patient_confirmation": True}
        ),
    ]
    
    for action in actions:
        obs = env.step(action)
        if obs.done:
            break
    
    state = env.state
    
    # Should have a reasonable final score
    if state.final_score is not None:
        assert state.final_score > 0.3  # At least some credit for booking


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
