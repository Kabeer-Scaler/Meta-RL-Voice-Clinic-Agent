"""Tests for ReflectionEngine."""

import pytest
from src.voiceclinicagent.reflection import ReflectionEngine
from src.voiceclinicagent.scenario_loader import ScenarioLoader


def test_reflection_token_format():
    """Test that reflection token has correct format."""
    engine = ReflectionEngine()
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    scenario = loader.get_scenario("easy_001")
    
    token = engine.compute_reflection_token(
        scenario=scenario,
        action_history=[],
        revealed_facts={},
        available_slots={"general": ["slot_1", "slot_2"]},
        urgent_queue_length=0,
        turn_idx=0,
        max_turns=15,
    )
    
    # Should be 4-element list
    assert isinstance(token, list)
    assert len(token) == 4
    
    # All elements should be floats in [0.0, 1.0]
    for val in token:
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


def test_need_escalation_high_urgency():
    """Test that high urgency scenarios trigger need_escalation."""
    engine = ReflectionEngine()
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    # Hard scenarios have high urgency
    scenario = loader.get_scenario("hard_001")
    
    token = engine.compute_reflection_token(
        scenario=scenario,
        action_history=[],
        revealed_facts={},
        available_slots={"general": ["slot_1"]},
        urgent_queue_length=3,
        turn_idx=5,
        max_turns=25,
    )
    
    # need_escalation (index 0) should be elevated
    assert token[0] > 0.45


def test_info_missing_signal():
    """Test that info_missing reflects unrevealed facts."""
    engine = ReflectionEngine()
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    scenario = loader.get_scenario("easy_001")
    
    # No facts revealed yet
    token = engine.compute_reflection_token(
        scenario=scenario,
        action_history=[],
        revealed_facts={},
        available_slots={"general": ["slot_1", "slot_2"]},
        urgent_queue_length=0,
        turn_idx=0,
        max_turns=15,
    )
    
    # info_missing (index 1) should be high
    assert token[1] > 0.0


def test_privacy_risk_from_pii_questions():
    """Test that PII questions increase privacy_risk_high."""
    engine = ReflectionEngine()
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    scenario = loader.get_scenario("easy_001")
    
    # Action history with PII questions
    action_history = [
        {
            "action_type": "ask_question",
            "payload": {
                "question_type": "personal_info",
                "question_text": "What is your phone number?"
            }
        },
        {
            "action_type": "ask_question",
            "payload": {
                "question_type": "personal_info",
                "question_text": "What is your name?"
            }
        },
    ]
    
    token = engine.compute_reflection_token(
        scenario=scenario,
        action_history=action_history,
        revealed_facts={},
        available_slots={"general": ["slot_1"]},
        urgent_queue_length=0,
        turn_idx=2,
        max_turns=15,
    )
    
    # privacy_risk_high (index 2) should be elevated
    assert token[2] > 0.5


def test_slot_pressure_high_queue():
    """Test that high urgent queue increases slot_pressure."""
    engine = ReflectionEngine()
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    scenario = loader.get_scenario("easy_001")
    
    # High urgent queue, few slots
    token = engine.compute_reflection_token(
        scenario=scenario,
        action_history=[],
        revealed_facts={},
        available_slots={"general": ["slot_1"]},
        urgent_queue_length=5,
        turn_idx=0,
        max_turns=15,
    )
    
    # slot_pressure (index 3) should be high
    assert token[3] > 0.5


def test_slot_pressure_no_slots():
    """Test that no available slots gives maximum slot_pressure."""
    engine = ReflectionEngine()
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    scenario = loader.get_scenario("easy_001")
    
    # No slots available
    token = engine.compute_reflection_token(
        scenario=scenario,
        action_history=[],
        revealed_facts={},
        available_slots={},
        urgent_queue_length=3,
        turn_idx=0,
        max_turns=15,
    )
    
    # slot_pressure (index 3) should be 1.0
    assert token[3] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
