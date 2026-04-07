"""Tests for VoiceClinicEnvironment core functionality."""

import pytest
from src.voiceclinicagent.env import VoiceClinicEnvironment
from src.voiceclinicagent.api_models import VoiceClinicAction


def test_reset_returns_valid_observation():
    """Test that reset returns a valid observation."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Check observation structure
    assert obs.done is False
    assert obs.reward is None
    assert obs.task_level == "easy"
    assert obs.turn_idx == 0
    assert obs.max_turns == 15
    assert isinstance(obs.patient_message, str)
    assert isinstance(obs.conversation_summary, str)
    assert isinstance(obs.patient_flags, dict)
    assert isinstance(obs.clinic_state, dict)
    assert isinstance(obs.reflection_token, list)
    assert len(obs.reflection_token) == 4


def test_step_updates_state():
    """Test that step updates environment state."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    initial_turn = obs.turn_idx
    
    # Take a step
    action = VoiceClinicAction(
        action_type="ask_question",
        payload={
            "question_type": "symptoms",
            "question_text": "What symptoms are you experiencing?"
        }
    )
    
    obs = env.step(action)
    
    # Turn should increment
    assert obs.turn_idx == initial_turn + 1
    assert obs.reward is not None
    assert isinstance(obs.reward, float)


def test_state_returns_final_score_when_done():
    """Test that state property returns final_score when episode is done."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Run until done
    max_steps = 20
    for _ in range(max_steps):
        if obs.done:
            break
        
        # Simple action
        action = VoiceClinicAction(
            action_type="end_call",
            payload={}
        )
        obs = env.step(action)
    
    # Get state
    state = env.state
    
    # If done, should have final_score
    if obs.done:
        assert state.final_score is not None
        assert 0.0 <= state.final_score <= 1.0


def test_multiple_resets():
    """Test that environment can be reset multiple times."""
    env = VoiceClinicEnvironment()
    
    # First episode
    obs1 = env.reset(seed=42, task_id="easy_001")
    assert obs1.turn_idx == 0
    
    # Take a step
    action = VoiceClinicAction(
        action_type="ask_question",
        payload={"question_type": "symptoms", "question_text": "What's wrong?"}
    )
    obs1 = env.step(action)
    assert obs1.turn_idx == 1
    
    # Reset again
    obs2 = env.reset(seed=43, task_id="medium_001")
    assert obs2.turn_idx == 0
    assert obs2.task_level == "medium"


def test_different_task_ids():
    """Test that environment works with different task IDs."""
    env = VoiceClinicEnvironment()
    
    task_ids = ["easy_001", "medium_001", "hard_001"]
    
    for task_id in task_ids:
        obs = env.reset(seed=42, task_id=task_id)
        assert obs.done is False
        assert obs.turn_idx == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
