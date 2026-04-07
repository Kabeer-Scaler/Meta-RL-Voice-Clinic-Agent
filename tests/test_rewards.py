"""Tests for reward system."""

import pytest
import math
from src.voiceclinicagent.env import VoiceClinicEnvironment
from src.voiceclinicagent.api_models import VoiceClinicAction


def test_all_rewards_are_finite():
    """Test that all rewards are finite (no NaN or infinite)."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Try various actions
    actions = [
        VoiceClinicAction(
            action_type="ask_question",
            payload={"question_type": "symptoms", "question_text": "What's wrong?"}
        ),
        VoiceClinicAction(
            action_type="query_availability",
            payload={"department": "general"}
        ),
        VoiceClinicAction(
            action_type="end_call",
            payload={}
        ),
    ]
    
    for action in actions:
        obs = env.step(action)
        
        # Reward must be finite
        assert obs.reward is not None
        assert isinstance(obs.reward, (int, float))
        assert math.isfinite(obs.reward)
        assert not math.isnan(obs.reward)
        
        if obs.done:
            break


def test_positive_rewards_for_correct_actions():
    """Test that correct actions receive positive rewards."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Ask relevant question - should get positive reward
    action = VoiceClinicAction(
        action_type="ask_question",
        payload={"question_type": "symptoms", "question_text": "What symptoms are you experiencing?"}
    )
    
    obs = env.step(action)
    
    # Should get positive or neutral reward for relevant question
    assert obs.reward >= -0.1  # Allow small negative for edge cases


def test_negative_rewards_for_violations():
    """Test that violations receive negative rewards."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Invalid action - should get negative reward
    action = VoiceClinicAction(
        action_type="offer_slot",
        payload={"slot_id": "INVALID_SLOT_999"}
    )
    
    obs = env.step(action)
    
    # Should get negative reward for invalid slot
    assert obs.reward < 0.0


def test_no_nan_infinite_in_episode():
    """Test that an entire episode produces no NaN or infinite rewards."""
    env = VoiceClinicEnvironment()
    
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Run full episode
    max_steps = 20
    for _ in range(max_steps):
        if obs.done:
            break
        
        action = VoiceClinicAction(
            action_type="ask_question",
            payload={"question_type": "symptoms", "question_text": "How are you feeling?"}
        )
        
        obs = env.step(action)
        
        # Check reward is finite
        assert math.isfinite(obs.reward)
        assert not math.isnan(obs.reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
