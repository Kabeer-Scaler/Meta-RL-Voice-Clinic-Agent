"""Tests for privacy compliance in VoiceClinicAgent."""

import pytest
from src.voiceclinicagent.env import VoiceClinicEnvironment
from src.voiceclinicagent.api_models import VoiceClinicAction


def test_blocked_pii_store_terminates():
    """Test that storing blocked PII keys terminates episode."""
    env = VoiceClinicEnvironment()
    
    # Reset to easy task
    obs = env.reset(seed=42, task_id="easy_001")
    assert obs.done is False
    
    # Attempt to store blocked PII key
    action = VoiceClinicAction(
        action_type="store_memory_safe",
        payload={"memory_key": "phone", "value": "1234567890"}
    )
    
    obs = env.step(action)
    
    # Should terminate episode
    assert obs.done is True
    assert "PRIVACY VIOLATION" in obs.patient_message


def test_blocked_pii_recall_terminates():
    """Test that recalling blocked PII keys terminates episode."""
    env = VoiceClinicEnvironment()
    
    # Reset to easy task
    obs = env.reset(seed=42, task_id="easy_001")
    assert obs.done is False
    
    # Attempt to recall blocked PII key
    action = VoiceClinicAction(
        action_type="recall_memory",
        payload={"memory_key": "aadhaar"}
    )
    
    obs = env.step(action)
    
    # Should terminate episode
    assert obs.done is True
    assert "PRIVACY VIOLATION" in obs.patient_message


def test_allowed_keys_succeed():
    """Test that allowed keys work without termination."""
    env = VoiceClinicEnvironment()
    
    # Reset to easy task
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Store allowed key
    action = VoiceClinicAction(
        action_type="store_memory_safe",
        payload={"memory_key": "preferences", "value": "morning slots"}
    )
    
    obs = env.step(action)
    
    # Should NOT terminate
    assert obs.done is False
    assert "stored" in obs.patient_message.lower() or "success" in obs.patient_message.lower()
    
    # Recall allowed key
    action = VoiceClinicAction(
        action_type="recall_memory",
        payload={"memory_key": "preferences"}
    )
    
    obs = env.step(action)
    
    # Should NOT terminate
    assert obs.done is False
    assert "morning slots" in obs.patient_message


def test_conversation_pii_negative_reward_but_continues():
    """Test that asking PII questions gives negative reward but doesn't terminate."""
    env = VoiceClinicEnvironment()
    
    # Reset to easy task
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Ask question about name (PII in conversation)
    action = VoiceClinicAction(
        action_type="ask_question",
        payload={
            "question_type": "personal_info",
            "question_text": "What is your full name?"
        }
    )
    
    obs = env.step(action)
    
    # Should continue (not terminate) but may have negative reward
    assert obs.done is False
    # Privacy risk mask should flag name
    assert obs.privacy_risk_mask.get("name", 0) == 1


def test_memory_vault_summary_in_observation():
    """Test that memory vault summary appears in observation."""
    env = VoiceClinicEnvironment()
    
    # Reset
    obs = env.reset(seed=42, task_id="easy_001")
    
    # Initially empty
    assert isinstance(obs.memory_vault_summary, dict)
    
    # Store something
    action = VoiceClinicAction(
        action_type="store_memory_safe",
        payload={"memory_key": "symptoms_summary", "value": "fever and cough"}
    )
    
    obs = env.step(action)
    
    # Summary should show stored key
    assert obs.memory_vault_summary.get("symptoms_summary") == 1
    assert obs.memory_vault_summary.get("preferences") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
