"""Tests for PrivacySafeMemoryVault."""

import pytest
from src.voiceclinicagent.memory_vault import PrivacySafeMemoryVault


def test_store_allowed_keys():
    """Test storing allowed keys succeeds."""
    vault = PrivacySafeMemoryVault()
    
    # Store allowed keys
    success, terminate, error = vault.store("preferences", "morning slots", "patient_001")
    assert success is True
    assert terminate is False
    assert error is None
    
    success, terminate, error = vault.store("symptoms_summary", "headache, fever", "patient_001")
    assert success is True
    assert terminate is False
    assert error is None


def test_store_blocked_keys():
    """Test storing blocked PII keys triggers termination."""
    vault = PrivacySafeMemoryVault()
    
    # Attempt to store blocked keys
    blocked_keys = ["name", "phone", "aadhaar", "insurance_id", "email", "dob"]
    
    for key in blocked_keys:
        success, terminate, error = vault.store(key, "some_value", "patient_001")
        assert success is False, f"Key {key} should fail"
        assert terminate is True, f"Key {key} should trigger termination"
        assert "BLOCKED" in error, f"Key {key} should have BLOCKED in error"


def test_store_invalid_keys():
    """Test storing invalid keys fails without termination."""
    vault = PrivacySafeMemoryVault()
    
    success, terminate, error = vault.store("invalid_key", "value", "patient_001")
    assert success is False
    assert terminate is False  # Invalid but not PII
    assert "Invalid memory key" in error


def test_recall_allowed_keys():
    """Test recalling allowed keys succeeds."""
    vault = PrivacySafeMemoryVault()
    
    # Store then recall
    vault.store("booking_notes", "prefers Dr. Smith", "patient_001")
    
    success, terminate, value, error = vault.recall("booking_notes", "patient_001")
    assert success is True
    assert terminate is False
    assert value == "prefers Dr. Smith"
    assert error is None


def test_recall_blocked_keys():
    """Test recalling blocked PII keys triggers termination."""
    vault = PrivacySafeMemoryVault()
    
    blocked_keys = ["name", "phone", "aadhaar", "insurance_id"]
    
    for key in blocked_keys:
        success, terminate, value, error = vault.recall(key, "patient_001")
        assert success is False
        assert terminate is True
        assert value is None
        assert "BLOCKED" in error


def test_recall_nonexistent_key():
    """Test recalling non-existent key fails gracefully."""
    vault = PrivacySafeMemoryVault()
    
    success, terminate, value, error = vault.recall("preferences", "patient_001")
    assert success is False
    assert terminate is False
    assert value is None
    assert "No memory stored" in error


def test_memory_isolation():
    """Test memory is isolated between patients."""
    vault = PrivacySafeMemoryVault()
    
    # Store for patient_001
    vault.store("preferences", "morning", "patient_001")
    
    # Store for patient_002
    vault.store("preferences", "evening", "patient_002")
    
    # Recall for each patient
    _, _, value1, _ = vault.recall("preferences", "patient_001")
    _, _, value2, _ = vault.recall("preferences", "patient_002")
    
    assert value1 == "morning"
    assert value2 == "evening"


def test_get_summary():
    """Test get_summary returns presence flags."""
    vault = PrivacySafeMemoryVault()
    
    # Store some keys
    vault.store("preferences", "value1", "patient_001")
    vault.store("symptoms_summary", "value2", "patient_001")
    
    summary = vault.get_summary("patient_001")
    
    # Check presence flags
    assert summary["preferences"] == 1
    assert summary["symptoms_summary"] == 1
    assert summary["booking_notes"] == 0
    assert summary["follow_up_needed"] == 0


def test_clear_patient():
    """Test clearing patient memory."""
    vault = PrivacySafeMemoryVault()
    
    # Store data
    vault.store("preferences", "morning", "patient_001")
    
    # Clear
    vault.clear_patient("patient_001")
    
    # Verify cleared
    summary = vault.get_summary("patient_001")
    assert all(v == 0 for v in summary.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
