"""Tests for scenario loading and validation."""

import pytest
from src.voiceclinicagent.scenario_loader import ScenarioLoader


def test_all_scenarios_load():
    """Test that all scenarios load without errors."""
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    # Should have at least 3 scenarios (one per difficulty)
    task_ids = loader.list_task_ids()
    assert len(task_ids) >= 3
    
    # Should have easy, medium, hard
    easy_tasks = [tid for tid in task_ids if tid.startswith("easy_")]
    medium_tasks = [tid for tid in task_ids if tid.startswith("medium_")]
    hard_tasks = [tid for tid in task_ids if tid.startswith("hard_")]
    
    assert len(easy_tasks) >= 1
    assert len(medium_tasks) >= 1
    assert len(hard_tasks) >= 1


def test_scenarios_have_required_fields():
    """Test that all scenarios have required fields."""
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    required_fields = [
        "task_id",
        "task_level",
        "max_turns",
        "patient_profile",
        "initial_clinic_state",
        "clinic_config",
        "ground_truth",
        "success_criteria",
    ]
    
    for task_id in loader.list_task_ids():
        scenario = loader.get_scenario(task_id)
        
        for field in required_fields:
            assert hasattr(scenario, field), f"Scenario {task_id} missing field: {field}"


def test_ground_truth_complete():
    """Test that ground_truth has necessary information."""
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    for task_id in loader.list_task_ids():
        scenario = loader.get_scenario(task_id)
        ground_truth = scenario.ground_truth
        
        # Should have correct department
        assert hasattr(ground_truth, "correct_department")
        assert ground_truth.correct_department is not None
        
        # Should have urgency level
        assert hasattr(ground_truth, "urgency_level")
        assert ground_truth.urgency_level is not None


def test_clinic_state_has_slots():
    """Test that initial clinic state has available slots."""
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    for task_id in loader.list_task_ids():
        scenario = loader.get_scenario(task_id)
        clinic_state = scenario.initial_clinic_state
        
        # Should have available_slots
        assert hasattr(clinic_state, "available_slots")
        assert isinstance(clinic_state.available_slots, dict)
        
        # Should have at least one slot
        total_slots = sum(len(slots) for slots in clinic_state.available_slots.values())
        assert total_slots > 0, f"Scenario {task_id} has no available slots"


def test_patient_profile_complete():
    """Test that patient profiles have necessary fields."""
    loader = ScenarioLoader("scenarios")
    loader.load_all()
    
    for task_id in loader.list_task_ids():
        scenario = loader.get_scenario(task_id)
        profile = scenario.patient_profile
        
        # Should have urgency level
        assert hasattr(profile, "urgency_level")
        assert 0.0 <= profile.urgency_level <= 1.0
        
        # Should have language mix
        assert hasattr(profile, "language_mix")
        assert 0.0 <= profile.language_mix <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
