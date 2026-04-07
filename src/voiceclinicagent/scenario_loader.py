"""Scenario loader for VoiceClinicAgent."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from .models import Scenario


class ScenarioLoader:
    """
    Loads and caches scenario definitions from JSON files.
    
    Scenarios are organized in directories:
    - scenarios/easy/
    - scenarios/medium/
    - scenarios/hard/
    """
    
    def __init__(self, scenario_dir: str = "scenarios"):
        """
        Initialize scenario loader.
        
        Args:
            scenario_dir: Root directory containing scenario files
        """
        self.scenario_dir = Path(scenario_dir)
        self._cache: Dict[str, Scenario] = {}
        self._loaded = False
    
    def load_all(self) -> None:
        """Load all scenarios from disk into cache."""
        if self._loaded:
            return
        
        for level in ["easy", "medium", "hard"]:
            level_dir = self.scenario_dir / level
            if not level_dir.exists():
                print(f"Warning: Scenario directory not found: {level_dir}")
                continue
            
            for json_file in level_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    scenario = Scenario(**data)
                    self._cache[scenario.task_id] = scenario
                    print(f"Loaded scenario: {scenario.task_id}")
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        self._loaded = True
        print(f"Total scenarios loaded: {len(self._cache)}")
    
    def get_scenario(self, task_id: str) -> Scenario:
        """
        Get scenario by task ID.
        
        Args:
            task_id: Task identifier (e.g., "easy_001")
            
        Returns:
            Scenario object
            
        Raises:
            ValueError: If task_id not found
        """
        if not self._loaded:
            self.load_all()
        
        if task_id not in self._cache:
            raise ValueError(f"Scenario not found: {task_id}. Available: {list(self._cache.keys())}")
        
        return self._cache[task_id]
    
    def list_task_ids(self) -> List[str]:
        """
        List all available task IDs.
        
        Returns:
            List of task IDs
        """
        if not self._loaded:
            self.load_all()
        
        return sorted(self._cache.keys())
    
    def get_scenarios_by_level(self, level: str) -> List[Scenario]:
        """
        Get all scenarios for a difficulty level.
        
        Args:
            level: "easy", "medium", or "hard"
            
        Returns:
            List of scenarios
        """
        if not self._loaded:
            self.load_all()
        
        return [s for s in self._cache.values() if s.task_level == level]
