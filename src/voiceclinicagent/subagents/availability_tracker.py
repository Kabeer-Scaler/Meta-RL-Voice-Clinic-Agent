"""AvailabilityTracker sub-agent: manages slot inventory and timed events."""

import random
from typing import Dict, List, Optional, Any


class AvailabilityTracker:
    """
    Tracks available appointment slots and processes timed events
    (doctor delays, urgent slot consumption).
    """

    def __init__(self):
        self._slots: Dict[str, Dict[str, Any]] = {}   # slot_id -> metadata
        self._available: Dict[str, List[str]] = {}    # dept -> [slot_id, ...]
        self._delay_minutes: int = 0
        self._rng: Optional[random.Random] = None

    def initialize(self, clinic_config: Any, initial_clinic_state: Any, episode_rng: random.Random) -> None:
        self._rng = episode_rng
        self._delay_minutes = initial_clinic_state.doctor_delay_minutes

        # Build slot metadata from clinic_config
        self._slots = {}
        for dept, slot_list in clinic_config.initial_slots.items():
            for slot in slot_list:
                self._slots[slot["slot_id"]] = {
                    "slot_id":    slot["slot_id"],
                    "datetime":   slot.get("datetime", ""),
                    "doctor":     slot.get("doctor", ""),
                    "department": dept,
                }

        # Build available index from initial_clinic_state
        self._available = {}
        for dept, ids in initial_clinic_state.available_slots.items():
            self._available[dept] = list(ids)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_available_slots(self, department: Optional[str] = None) -> Dict[str, List[str]]:
        if department:
            return {department: list(self._available.get(department, []))}
        return {dept: list(ids) for dept, ids in self._available.items()}

    def resolve_slot_id(self, slot_id: str) -> Optional[Dict[str, Any]]:
        return self._slots.get(slot_id)

    def is_slot_available(self, slot_id: str) -> bool:
        return any(slot_id in ids for ids in self._available.values())

    @property
    def delay_minutes(self) -> int:
        return self._delay_minutes

    @property
    def total_available(self) -> int:
        return sum(len(ids) for ids in self._available.values())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def reserve_slot(self, slot_id: str) -> bool:
        """Remove slot from available list. Returns True if it was available."""
        for dept, ids in self._available.items():
            if slot_id in ids:
                ids.remove(slot_id)
                return True
        return False

    def apply_timed_event(self, event: Any) -> str:
        """
        Process a timed event from the scenario.
        Returns a message string describing what happened.
        """
        event_type = event.event_type
        payload = event.payload if hasattr(event, "payload") else {}

        if event_type == "doctor_delay":
            delay = payload.get("delay_minutes", 30)
            self._delay_minutes += delay
            msg = payload.get("message", f"Doctor is running {delay} minutes late.")
            return msg

        elif event_type == "urgent_consume_slot":
            slot_id = payload.get("slot_id", "")
            self.reserve_slot(slot_id)
            msg = payload.get("message", f"Slot {slot_id} taken by urgent case.")
            return msg

        return f"Unknown event: {event_type}"
