"""ReceptionistAgent sub-agent: validates bookings and handles admin checks."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""


class ReceptionistAgent:
    """
    Validates booking requests, checks for duplicates, and handles
    insurance/admin friction for medium and hard tasks.
    """

    def __init__(self):
        self._confirmed_bookings: List[str] = []   # slot_ids already booked
        self._insurance_check_done: bool = False
        self._insurance_check_turn: Optional[int] = None

    def initialize(self, scenario: Any) -> None:
        self._confirmed_bookings = []
        self._insurance_check_done = False
        self._insurance_check_turn = None

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def check_duplicate_booking(self, slot_id: str) -> bool:
        """Returns True if this slot has already been booked this episode."""
        return slot_id in self._confirmed_bookings

    def record_booking(self, slot_id: str) -> None:
        if slot_id not in self._confirmed_bookings:
            self._confirmed_bookings.append(slot_id)

    # ------------------------------------------------------------------
    # Booking validation
    # ------------------------------------------------------------------

    def validate_booking_request(
        self,
        slot_id: str,
        availability_tracker: Any,
    ) -> ValidationResult:
        if self.check_duplicate_booking(slot_id):
            return ValidationResult(valid=False, reason="Duplicate booking detected.")

        if not availability_tracker.is_slot_available(slot_id):
            return ValidationResult(valid=False, reason=f"Slot {slot_id} is no longer available.")

        return ValidationResult(valid=True, reason="Booking request is valid.")

    # ------------------------------------------------------------------
    # Insurance / admin friction (medium / hard tasks)
    # ------------------------------------------------------------------

    def process_insurance_check(self, turn: int) -> Optional[str]:
        """
        Returns an admin message if an insurance check is triggered,
        otherwise None. Check resolves after 2 turns.
        """
        if self._insurance_check_done:
            return None

        if self._insurance_check_turn is None:
            self._insurance_check_turn = turn
            return "Insurance verification in progress. Please hold for a moment."

        if turn >= self._insurance_check_turn + 2:
            self._insurance_check_done = True
            return "Insurance verified successfully. You may proceed."

        return "Insurance check still in progress..."
