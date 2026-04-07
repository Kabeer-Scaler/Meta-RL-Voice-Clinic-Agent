"""UrgentQueueManager sub-agent: tracks urgent queue pressure."""

from typing import Any


class UrgentQueueManager:
    """
    Tracks the urgent patient queue and computes queue pressure.
    High pressure means the agent should be more conservative about
    routing non-urgent patients to urgent slots.
    """

    MAX_QUEUE = 10  # queue length that maps to pressure = 1.0

    def __init__(self):
        self._queue_length: int = 0

    def initialize(self, initial_queue_length: int) -> None:
        self._queue_length = max(0, initial_queue_length)

    @property
    def queue_length(self) -> int:
        return self._queue_length

    def get_queue_pressure(self) -> float:
        """Returns queue pressure in [0.0, 1.0]."""
        return min(self._queue_length / self.MAX_QUEUE, 1.0)

    def process_timed_event(self, event: Any, availability_tracker: Any) -> None:
        """Update queue state based on timed events."""
        event_type = event.event_type
        payload = event.payload if hasattr(event, "payload") else {}

        if event_type == "urgent_consume_slot":
            # An urgent patient arrived and took a slot
            self._queue_length = max(0, self._queue_length - 1)
            slot_id = payload.get("slot_id", "")
            if slot_id:
                availability_tracker.reserve_slot(slot_id)

        elif event_type == "urgent_arrival":
            # New urgent patient joins queue
            count = payload.get("count", 1)
            self._queue_length += count
