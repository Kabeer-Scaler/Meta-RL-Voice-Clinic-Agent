"""Transcript builder for conversation history."""

from typing import List, Dict, Any


class TranscriptBuilder:
    """
    Maintains conversation history and generates summaries.
    
    Tracks all agent actions and patient responses for the episode.
    """
    
    def __init__(self):
        self.turns: List[Dict[str, Any]] = []
    
    def add_turn(
        self,
        action: Dict[str, Any],
        patient_response: str,
        reward: float,
    ) -> None:
        """
        Add a conversation turn.
        
        Args:
            action: Agent action dict
            patient_response: Patient's response text
            reward: Reward received for this action
        """
        self.turns.append({
            "turn": len(self.turns) + 1,
            "action_type": action.get("action_type", ""),
            "action_payload": action.get("payload", {}),
            "patient_response": patient_response,
            "reward": reward,
        })
    
    def get_conversation_summary(self, max_turns: int = 5) -> str:
        """
        Generate a brief conversation summary.
        
        Args:
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Summary string
        """
        if not self.turns:
            return "Conversation just started."
        
        recent_turns = self.turns[-max_turns:]
        
        summary_parts = []
        for turn in recent_turns:
            action_type = turn["action_type"]
            summary_parts.append(f"Turn {turn['turn']}: Agent {action_type}")
        
        return " | ".join(summary_parts)
    
    def get_full_transcript(self) -> str:
        """
        Get full conversation transcript.
        
        Returns:
            Full transcript as formatted string
        """
        if not self.turns:
            return "No conversation yet."
        
        lines = []
        for turn in self.turns:
            lines.append(f"Turn {turn['turn']}:")
            lines.append(f"  Agent: {turn['action_type']}")
            lines.append(f"  Patient: {turn['patient_response']}")
            lines.append(f"  Reward: {turn['reward']:.2f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get list of all actions taken.
        
        Returns:
            List of action dicts
        """
        return [
            {
                "action_type": turn["action_type"],
                "payload": turn["action_payload"],
            }
            for turn in self.turns
        ]
    
    def count_action_type(self, action_type: str) -> int:
        """
        Count how many times an action type was used.
        
        Args:
            action_type: Action type to count
            
        Returns:
            Count
        """
        return sum(1 for turn in self.turns if turn["action_type"] == action_type)
    
    def get_turn_count(self) -> int:
        """Get total number of turns."""
        return len(self.turns)
