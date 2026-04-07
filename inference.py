"""
Inference Script for VoiceClinicAgent - OpenEnv Format Compliant

MANDATORY REQUIREMENTS:
- Uses OpenAI Client for all LLM calls
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
- Emits structured stdout logs: [START], [STEP], [END]
- Returns scores in [0.0, 1.0]
"""

import os
import sys
import json
from typing import List, Optional
from openai import OpenAI

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import the OpenEnv client
from src.voiceclinicagent.client import VoiceClinicEnv
from src.voiceclinicagent.api_models import VoiceClinicAction


# Environment variables with defaults
# Note: API_BASE_URL is for LLM API, ENV_BASE_URL is for the environment server
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")
BENCHMARK = "voice-clinic-agent"


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] log line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] log line."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] log line."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


class LLMAgent:
    """
    LLM-based agent using OpenAI Client for decision making.
    
    Uses the OpenAI API to make intelligent decisions based on:
    - Patient messages
    - Conversation history
    - Reflection token (urgency signals)
    - Available actions
    """
    
    def __init__(self, api_key: str, model: str, base_url: str):
        """Initialize LLM agent with OpenAI client."""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.conversation_history = []
        self.actions_taken = []
    
    def reset(self):
        """Reset agent state for new episode."""
        self.conversation_history = []
        self.actions_taken = []
    
    def _build_prompt(self, observation) -> str:
        """Build prompt for LLM based on observation."""
        patient_flags = observation.patient_flags or {}
        clinic_state = observation.clinic_state or {}
        reflection_token = observation.reflection_token or [0.0, 0.0, 0.0, 0.0]
        
        prompt = f"""You are a clinic appointment booking assistant. Your goal is to efficiently book appointments while maintaining privacy and handling urgent cases appropriately.

Current Situation:
- Turn: {observation.turn_idx + 1}/{observation.max_turns}
- Patient Message: "{observation.patient_message}"
- Conversation Summary: {observation.conversation_summary}

Reflection Signals (0.0-1.0):
- Need Escalation: {reflection_token[0]:.2f} (>0.7 = urgent)
- Info Missing: {reflection_token[1]:.2f}
- Privacy Risk: {reflection_token[2]:.2f}
- Slot Pressure: {reflection_token[3]:.2f}

Patient Flags:
- Urgency Hint: {patient_flags.get('urgency_score_hint', 0.0):.2f}
- Parent Booking: {patient_flags.get('parent_booking', 0)}
- High Risk: {patient_flags.get('high_risk_flag', 0)}

Available Actions:
1. ask_question - Ask patient a question (types: symptoms, duration, preferred_time, severity, medical_history)
2. query_availability - Check available slots for a department
3. offer_slot - Offer a specific appointment slot
4. confirm_booking - Confirm and finalize the booking
5. escalate_urgent - Escalate to urgent queue (for emergencies)
6. escalate_human - Escalate to human operator
7. end_call - End the conversation

Previous Actions: {', '.join(self.actions_taken[-5:]) if self.actions_taken else 'None'}

Instructions:
- If need_escalation > 0.7 or urgency_hint > 0.7, consider escalating
- Ask required questions BEFORE querying availability
- Query availability BEFORE offering slots
- Offer slot BEFORE confirming booking
- Avoid asking unnecessary questions
- Respect privacy (don't ask for PII like name, phone, address)

Respond with ONLY a JSON object in this format:
{{"action_type": "ask_question", "payload": {{"question_type": "symptoms", "question_text": "What symptoms are you experiencing?"}}}}

Choose the best action now:"""
        
        return prompt
    
    def select_action(self, observation) -> VoiceClinicAction:
        """
        Select action using LLM.
        
        Args:
            observation: VoiceClinicObservation object
            
        Returns:
            VoiceClinicAction
        """
        try:
            # Build prompt
            prompt = self._build_prompt(observation)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful clinic booking assistant. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response
            import json
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            action_dict = json.loads(response_text)
            
            # Create action
            action = VoiceClinicAction(
                action_type=action_dict["action_type"],
                payload=action_dict.get("payload", {})
            )
            
            # Track action
            self.actions_taken.append(action_dict["action_type"])
            
            return action
            
        except Exception as e:
            # Fallback to safe action on error
            print(f"[WARNING] LLM error: {e}, using fallback action", file=sys.stderr)
            return VoiceClinicAction(
                action_type="ask_question",
                payload={
                    "question_type": "symptoms",
                    "question_text": "What symptoms are you experiencing?"
                }
            )


class RuleBasedAgent:
    """
    Simple rule-based baseline agent for VoiceClinicAgent.
    
    Strategy:
    1. Ask about symptoms
    2. Check urgency from reflection token
    3. If urgent (>0.7), escalate
    4. Otherwise, ask required questions
    5. Query availability for correct department
    6. Offer first available slot
    7. Confirm booking
    """
    
    def __init__(self):
        self.asked_symptoms = False
        self.asked_duration = False
        self.asked_preferred_time = False
        self.queried_availability = False
        self.offered_slot = False
        self.current_slot_id = None
    
    def reset(self):
        """Reset agent state for new episode."""
        self.asked_symptoms = False
        self.asked_duration = False
        self.asked_preferred_time = False
        self.queried_availability = False
        self.offered_slot = False
        self.current_slot_id = None
    
    def select_action(self, observation) -> VoiceClinicAction:
        """
        Select action based on observation.
        
        Args:
            observation: VoiceClinicObservation object
            
        Returns:
            VoiceClinicAction
        """
        patient_flags = observation.patient_flags or {}
        clinic_state = observation.clinic_state or {}
        clinical_history = observation.clinical_history
        reflection_token = observation.reflection_token or [0.0, 0.0, 0.0, 0.0]
        turn_idx = observation.turn_idx
        
        # Extract clinical history info
        if hasattr(clinical_history, 'high_risk_flag'):
            high_risk = clinical_history.high_risk_flag
            slot_pref = clinical_history.slot_preference_time
            preferred_dept = clinical_history.preferred_department or "general"
        else:
            high_risk = False
            slot_pref = None
            preferred_dept = "general"
        
        urgency_hint = patient_flags.get("urgency_score_hint", 0.0)
        
        # Rule 1: Use reflection token for urgency detection
        # reflection_token[0] = need_escalation signal
        need_escalation = reflection_token[0]
        
        if need_escalation > 0.7 and turn_idx >= 2:
            return VoiceClinicAction(
                action_type="escalate_urgent",
                payload={
                    "urgency_reason": f"High urgency detected (score: {need_escalation:.2f})",
                    "symptoms": observation.patient_message[:100]
                }
            )
        
        # Rule 2: Fallback to urgency_hint
        if urgency_hint >= 0.7 or high_risk:
            if turn_idx >= 2:
                return VoiceClinicAction(
                    action_type="escalate_urgent",
                    payload={
                        "urgency_reason": "High urgency score or high-risk patient",
                        "symptoms": observation.patient_message[:100]
                    }
                )
        
        # Rule 3: Ask about symptoms
        if not self.asked_symptoms:
            self.asked_symptoms = True
            return VoiceClinicAction(
                action_type="ask_question",
                payload={
                    "question_type": "symptoms",
                    "question_text": "What symptoms are you experiencing?"
                }
            )
        
        # Rule 4: Ask about duration
        if not self.asked_duration:
            self.asked_duration = True
            return VoiceClinicAction(
                action_type="ask_question",
                payload={
                    "question_type": "duration",
                    "question_text": "How long have you had these symptoms?"
                }
            )
        
        # Rule 5: Ask about preferred time (if not in history)
        if not self.asked_preferred_time and not slot_pref:
            self.asked_preferred_time = True
            return VoiceClinicAction(
                action_type="ask_question",
                payload={
                    "question_type": "preferred_time",
                    "question_text": "What time of day works best for you?"
                }
            )
        
        # Rule 6: Query availability
        if not self.queried_availability:
            self.queried_availability = True
            return VoiceClinicAction(
                action_type="query_availability",
                payload={"department": preferred_dept}
            )
        
        # Rule 7: Offer first available slot
        if not self.offered_slot:
            available_slots = clinic_state.get("available_slots", {})
            
            # Try preferred department first
            if preferred_dept in available_slots and available_slots[preferred_dept]:
                self.current_slot_id = available_slots[preferred_dept][0]
            else:
                # Fallback to any available slot
                for dept, slots in available_slots.items():
                    if slots:
                        self.current_slot_id = slots[0]
                        break
            
            if self.current_slot_id:
                self.offered_slot = True
                return VoiceClinicAction(
                    action_type="offer_slot",
                    payload={"slot_id": self.current_slot_id}
                )
        
        # Rule 8: Confirm booking
        if self.offered_slot and self.current_slot_id:
            return VoiceClinicAction(
                action_type="confirm_booking",
                payload={
                    "slot_id": self.current_slot_id,
                    "patient_confirmation": True
                }
            )
        
        # Fallback: end call
        return VoiceClinicAction(action_type="end_call", payload={})


def run_episode(task_id: str, env_base: str, api_base: str, model_name: str, agent_type: str = "rule-based") -> float:
    """
    Run a single episode with structured logging.
    
    Args:
        task_id: Task identifier (easy_001, medium_001, hard_001)
        env_base: Base URL for the environment API (local server)
        api_base: Base URL for the LLM API (OpenAI, HF, etc.)
        model_name: Model name for logging
        agent_type: "rule-based" or "llm"
        
    Returns:
        Final score in [0.0, 1.0]
    """
    # Create agent based on type
    if agent_type == "llm":
        if not HF_TOKEN:
            print("[ERROR] HF_TOKEN not set, cannot use LLM agent", file=sys.stderr)
            return 0.0
        agent = LLMAgent(api_key=HF_TOKEN, model=model_name, base_url=api_base)
    else:
        agent = RuleBasedAgent()
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    # Emit [START] log
    log_start(task=task_id, env=BENCHMARK, model=model_name)
    
    try:
        # Create OpenEnv client (connects to local environment server)
        with VoiceClinicEnv(base_url=env_base).sync() as env:
            # Reset environment
            result = env.reset(task_id=task_id, seed=42)
            observation = result.observation
            
            # Episode loop
            done = observation.done
            max_turns = observation.max_turns
            
            while not done and steps_taken < max_turns:
                steps_taken += 1
                
                # Select action
                action = agent.select_action(observation)
                
                # Format action string for logging
                action_str = f"{action.action_type}"
                if action.payload:
                    # Truncate payload for logging
                    payload_str = str(action.payload)[:50]
                    action_str = f"{action.action_type}({payload_str})"
                
                # Execute step
                try:
                    result = env.step(action)
                    observation = result.observation
                    done = observation.done
                    reward = observation.reward if observation.reward is not None else 0.0
                    error = None
                    
                    rewards.append(reward)
                    
                    # Emit [STEP] log
                    log_step(
                        step=steps_taken,
                        action=action_str,
                        reward=reward,
                        done=done,
                        error=error
                    )
                    
                except Exception as e:
                    error = str(e)
                    reward = 0.0
                    rewards.append(reward)
                    
                    # Emit [STEP] log with error
                    log_step(
                        step=steps_taken,
                        action=action_str,
                        reward=reward,
                        done=True,
                        error=error
                    )
                    break
            
            # Get final state
            try:
                state = env.state()
                score = state.final_score if state.final_score is not None else 0.0
                # Clamp score to [0.0, 1.0]
                score = min(max(score, 0.0), 1.0)
                success = score >= 0.1  # Success threshold
            except Exception:
                score = 0.0
                success = False
    
    except Exception as e:
        # Episode failed to start
        error = str(e)
        log_step(step=0, action="reset", reward=0.0, done=True, error=error)
        score = 0.0
        success = False
    
    finally:
        # Always emit [END] log
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score


def main():
    """Main entry point for inference baseline."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="VoiceClinicAgent Inference")
    parser.add_argument(
        "--agent",
        type=str,
        default="rule-based",
        choices=["rule-based", "llm"],
        help="Agent type to use (default: rule-based)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["easy_001", "medium_001", "hard_001"],
        help="Task IDs to evaluate"
    )
    args = parser.parse_args()
    
    # Determine agent type and model name
    agent_type = args.agent
    if agent_type == "llm":
        if not HF_TOKEN:
            print("[ERROR] HF_TOKEN environment variable not set. Cannot use LLM agent.", file=sys.stderr)
            print("[ERROR] Please set HF_TOKEN to your OpenAI API key or Hugging Face token.", file=sys.stderr)
            sys.exit(1)
        model_display = MODEL_NAME
    else:
        model_display = "rule-based-agent"
    
    # Task IDs to evaluate
    task_ids = args.tasks
    
    # Run all tasks
    scores = []
    
    print(f"\n[INFO] Running {agent_type} agent on {len(task_ids)} tasks...", file=sys.stderr)
    if agent_type == "llm":
        print(f"[INFO] Using model: {MODEL_NAME}", file=sys.stderr)
        print(f"[INFO] API Base: {API_BASE_URL}", file=sys.stderr)
    print("", file=sys.stderr)
    
    for task_id in task_ids:
        try:
            score = run_episode(
                task_id=task_id,
                env_base=ENV_BASE_URL,
                api_base=API_BASE_URL,
                model_name=model_display,
                agent_type=agent_type
            )
            scores.append(score)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr, flush=True)
            scores.append(0.0)
    
    # Print summary to stderr (not stdout, to avoid interfering with structured logs)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\n[SUMMARY] Average score: {avg_score:.3f}", file=sys.stderr, flush=True)
    print(f"[SUMMARY] Scores: {scores}", file=sys.stderr, flush=True)
    
    return scores


if __name__ == "__main__":
    main()
