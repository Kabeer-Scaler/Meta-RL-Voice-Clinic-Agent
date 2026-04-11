"""
Inference Script for VoiceClinicAgent - OpenEnv Format Compliant

MANDATORY REQUIREMENTS:
- Uses OpenAI Client for all LLM calls
- Reads API_BASE_URL, MODEL_NAME, API_KEY from environment
- Emits structured stdout logs: [START], [STEP], [END]
- Returns scores in [0.0, 1.0]
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI
import requests

# CRITICAL: Keep module-level reads for validator compatibility, but never
# fall back to a third-party provider URL. The validator must supply the proxy.
DEFAULT_API_BASE_URL = ""
DEFAULT_MODEL_NAME = "gpt-4.1-mini"
DEFAULT_ENV_BASE_URL = "http://localhost:7860"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", DEFAULT_ENV_BASE_URL)

# Import the OpenEnv client - handle case where it's not available.
# The WebSocket-backed client preserves episode state correctly when supported.
try:
    from src.voiceclinicagent.client import VoiceClinicEnv
    CLIENT_AVAILABLE = VoiceClinicEnv is not None
except (ImportError, TypeError):
    VoiceClinicEnv = None
    CLIENT_AVAILABLE = False

from src.voiceclinicagent.api_models import VoiceClinicAction
from src.voiceclinicagent.env import VoiceClinicEnvironment


# Constants
BENCHMARK = "voice-clinic-agent"


@dataclass(frozen=True)
class RuntimeConfig:
    api_base_url: str
    api_key: str
    model_name: str
    env_base_url: str


def _clean_env_value(value: Optional[str]) -> str:
    """Normalize optional env vars so whitespace-only values are treated as missing."""
    return value.strip() if isinstance(value, str) else ""


def _mask_secret(secret: str) -> str:
    """Mask a secret before logging it to stderr."""
    if not secret:
        return "<missing>"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def load_runtime_config() -> RuntimeConfig:
    """
    Re-read env vars immediately before execution.

    This avoids using stale values from module import time and strips accidental
    whitespace that would otherwise break proxy auth or URL parsing.
    """
    api_base_url = _clean_env_value(os.environ.get("API_BASE_URL", API_BASE_URL))
    api_key = _clean_env_value(os.environ.get("API_KEY", API_KEY))
    model_name = _clean_env_value(os.environ.get("MODEL_NAME", MODEL_NAME)) or DEFAULT_MODEL_NAME
    env_base_url = _clean_env_value(os.environ.get("ENV_BASE_URL", ENV_BASE_URL)) or DEFAULT_ENV_BASE_URL

    if not api_key:
        raise ValueError("API_KEY environment variable is required")
    if not api_base_url:
        raise ValueError("API_BASE_URL environment variable is required")
    if "api.openai.com" in api_base_url or "router.huggingface.co" in api_base_url:
        raise ValueError(
            "API_BASE_URL is pointing at a fallback provider URL instead of the "
            "validator's injected LiteLLM proxy URL."
        )

    return RuntimeConfig(
        api_base_url=api_base_url,
        api_key=api_key,
        model_name=model_name,
        env_base_url=env_base_url,
    )


def _build_openai_client(api_key: str, base_url: str) -> OpenAI:
    """Create an OpenAI client with conservative retry and timeout settings."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
        timeout=20.0,
    )


def _extract_completion_text(response) -> str:
    """Best-effort extraction of assistant text across SDK response shapes."""
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if message is None:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                else:
                    text = getattr(item, "text", "") or getattr(item, "content", "") or ""
                if text:
                    chunks.append(str(text).strip())
            return "\n".join(chunk for chunk in chunks if chunk)
    except Exception:
        return ""

    return ""


def verify_proxy_call(config: RuntimeConfig, attempts: int = 3) -> bool:
    """
    Try to make at least one call through the injected validator proxy.

    Returns True on success. On failure, logs details and returns False so the
    script can still complete cleanly instead of crashing Phase 2.
    """
    client = _build_openai_client(config.api_key, config.api_base_url)
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            print(
                f"[INFO] Validator proxy call succeeded on attempt {attempt}",
                file=sys.stderr,
                flush=True,
            )
            response_text = _extract_completion_text(response)
            if response_text:
                print(
                    f"[DEBUG] Proxy response: {response_text[:120]}",
                    file=sys.stderr,
                    flush=True,
                )
            return True
        except Exception as exc:
            last_error = exc
            print(
                f"[WARNING] Validator proxy call attempt {attempt}/{attempts} failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    if last_error is not None:
        print(
            "[ERROR] All validator proxy call attempts failed. "
            f"API_BASE_URL={config.api_base_url!r}, "
            f"API_KEY={_mask_secret(config.api_key)}, "
            f"MODEL_NAME={config.model_name!r}. "
            f"Last error: {type(last_error).__name__}: {last_error}",
            file=sys.stderr,
            flush=True,
        )
    return False


def wait_for_environment(env_base_url: str, attempts: int = 12, timeout_s: int = 5) -> bool:
    """Wait briefly for the environment server to become reachable."""
    for attempt in range(1, attempts + 1):
        try:
            health_response = requests.get(f"{env_base_url}/health", timeout=timeout_s)
            if health_response.status_code == 200:
                print(
                    f"[INFO] Environment server reachable on attempt {attempt}",
                    file=sys.stderr,
                    flush=True,
                )
                return True
            print(
                f"[WARNING] Environment health check attempt {attempt}/{attempts} "
                f"returned {health_response.status_code}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(
                f"[WARNING] Environment health check attempt {attempt}/{attempts} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
    return False


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


def _format_action_for_log(action: VoiceClinicAction) -> str:
    """Format an action compactly for the structured [STEP] logs."""
    action_str = f"{action.action_type}"
    if action.payload:
        payload_str = str(action.payload)[:50]
        action_str = f"{action.action_type}({payload_str})"
    return action_str


def _run_episode_direct(task_id: str, model_name: str, agent) -> tuple[float, bool, int, List[float]]:
    """Run an episode directly against the in-process environment as a fallback."""
    env = VoiceClinicEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        if hasattr(agent, "reset"):
            agent.reset()
        observation = env.reset(seed=42, task_id=task_id)
        done = observation.done
        max_turns = observation.max_turns

        while not done and steps_taken < max_turns:
            steps_taken += 1
            try:
                action = agent.select_action(observation)
                action_str = _format_action_for_log(action)
            except Exception as exc:
                rewards.append(0.0)
                log_step(
                    step=steps_taken,
                    action="select_action",
                    reward=0.0,
                    done=True,
                    error=str(exc),
                )
                break

            try:
                observation = env.step(action)
                done = observation.done
                reward = observation.reward if observation.reward is not None else 0.0
                rewards.append(reward)
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=None,
                )
            except Exception as exc:
                rewards.append(0.0)
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=0.0,
                    done=True,
                    error=str(exc),
                )
                done = True

        state = env.state
        score = state.final_score if state.final_score is not None else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1
    finally:
        try:
            env.close()
        except Exception:
            pass

    return score, success, steps_taken, rewards


def _run_episode_via_openenv_client(task_id: str, env_base: str, agent) -> tuple[float, bool, int, List[float]]:
    """Run an episode through the OpenEnv client against the local server."""
    if not CLIENT_AVAILABLE or VoiceClinicEnv is None:
        raise RuntimeError("OpenEnv client is not available")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    with VoiceClinicEnv(base_url=env_base).sync() as env:
        if hasattr(agent, "reset"):
            agent.reset()

        result = env.reset(task_id=task_id, seed=42)
        observation = result.observation
        done = observation.done
        max_turns = observation.max_turns

        while not done and steps_taken < max_turns:
            steps_taken += 1

            try:
                action = agent.select_action(observation)
                action_str = _format_action_for_log(action)
            except Exception as exc:
                rewards.append(0.0)
                log_step(
                    step=steps_taken,
                    action="select_action",
                    reward=0.0,
                    done=True,
                    error=str(exc),
                )
                break

            try:
                result = env.step(action)
                observation = result.observation
                done = observation.done
                reward = observation.reward if observation.reward is not None else 0.0
                rewards.append(reward)
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=None,
                )
            except Exception as exc:
                rewards.append(0.0)
                log_step(
                    step=steps_taken,
                    action=action_str,
                    reward=0.0,
                    done=True,
                    error=str(exc),
                )
                break

        state_accessor = getattr(env, "state", None)
        state = state_accessor() if callable(state_accessor) else state_accessor
        if state is not None:
            score = state.final_score if state.final_score is not None else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= 0.1

    return score, success, steps_taken, rewards


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
        self.client = _build_openai_client(api_key=api_key, base_url=base_url)
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
        # Build prompt
        prompt = self._build_prompt(observation)
        
        # Call LLM
        print(f"[DEBUG] Making LLM API call to {self.client.base_url}", file=sys.stderr, flush=True)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful clinic booking assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        print(f"[DEBUG] LLM API call successful", file=sys.stderr, flush=True)
        
        # Parse response
        import json
        response_text = _extract_completion_text(response)

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            action_dict = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback to safe action if JSON parsing fails
            print(f"[WARNING] Failed to parse LLM response as JSON: {response_text[:100]}", file=sys.stderr, flush=True)
            action_dict = {
                "action_type": "ask_question",
                "payload": {
                    "question_type": "symptoms",
                    "question_text": "What symptoms are you experiencing?"
                }
            }
        
        # Create action
        action = VoiceClinicAction(
            action_type=action_dict["action_type"],
            payload=action_dict.get("payload", {})
        )
        
        # Track action
        self.actions_taken.append(action_dict["action_type"])
        
        return action


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


def run_episode(task_id: str, env_base: str, api_base: str, api_key: str, model_name: str, agent_type: str = "rule-based") -> float:
    """
    Run a single episode with structured logging.
    
    Args:
        task_id: Task identifier (easy_001, medium_001, hard_001)
        env_base: Base URL for the environment API (local server)
        api_base: Base URL for the LLM API (OpenAI, HF, etc.)
        api_key: API key for LLM
        model_name: Model name for logging
        agent_type: "rule-based" or "llm"
        
    Returns:
        Final score in [0.0, 1.0]
    """
    # Create agent based on type
    if agent_type == "llm":
        if not api_key:
            raise ValueError("API_KEY is required for LLM agent")
        print(f"[DEBUG] Creating LLMAgent with base_url={api_base}, model={model_name}", file=sys.stderr, flush=True)
        agent = LLMAgent(api_key=api_key, model=model_name, base_url=api_base)
        print(f"[DEBUG] LLMAgent created successfully", file=sys.stderr, flush=True)
    else:
        agent = RuleBasedAgent()
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    # Emit [START] log
    log_start(task=task_id, env=BENCHMARK, model=model_name)
    
    try:
        if CLIENT_AVAILABLE:
            try:
                score, success, steps_taken, rewards = _run_episode_via_openenv_client(
                    task_id=task_id,
                    env_base=env_base,
                    agent=agent,
                )
            except Exception as exc:
                print(
                    f"[WARNING] OpenEnv client path failed for task {task_id}: {type(exc).__name__}: {exc}. "
                    "Falling back to the in-process environment.",
                    file=sys.stderr,
                    flush=True,
                )
                score, success, steps_taken, rewards = _run_episode_direct(
                    task_id=task_id,
                    model_name=model_name,
                    agent=agent,
                )
        else:
            print(
                f"[WARNING] OpenEnv client unavailable for task {task_id}; using in-process environment fallback.",
                file=sys.stderr,
                flush=True,
            )
            score, success, steps_taken, rewards = _run_episode_direct(
                task_id=task_id,
                model_name=model_name,
                agent=agent,
            )
    
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
    
    # Ensure stdout is unbuffered for immediate output
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    # Test that stdout logging works immediately (send to stderr to avoid interfering with structured logs)
    print("[TEST] Structured logging initialized", file=sys.stderr, flush=True)
    
    config = load_runtime_config()

    # Debug: Log normalized runtime config (to stderr, not stdout)
    print(f"[DEBUG] API_KEY: {_mask_secret(config.api_key)}", file=sys.stderr, flush=True)
    print(f"[DEBUG] API_BASE_URL: {config.api_base_url}", file=sys.stderr, flush=True)
    print(f"[DEBUG] MODEL_NAME: {config.model_name}", file=sys.stderr, flush=True)
    print(f"[DEBUG] ENV_BASE_URL: {config.env_base_url}", file=sys.stderr, flush=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="VoiceClinicAgent Inference")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["easy_001", "medium_001", "hard_001"],
        help="Task IDs to evaluate"
    )
    args = parser.parse_args()
    
    # Always use the LLM path in the submission entrypoint so proxy traffic is guaranteed.
    agent_type = "llm"
    print(f"[DEBUG] Selected agent type: {agent_type}", file=sys.stderr, flush=True)
    model_display = config.model_name
    
    # Task IDs to evaluate
    task_ids = args.tasks
    
    # Make a validator proxy call before running episodes.
    print("[INFO] Making validator proxy call...", file=sys.stderr, flush=True)
    proxy_ok = verify_proxy_call(config)
    if not proxy_ok:
        print(
            "[WARNING] Continuing even though the validator proxy call did not succeed. "
            "Episode-level LLM calls may still recover if the failure was transient.",
            file=sys.stderr,
            flush=True,
        )

    # Wait briefly for the local environment server to come up.
    env_ok = wait_for_environment(config.env_base_url)
    if not env_ok:
        print(
            f"[WARNING] Environment server was not reachable at {config.env_base_url} before episode start",
            file=sys.stderr,
            flush=True,
        )
    
    # Run all tasks
    scores = []
    
    print(f"\n[INFO] Running {agent_type} agent on {len(task_ids)} tasks...", file=sys.stderr)
    if agent_type == "llm":
        print(f"[INFO] Using model: {config.model_name}", file=sys.stderr)
        print(f"[INFO] API Base: {config.api_base_url}", file=sys.stderr)
    print("", file=sys.stderr)
    
    for task_id in task_ids:
        try:
            score = run_episode(
                task_id=task_id,
                env_base=config.env_base_url,
                api_base=config.api_base_url,
                api_key=config.api_key,
                model_name=model_display,
                agent_type=agent_type
            )
            scores.append(score)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr, flush=True)
            # Still emit structured logs for failed task
            print(f"[START] task={task_id} env={BENCHMARK} model={model_display}", flush=True)
            print(f"[STEP] step=0 action=reset reward=0.00 done=true error={str(e)}", flush=True)
            print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
            scores.append(0.0)
    
    # Print summary to stderr (not stdout, to avoid interfering with structured logs)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\n[SUMMARY] Average score: {avg_score:.3f}", file=sys.stderr, flush=True)
    print(f"[SUMMARY] Scores: {scores}", file=sys.stderr, flush=True)
    
    return scores


if __name__ == "__main__":
    main()
