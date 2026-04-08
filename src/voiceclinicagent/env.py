"""VoiceClinicEnvironment following OpenEnv pattern."""

from typing import Optional, Dict, Any
from openenv_core.env_server import Environment
from .api_models import VoiceClinicAction, VoiceClinicObservation, VoiceClinicState
from .scenario_loader import ScenarioLoader
from .user_simulator import SyntheticPatientSimulator
from .transcript import TranscriptBuilder
from .observation_builder import ObservationBuilder
from .action_parser import ActionParser
from .rewards import RewardCalculator
from .grader import Grader
from .termination import TerminationChecker
from .subagents import AvailabilityTracker, ReceptionistAgent, UrgentQueueManager
from .memory_vault import PrivacySafeMemoryVault
from .reflection import ReflectionEngine
from .utils.seeding import make_episode_rng
from .utils.ids import generate_episode_id


class VoiceClinicEnvironment(Environment[VoiceClinicAction, VoiceClinicObservation, VoiceClinicState]):
    """
    OpenEnv-compatible environment for clinic appointment booking.
    
    Implements the OpenEnv pattern:
    - reset() -> Observation
    - step(action) -> Observation
    - state property -> State
    """
    
    def __init__(self, scenario_dir: str = "scenarios"):
        # Initialize parent Environment class
        super().__init__()
        
        # Load scenarios
        self.scenario_loader = ScenarioLoader(scenario_dir)
        self.scenario_loader.load_all()
        
        # Components
        self.patient_simulator = SyntheticPatientSimulator()
        self.transcript = TranscriptBuilder()
        self.observation_builder = ObservationBuilder()
        self.action_parser = ActionParser()
        self.reward_calculator = RewardCalculator()
        self.grader = Grader()
        self.termination_checker = TerminationChecker()

        # Sub-agents (Phase 4)
        self.availability_tracker = AvailabilityTracker()
        self.receptionist = ReceptionistAgent()
        self.urgent_queue_manager = UrgentQueueManager()
        
        # Phase 5: Memory vault and reflection
        self.memory_vault = PrivacySafeMemoryVault()
        self.reflection_engine = ReflectionEngine()
        
        # Episode state
        self._state = VoiceClinicState()
        self._scenario = None
        self._episode_rng = None
        self._done = False
        self._truncated = False
        self._current_reward = 0.0
        self._final_score = None
        
        # Clinic state
        self._available_slots: Dict[str, list] = {}
        self._doctor_delay_minutes = 0
        self._urgent_queue_length = 0
    
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> VoiceClinicObservation:
        """
        Start a new episode.
        
        Args:
            seed: Random seed for deterministic episodes
            episode_id: Optional episode identifier (unused, generated internally)
            **kwargs: Additional arguments (task_id can be passed here)
            
        Returns:
            Initial observation
        """
        # Extract task_id from kwargs, default to easy_001
        task_id = kwargs.get("task_id", "easy_001")
        
        # Load scenario
        self._scenario = self.scenario_loader.get_scenario(task_id)
        
        # Create seeded RNG
        self._episode_rng = make_episode_rng(seed)
        
        # Initialize state
        self._state = VoiceClinicState(
            episode_id=generate_episode_id(),
            step_count=0,
            task_id=task_id,
            max_turns=self._scenario.max_turns,
        )
        
        # Initialize patient simulator
        self.patient_simulator.initialize(self._scenario, self._episode_rng)
        
        # Initialize transcript
        self.transcript = TranscriptBuilder()
        
        # Initialize clinic state from scenario
        self._available_slots = dict(self._scenario.initial_clinic_state.available_slots)
        self._doctor_delay_minutes = self._scenario.initial_clinic_state.doctor_delay_minutes
        self._urgent_queue_length = self._scenario.initial_clinic_state.urgent_queue_length

        # Initialize sub-agents
        self.availability_tracker.initialize(
            self._scenario.clinic_config,
            self._scenario.initial_clinic_state,
            self._episode_rng,
        )
        self.receptionist.initialize(self._scenario)
        self.urgent_queue_manager.initialize(self._scenario.initial_clinic_state.urgent_queue_length)
        
        # Clear memory vault for new episode
        patient_id = self._scenario.patient_profile.patient_id if hasattr(self._scenario.patient_profile, 'patient_id') else 'unknown'
        self.memory_vault.clear_patient(patient_id)
        
        # Reset reward calculator conversation state
        self.reward_calculator.reset()
        
        self._done = False
        self._truncated = False
        self._current_reward = None
        self._final_score = None
        
        # Build initial observation
        patient_flags = self.observation_builder.build_patient_flags(
            self._scenario.patient_profile,
            self.patient_simulator.frustration,
            self.patient_simulator.get_revealed_facts(),
        )
        
        clinic_state = self.observation_builder.build_clinic_state(
            self._available_slots,
            self._doctor_delay_minutes,
            self._urgent_queue_length,
        )
        
        # Build clinical history from scenario
        clinical_history = self.observation_builder.build_clinical_history_from_scenario(self._scenario)
        
        # Compute reflection token
        reflection_token = self.reflection_engine.compute_reflection_token(
            scenario=self._scenario,
            action_history=[],
            revealed_facts=self.patient_simulator.get_revealed_facts(),
            available_slots=self._available_slots,
            urgent_queue_length=self._urgent_queue_length,
            turn_idx=0,
            max_turns=self._scenario.max_turns,
        )
        
        # Get memory vault summary
        patient_id = self._scenario.patient_profile.patient_id if hasattr(self._scenario.patient_profile, 'patient_id') else 'unknown'
        memory_vault_summary = self.memory_vault.get_summary(patient_id)
        
        # Create initial patient message
        initial_message = "Hello, I need to book an appointment."
        
        from .user_simulator import PatientResponse
        initial_response = PatientResponse(
            text=initial_message,
            revealed_facts=[],
            frustration_level=0.0,
        )
        
        return self.observation_builder.build(
            task_level=self._scenario.task_level,
            turn_idx=0,
            max_turns=self._scenario.max_turns,
            patient_response=initial_response,
            conversation_summary="Conversation just started.",
            patient_flags=patient_flags,
            clinic_state=clinic_state,
            clinical_history=clinical_history,
            reflection_token=reflection_token,
            memory_vault_summary=memory_vault_summary,
            privacy_risk_mask=self.observation_builder.build_privacy_risk_mask([]),
            done=False,
            reward=None,
        )
    
    def step(self, action: VoiceClinicAction, timeout_s: Optional[float] = None, **kwargs: Any) -> VoiceClinicObservation:
        """
        Execute one environment step.
        
        Args:
            action: Agent action
            timeout_s: Optional timeout (unused in this environment)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Observation after action
        """
        # Convert action to dict for internal processing
        action_dict = {
            "action_type": action.action_type,
            "payload": action.payload,
        }
        
        # Parse and validate action
        parsed = self.action_parser.parse(action_dict)
        if not parsed.is_valid:
            # Invalid action - return negative reward and continue
            self._current_reward = -0.5
            
            patient_flags = self.observation_builder.build_patient_flags(
                self._scenario.patient_profile,
                self.patient_simulator.frustration,
                self.patient_simulator.get_revealed_facts(),
            )
            
            clinic_state = self.observation_builder.build_clinic_state(
                self._available_slots,
                self._doctor_delay_minutes,
                self._urgent_queue_length,
            )
            
            from .user_simulator import PatientResponse
            error_response = PatientResponse(
                text=f"I don't understand. ({parsed.error})",
                revealed_facts=[],
                frustration_level=self.patient_simulator.frustration,
            )
            
            # Build clinical history
            clinical_history = self.observation_builder.build_clinical_history_from_scenario(self._scenario)
            
            # Compute reflection token
            patient_id = self._scenario.patient_profile.patient_id if hasattr(self._scenario.patient_profile, 'patient_id') else 'unknown'
            reflection_token = self.reflection_engine.compute_reflection_token(
                scenario=self._scenario,
                action_history=self.transcript.get_action_history(),
                revealed_facts=self.patient_simulator.get_revealed_facts(),
                available_slots=self._available_slots,
                urgent_queue_length=self._urgent_queue_length,
                turn_idx=self._state.step_count,
                max_turns=self._scenario.max_turns,
            )
            memory_vault_summary = self.memory_vault.get_summary(patient_id)
            
            return self.observation_builder.build(
                task_level=self._scenario.task_level,
                turn_idx=self._state.step_count,
                max_turns=self._scenario.max_turns,
                patient_response=error_response,
                conversation_summary=self.transcript.get_conversation_summary(),
                patient_flags=patient_flags,
                clinic_state=clinic_state,
                clinical_history=clinical_history,
                reflection_token=reflection_token,
                memory_vault_summary=memory_vault_summary,
                privacy_risk_mask=self.observation_builder.build_privacy_risk_mask(
                    self.transcript.get_action_history()
                ),
                done=False,
                reward=self._current_reward,
            )
        
        # Increment step count
        self._state.step_count += 1
        
        # Get patient_id for memory operations
        patient_id = self._scenario.patient_profile.patient_id if hasattr(self._scenario.patient_profile, 'patient_id') else 'unknown'
        
        # Handle memory vault actions (store_memory_safe, recall_memory)
        privacy_violation_hard = False
        memory_response_text = None
        
        if action_dict.get("action_type") == "store_memory_safe":
            key = action_dict.get("payload", {}).get("memory_key", "")
            value = action_dict.get("payload", {}).get("value", "")
            
            success, terminate, error = self.memory_vault.store(key, value, patient_id)
            
            if terminate:
                # Blocked PII key - hard termination
                privacy_violation_hard = True
                memory_response_text = f"PRIVACY VIOLATION: {error}"
            elif not success:
                memory_response_text = f"Memory storage failed: {error}"
            else:
                memory_response_text = f"Stored {key} successfully."
        
        elif action_dict.get("action_type") == "recall_memory":
            key = action_dict.get("payload", {}).get("memory_key", "")
            
            success, terminate, value, error = self.memory_vault.recall(key, patient_id)
            
            if terminate:
                # Blocked PII key - hard termination
                privacy_violation_hard = True
                memory_response_text = f"PRIVACY VIOLATION: {error}"
            elif not success:
                memory_response_text = f"Memory recall failed: {error}"
            else:
                memory_response_text = f"Recalled {key}: {value}"

        # Process timed events for this turn
        for event in self._scenario.timed_events:
            if event.turn == self._state.step_count:
                self.urgent_queue_manager.process_timed_event(event, self.availability_tracker)
                self.availability_tracker.apply_timed_event(event)

        # Sync clinic state from sub-agents
        self._available_slots = {
            dept: list(ids)
            for dept, ids in self.availability_tracker.get_available_slots().items()
        }
        self._doctor_delay_minutes = self.availability_tracker.delay_minutes
        self._urgent_queue_length = self.urgent_queue_manager.queue_length
        
        # Get patient response (or use memory response for memory actions)
        if memory_response_text:
            from .user_simulator import PatientResponse
            patient_response = PatientResponse(
                text=memory_response_text,
                revealed_facts=[],
                frustration_level=self.patient_simulator.frustration,
            )
        else:
            patient_response = self.patient_simulator.respond_to_action(
                action_dict,
                self.transcript.get_action_history(),
            )
        
        # Compute reward using full reward system
        reward = self.reward_calculator.compute_step_reward(
            action=action_dict,
            patient_response=patient_response,
            action_history=self.transcript.get_action_history(),
            scenario=self._scenario,
            available_slots=self._available_slots,
        )

        # For confirm_booking: validate via receptionist and reserve slot
        if action_dict.get("action_type") == "confirm_booking":
            slot_id = action_dict.get("payload", {}).get("slot_id", "")
            validation = self.receptionist.validate_booking_request(slot_id, self.availability_tracker)
            if validation.valid:
                self.availability_tracker.reserve_slot(slot_id)
                self.receptionist.record_booking(slot_id)
                self._state.booking_confirmed = True
            else:
                # Invalid booking attempt — penalise and don't terminate
                reward = -0.3
        self._current_reward = reward
        
        # Check termination using full termination checker
        done, truncated, reason = self.termination_checker.check(
            action=action_dict,
            turn_idx=self._state.step_count,
            max_turns=self._scenario.max_turns,
            frustration=self.patient_simulator.frustration,
            privacy_violation_hard=privacy_violation_hard,
        )
        self._done = done
        self._truncated = truncated
        
        # If episode is done, compute final score
        if done or truncated:
            from .models import EpisodeState
            episode_state = EpisodeState(
                episode_id=self._state.episode_id,
                task_id=self._scenario.task_id,
                turn_idx=self._state.step_count,
                max_turns=self._scenario.max_turns,
                done=done,
                truncated=truncated,
                termination_reason=reason,
                action_history=self.transcript.get_action_history(),
                patient_frustration=self.patient_simulator.frustration,
                booking_confirmed=action_dict.get("action_type") == "confirm_booking",
                escalated_urgent=action_dict.get("action_type") == "escalate_urgent",
                escalated_human=action_dict.get("action_type") == "escalate_human",
                escalation_turn=self._state.step_count if action_dict.get("action_type") in ["escalate_urgent", "escalate_human"] else None,
            )

            grade_report = self.grader.compute_final_score(
                episode_state=episode_state,
                scenario=self._scenario,
                action_history=self.transcript.get_action_history(),
            )

            # final_score is always in [0.0, 1.0] — this is what the hackathon reads
            self._final_score = grade_report.final_score
            self._state.final_score = grade_report.final_score
            self._state.termination_reason = reason
            self._state.grade_report = grade_report.model_dump()

        # Track cumulative reward (for RL training, not for hackathon scoring)
        self._state.cumulative_reward += reward
        
        # Update transcript
        self.transcript.add_turn(action_dict, patient_response.text, reward)
        
        # Build observation
        patient_flags = self.observation_builder.build_patient_flags(
            self._scenario.patient_profile,
            self.patient_simulator.frustration,
            self.patient_simulator.get_revealed_facts(),
        )
        
        clinic_state = self.observation_builder.build_clinic_state(
            self._available_slots,
            self._doctor_delay_minutes,
            self._urgent_queue_length,
        )
        
        # Build clinical history
        clinical_history = self.observation_builder.build_clinical_history_from_scenario(self._scenario)
        
        # Compute reflection token
        reflection_token = self.reflection_engine.compute_reflection_token(
            scenario=self._scenario,
            action_history=self.transcript.get_action_history(),
            revealed_facts=self.patient_simulator.get_revealed_facts(),
            available_slots=self._available_slots,
            urgent_queue_length=self._urgent_queue_length,
            turn_idx=self._state.step_count,
            max_turns=self._scenario.max_turns,
        )
        
        # Get memory vault summary
        memory_vault_summary = self.memory_vault.get_summary(patient_id)
        
        return self.observation_builder.build(
            task_level=self._scenario.task_level,
            turn_idx=self._state.step_count,
            max_turns=self._scenario.max_turns,
            patient_response=patient_response,
            conversation_summary=self.transcript.get_conversation_summary(),
            patient_flags=patient_flags,
            clinic_state=clinic_state,
            clinical_history=clinical_history,
            reflection_token=reflection_token,
            memory_vault_summary=memory_vault_summary,
            privacy_risk_mask=self.observation_builder.build_privacy_risk_mask(
                self.transcript.get_action_history()
            ),
            done=done,
            reward=reward,
        )
    
    def _build_clinic_state_dict(self) -> Dict[str, Any]:
        """Build clinic state dictionary for internal use."""
        return {
            "available_slots": self._available_slots,
            "doctor_delay_minutes": self._doctor_delay_minutes,
            "urgent_queue_length": self._urgent_queue_length,
        }
    
    @property
    def state(self) -> VoiceClinicState:
        """Return current episode state."""
        return self._state
    
    def close(self) -> None:
        """Clean up environment resources (required by OpenEnv)."""
        # No resources to clean up in this environment
        pass
