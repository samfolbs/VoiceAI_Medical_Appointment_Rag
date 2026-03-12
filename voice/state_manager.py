"""
voice/state_manager.py
Conversation state machine for the appointment booking flow.
Unchanged logic, clean module.
"""
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    GREETING = "greeting"
    COLLECTING_SYMPTOMS = "collecting_symptoms"
    EMERGENCY_DETECTED = "emergency_detected"
    RECOMMENDING_SPECIALTY = "recommending_specialty"
    CHECKING_AVAILABILITY = "checking_availability"
    COLLECTING_PATIENT_INFO = "collecting_patient_info"
    CONFIRMING_APPOINTMENT = "confirming_appointment"
    COMPLETED = "completed"
    ERROR = "error"
    IDLE = "idle"


class StateTransitionError(Exception):
    pass


class StateManager:
    """Validates and tracks conversation state transitions."""

    VALID_TRANSITIONS: Dict[ConversationState, List[ConversationState]] = {
        ConversationState.GREETING: [
            ConversationState.COLLECTING_SYMPTOMS,
            ConversationState.CHECKING_AVAILABILITY,
            ConversationState.IDLE,
        ],
        ConversationState.COLLECTING_SYMPTOMS: [
            ConversationState.EMERGENCY_DETECTED,
            ConversationState.RECOMMENDING_SPECIALTY,
            ConversationState.ERROR,
        ],
        ConversationState.EMERGENCY_DETECTED: [
            ConversationState.COMPLETED,
            ConversationState.IDLE,
        ],
        ConversationState.RECOMMENDING_SPECIALTY: [
            ConversationState.CHECKING_AVAILABILITY,
            ConversationState.COLLECTING_SYMPTOMS,
            ConversationState.ERROR,
        ],
        ConversationState.CHECKING_AVAILABILITY: [
            ConversationState.COLLECTING_PATIENT_INFO,
            ConversationState.RECOMMENDING_SPECIALTY,
            ConversationState.ERROR,
        ],
        ConversationState.COLLECTING_PATIENT_INFO: [
            ConversationState.CONFIRMING_APPOINTMENT,
            ConversationState.ERROR,
        ],
        ConversationState.CONFIRMING_APPOINTMENT: [
            ConversationState.COMPLETED,
            ConversationState.COLLECTING_PATIENT_INFO,
            ConversationState.ERROR,
        ],
        ConversationState.COMPLETED: [
            ConversationState.IDLE,
            ConversationState.GREETING,
        ],
        ConversationState.ERROR: [
            ConversationState.GREETING,
            ConversationState.IDLE,
        ],
        ConversationState.IDLE: [
            ConversationState.GREETING,
            ConversationState.COLLECTING_SYMPTOMS,
        ],
    }

    _STATE_PROMPTS: Dict[ConversationState, str] = {
        ConversationState.GREETING: "Greet the patient warmly and ask how you can help.",
        ConversationState.COLLECTING_SYMPTOMS: (
            "Collect symptom details. Ask clarifying questions about severity and duration."
        ),
        ConversationState.EMERGENCY_DETECTED: (
            "EMERGENCY DETECTED. Direct to 911 / ER immediately. Do NOT schedule."
        ),
        ConversationState.RECOMMENDING_SPECIALTY: (
            "Recommend the appropriate specialist based on symptoms."
        ),
        ConversationState.CHECKING_AVAILABILITY: (
            "Check appointment availability. Offer times and be flexible."
        ),
        ConversationState.COLLECTING_PATIENT_INFO: (
            "Collect name, DOB, phone, email, and insurance details."
        ),
        ConversationState.CONFIRMING_APPOINTMENT: (
            "Confirm all appointment details. Ask if the patient has questions."
        ),
        ConversationState.COMPLETED: (
            "Appointment confirmed. Provide confirmation details and prep instructions."
        ),
        ConversationState.ERROR: (
            "Apologise for the issue. Offer to transfer to a human representative."
        ),
        ConversationState.IDLE: "Wait for patient input.",
    }

    def __init__(self) -> None:
        self.current_state = ConversationState.GREETING
        self.collected_data: Dict[str, Any] = {}
        self.transition_history: List[Dict[str, Any]] = []
        self.error_count = 0
        self.max_errors = 3

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_state(self) -> ConversationState:
        return self.current_state

    def can_transition(self, new_state: ConversationState) -> bool:
        return new_state in self.VALID_TRANSITIONS.get(self.current_state, [])

    def get_next_states(self) -> List[str]:
        return [s.value for s in self.VALID_TRANSITIONS.get(self.current_state, [])]

    def get_state_prompt_context(self) -> str:
        return self._STATE_PROMPTS.get(
            self.current_state, "Assist the patient with their appointment needs."
        )

    def get_all_data(self) -> Dict[str, Any]:
        return self.collected_data.copy()

    def get_data(self, key: str, default: Any = None) -> Any:
        return self.collected_data.get(key, default)

    def should_escalate(self) -> bool:
        return (
            self.error_count >= self.max_errors
            or self.current_state == ConversationState.EMERGENCY_DETECTED
        )

    def is_in_error_state(self) -> bool:
        return (
            self.current_state == ConversationState.ERROR
            or self.error_count >= self.max_errors
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def transition(
        self,
        new_state: ConversationState,
        data: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        if not force and not self.can_transition(new_state):
            msg = f"Invalid transition {self.current_state.value} → {new_state.value}"
            logger.warning(msg)
            raise StateTransitionError(msg)

        self.transition_history.append(
            {
                "from": self.current_state.value,
                "to": new_state.value,
                "at": datetime.now().isoformat(),
                "forced": force,
            }
        )
        old = self.current_state
        self.current_state = new_state
        if data:
            self.collected_data.update(data)
        if new_state == ConversationState.ERROR:
            self.error_count += 1
        else:
            self.error_count = 0

        logger.info("State: %s → %s (forced=%s)", old.value, new_state.value, force)
        return True

    def add_data(self, key: str, value: Any) -> None:
        self.collected_data[key] = value

    def clear_data(self) -> None:
        self.collected_data.clear()

    def reset(self) -> None:
        self.current_state = ConversationState.GREETING
        self.collected_data.clear()
        self.transition_history.clear()
        self.error_count = 0
        logger.info("StateManager reset")
