"""
Conversation State Management for effective state machine for appointment booking workflow
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Conversation states for appointment booking"""
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
    """Raised when invalid state transition is attempted"""
    pass


class StateManager:
    """
    Lightweight state management 
    Tracks conversation flow and validates transitions
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        ConversationState.GREETING: [
            ConversationState.COLLECTING_SYMPTOMS,
            ConversationState.CHECKING_AVAILABILITY,
            ConversationState.IDLE
        ],
        ConversationState.COLLECTING_SYMPTOMS: [
            ConversationState.EMERGENCY_DETECTED,
            ConversationState.RECOMMENDING_SPECIALTY,
            ConversationState.ERROR
        ],
        ConversationState.EMERGENCY_DETECTED: [
            ConversationState.COMPLETED,
            ConversationState.IDLE
        ],
        ConversationState.RECOMMENDING_SPECIALTY: [
            ConversationState.CHECKING_AVAILABILITY,
            ConversationState.COLLECTING_SYMPTOMS,
            ConversationState.ERROR
        ],
        ConversationState.CHECKING_AVAILABILITY: [
            ConversationState.COLLECTING_PATIENT_INFO,
            ConversationState.RECOMMENDING_SPECIALTY,
            ConversationState.ERROR
        ],
        ConversationState.COLLECTING_PATIENT_INFO: [
            ConversationState.CONFIRMING_APPOINTMENT,
            ConversationState.ERROR
        ],
        ConversationState.CONFIRMING_APPOINTMENT: [
            ConversationState.COMPLETED,
            ConversationState.COLLECTING_PATIENT_INFO,
            ConversationState.ERROR
        ],
        ConversationState.COMPLETED: [
            ConversationState.IDLE,
            ConversationState.GREETING
        ],
        ConversationState.ERROR: [
            ConversationState.GREETING,
            ConversationState.IDLE
        ],
        ConversationState.IDLE: [
            ConversationState.GREETING,
            ConversationState.COLLECTING_SYMPTOMS
        ]
    }
    
    def __init__(self):
        """Initialize state manager"""
        self.current_state = ConversationState.GREETING
        self.collected_data: Dict[str, Any] = {}
        self.transition_history: List[Dict[str, Any]] = []
        self.error_count = 0
        self.max_errors = 3
        
        logger.info("StateManager initialized")
    
    def get_state(self) -> ConversationState:
        """Get current state"""
        return self.current_state
    
    def can_transition(self, new_state: ConversationState) -> bool:
        """
        Check if transition to new state is valid
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition is allowed
        """
        valid_next_states = self.VALID_TRANSITIONS.get(self.current_state, [])
        return new_state in valid_next_states
    
    def transition(
        self, 
        new_state: ConversationState, 
        data: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool:
        """
        Transition to new state with validation
        
        Args:
            new_state: Target state
            data: Optional data to store with this state
            force: Force transition even if invalid (use carefully)
            
        Returns:
            True if transition successful
            
        Raises:
            StateTransitionError: If transition is invalid and not forced
        """
        try:
            # Validate transition
            if not force and not self.can_transition(new_state):
                error_msg = (
                    f"Invalid transition from {self.current_state.value} "
                    f"to {new_state.value}"
                )
                logger.warning(error_msg)
                raise StateTransitionError(error_msg)
            
            # Record transition
            transition_record = {
                'from_state': self.current_state.value,
                'to_state': new_state.value,
                'timestamp': datetime.now().isoformat(),
                'data': data or {},
                'forced': force
            }
            self.transition_history.append(transition_record)
            
            # Update state
            old_state = self.current_state
            self.current_state = new_state
            
            # Update collected data
            if data:
                self.collected_data.update(data)
            
            # Reset error count on successful transition (unless to ERROR state)
            if new_state != ConversationState.ERROR:
                self.error_count = 0
            else:
                self.error_count += 1
            
            logger.info(
                f"State transition: {old_state.value} → {new_state.value} "
                f"(forced: {force})"
            )
            
            return True
            
        except StateTransitionError:
            # Re-raise transition errors
            raise
        except Exception as e:
            logger.error(f"Error during state transition: {e}", exc_info=True)
            return False
    
    def add_data(self, key: str, value: Any) -> None:
        """
        Add data to collected information
        
        Args:
            key: Data key
            value: Data value
        """
        self.collected_data[key] = value
        logger.debug(f"Added data: {key} = {value}")
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """
        Get collected data
        
        Args:
            key: Data key
            default: Default value if key not found
            
        Returns:
            Data value or default
        """
        return self.collected_data.get(key, default)
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all collected data"""
        return self.collected_data.copy()
    
    def clear_data(self) -> None:
        """Clear all collected data"""
        self.collected_data.clear()
        logger.info("Cleared all collected data")
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get current state context for LLM
        
        Returns:
            Context dictionary with state information
        """
        return {
            'current_state': self.current_state.value,
            'collected_data': self.collected_data,
            'conversation_step': len(self.transition_history),
            'error_count': self.error_count,
            'can_proceed': self.error_count < self.max_errors
        }
    
    def get_next_states(self) -> List[str]:
        """
        Get list of valid next states
        
        Returns:
            List of valid state names
        """
        next_states = self.VALID_TRANSITIONS.get(self.current_state, [])
        return [state.value for state in next_states]
    
    def is_in_error_state(self) -> bool:
        """Check if in error state or exceeded max errors"""
        return (
            self.current_state == ConversationState.ERROR or 
            self.error_count >= self.max_errors
        )
    
    def reset(self) -> None:
        """Reset to initial state"""
        logger.info("Resetting state manager")
        self.current_state = ConversationState.GREETING
        self.collected_data.clear()
        self.transition_history.clear()
        self.error_count = 0
    
    def get_history_summary(self) -> List[str]:
        """
        Get human-readable transition history
        
        Returns:
            List of transition descriptions
        """
        summary = []
        for trans in self.transition_history:
            summary.append(
                f"{trans['from_state']} → {trans['to_state']} "
                f"at {trans['timestamp']}"
            )
        return summary
    
    def should_escalate(self) -> bool:
        """
        Determine if conversation should escalate to human
        
        Returns:
            True if escalation needed
        """
        return (
            self.error_count >= self.max_errors or
            self.current_state == ConversationState.EMERGENCY_DETECTED
        )
    
    def get_state_prompt_context(self) -> str:
        """
        Get state-specific context for system prompt
        
        Returns:
            Context string for LLM
        """
        contexts = {
            ConversationState.GREETING: (
                "You are greeting the patient. Be warm and ask how you can help."
            ),
            ConversationState.COLLECTING_SYMPTOMS: (
                "You are collecting symptom information. Ask clarifying questions "
                "about their symptoms, severity, and duration."
            ),
            ConversationState.EMERGENCY_DETECTED: (
                "EMERGENCY DETECTED. Direct the patient to call 911 or go to "
                "emergency room immediately. Do NOT schedule an appointment."
            ),
            ConversationState.RECOMMENDING_SPECIALTY: (
                "Based on symptoms, recommend appropriate medical specialty and "
                "explain why this specialist is best suited."
            ),
            ConversationState.CHECKING_AVAILABILITY: (
                "You are checking appointment availability. Offer available times "
                "and be flexible with patient preferences."
            ),
            ConversationState.COLLECTING_PATIENT_INFO: (
                "Collect patient information: name, date of birth, phone number, "
                "and insurance information."
            ),
            ConversationState.CONFIRMING_APPOINTMENT: (
                "Confirm all appointment details with patient and ask if they "
                "have any questions."
            ),
            ConversationState.COMPLETED: (
                "Appointment is confirmed. Provide confirmation details and "
                "preparation instructions."
            ),
            ConversationState.ERROR: (
                "There was an error. Apologize and try to help the patient or "
                "offer to transfer to a human representative."
            ),
            ConversationState.IDLE: (
                "Waiting for patient input. Be ready to help with any request."
            )
        }
        
        return contexts.get(
            self.current_state, 
            "Assist the patient with their medical appointment needs."
        )