"""
Improved Voice Agent
Integrates all components: RAG, state management, persistence, caching
"""
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from deepgram import DeepgramClient, SpeakOptions
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from config import (
    DEEPGRAM_API_KEY,
    OPENAI_API_KEY,
    TTS_OPTIONS,
    SYSTEM_INSTRUCTIONS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS
)
from function_handler import handle_function_call, OPENAI_FUNCTIONS
from rag_service import get_rag_service
from state_manager import StateManager, ConversationState
from conversation_persistence import ConversationPersistence

logger = logging.getLogger(__name__)


class ImprovedVoiceAgent:
    """
    Enhanced voice agent with:
    - RAG integration for medical knowledge
    - State management for conversation flow
    - Conversation persistence
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        enable_rag: bool = True,
        enable_persistence: bool = True,
        enable_state_management: bool = True
    ):
        """
        Initialize improved voice agent
        
        Args:
            session_id: Optional session identifier
            enable_rag: Enable RAG for medical knowledge
            enable_persistence: Enable conversation persistence
            enable_state_management: Enable state machine
        """
        try:
            # Initialize core clients
            self.deepgram = DeepgramClient(DEEPGRAM_API_KEY)
            self.openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                timeout=30.0,
                max_retries=2
            )
            
            # Session management
            self.session_id = session_id or f"session_{int(datetime.now().timestamp())}"
            self.conversation_history: List[Dict[str, Any]] = []
            
            # TTS configuration
            self.tts_options = TTS_OPTIONS
            
            # RAG integration
            self.enable_rag = enable_rag
            if enable_rag:
                try:
                    self.rag_service = get_rag_service()
                    logger.info("RAG service enabled")
                except Exception as e:
                    logger.warning(f"RAG service failed to initialize: {e}")
                    self.enable_rag = False
                    self.rag_service = None
            else:
                self.rag_service = None
            
            # State management
            self.enable_state_management = enable_state_management
            if enable_state_management:
                self.state_manager = StateManager()
                logger.info("State management enabled")
            else:
                self.state_manager = None
            
            # Persistence
            self.enable_persistence = enable_persistence
            if enable_persistence:
                try:
                    self.persistence = ConversationPersistence()
                    # Try to load existing conversation
                    loaded_data = self.persistence.load_conversation(self.session_id)
                    if loaded_data:
                        self.conversation_history = loaded_data.get("history", [])
                        if self.state_manager and "state" in loaded_data:
                            # Restore state
                            try:
                                state_value = loaded_data["state"]
                                restored_state = ConversationState(state_value)
                                self.state_manager.current_state = restored_state
                                self.state_manager.collected_data = loaded_data.get("collected_data", {})
                                logger.info(f"Restored conversation state: {state_value}")
                            except Exception as e:
                                logger.warning(f"Could not restore state: {e}")
                        logger.info(f"Loaded conversation with {len(self.conversation_history)} messages")
                except Exception as e:
                    logger.warning(f"Persistence failed to initialize: {e}")
                    self.enable_persistence = False
                    self.persistence = None
            else:
                self.persistence = None
            
            logger.info(f"ImprovedVoiceAgent initialized (session: {self.session_id})")
            
        except Exception as e:
            logger.error(f"Failed to initialize ImprovedVoiceAgent: {e}", exc_info=True)
            raise
    
    async def text_to_speech(self, text: str, max_retries: int = 3) -> bytes:
        """
        Convert text to speech using Deepgram with retry logic
        
        Args:
            text: Text to convert
            max_retries: Maximum retry attempts
            
        Returns:
            Audio data as bytes
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return b""
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"TTS attempt {attempt + 1} for text: {text[:50]}...")
                
                options = SpeakOptions(
                    model=self.tts_options["model"],
                    encoding=self.tts_options["encoding"],
                    sample_rate=self.tts_options["sample_rate"]
                )
                
                response = self.deepgram.speak.v("1").stream(
                    {"text": text},
                    options
                )
                
                audio_data = b""
                for chunk in response.stream_memory:
                    audio_data += chunk
                
                if len(audio_data) > 0:
                    logger.info(f"TTS successful, generated {len(audio_data)} bytes")
                    return audio_data
                    
            except Exception as e:
                logger.error(f"TTS attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
        
        return b""
    
    async def _enhance_with_rag(self, user_message: str) -> Optional[str]:
        """
        Enhance conversation with RAG context
        
        Args:
            user_message: User's message
            
        Returns:
            RAG context string or None
        """
        if not self.enable_rag or not self.rag_service:
            return None
        
        try:
            # Check if RAG should be used
            should_use = await self.rag_service.should_use_rag(user_message)
            if not should_use:
                return None
            
            logger.info("Using RAG for medical knowledge enhancement")
            
            # Check for emergency
            emergency_check = await self.rag_service.check_emergency(user_message)
            if emergency_check.get("is_emergency"):
                # Update state to emergency
                if self.state_manager:
                    self.state_manager.transition(
                        ConversationState.EMERGENCY_DETECTED,
                        data={"emergency_reason": user_message},
                        force=True
                    )
                
                return f"""
 URGENT MEDICAL SITUATION DETECTED:
{emergency_check.get('message')}

CRITICAL: {emergency_check.get('recommendation')}

Do NOT attempt to schedule a regular appointment. Direct the patient to emergency services IMMEDIATELY.
"""
            
            # Get specialty recommendation if symptoms mentioned
            if any(word in user_message.lower() for word in ['pain', 'hurt', 'symptom', 'feel', 'sick']):
                specialty_rec = await self.rag_service.get_specialty_recommendation(user_message)
                
                if specialty_rec.get('specialties'):
                    specialties_str = ', '.join(specialty_rec['specialties'])
                    context = f"""
Medical Knowledge Context:
- Based on described symptoms, recommended specialties: {specialties_str}
- Urgency level: {specialty_rec.get('urgency', 'standard')}
- Confidence: {specialty_rec.get('confidence', 'medium')}
"""
                    if specialty_rec.get('supporting_info'):
                        context += f"\nSupporting information: {specialty_rec['supporting_info'][0][:200]}..."
                    
                    # Update state if in symptom collection
                    if self.state_manager and self.state_manager.get_state() == ConversationState.COLLECTING_SYMPTOMS:
                        self.state_manager.add_data("recommended_specialties", specialty_rec['specialties'])
                        self.state_manager.add_data("urgency", specialty_rec['urgency'])
                    
                    return context
            
            # Search for relevant medical knowledge
            results = await self.rag_service.search_medical_knowledge(
                query=user_message,
                n_results=2
            )
            
            if results:
                context = "\nMedical Knowledge Context:\n"
                for i, result in enumerate(results[:2], 1):
                    context += f"{i}. {result['text'][:200]}...\n"
                return context
            
            return None
            
        except Exception as e:
            logger.error(f"Error enhancing with RAG: {e}", exc_info=True)
            return None
    
    def _update_state_from_message(self, user_message: str, assistant_response: str):
        """
        Update conversation state based on message content
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
        """
        if not self.state_manager:
            return
        
        try:
            current_state = self.state_manager.get_state()
            message_lower = user_message.lower()
            
            # State transition logic
            if current_state == ConversationState.GREETING:
                if any(word in message_lower for word in ['pain', 'hurt', 'symptom', 'sick', 'feel']):
                    self.state_manager.transition(
                        ConversationState.COLLECTING_SYMPTOMS,
                        data={"initial_complaint": user_message}
                    )
                elif any(word in message_lower for word in ['appointment', 'schedule', 'book']):
                    self.state_manager.transition(ConversationState.CHECKING_AVAILABILITY)
            
            elif current_state == ConversationState.COLLECTING_SYMPTOMS:
                if "recommend" in assistant_response.lower() or "specialist" in assistant_response.lower():
                    self.state_manager.transition(ConversationState.RECOMMENDING_SPECIALTY)
            
            elif current_state == ConversationState.RECOMMENDING_SPECIALTY:
                if any(word in message_lower for word in ['yes', 'book', 'schedule', 'appointment']):
                    self.state_manager.transition(ConversationState.CHECKING_AVAILABILITY)
            
            elif current_state == ConversationState.CHECKING_AVAILABILITY:
                if "name" in assistant_response.lower() or "email" in assistant_response.lower():
                    self.state_manager.transition(ConversationState.COLLECTING_PATIENT_INFO)
            
            elif current_state == ConversationState.COLLECTING_PATIENT_INFO:
                if "confirm" in assistant_response.lower():
                    self.state_manager.transition(ConversationState.CONFIRMING_APPOINTMENT)
            
            elif current_state == ConversationState.CONFIRMING_APPOINTMENT:
                if "confirmed" in assistant_response.lower() or "booked" in assistant_response.lower():
                    self.state_manager.transition(ConversationState.COMPLETED)
            
        except Exception as e:
            logger.error(f"Error updating state: {e}", exc_info=True)
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt with state context
        
        Returns:
            Enhanced system prompt
        """
        base_prompt = SYSTEM_INSTRUCTIONS
        
        if self.state_manager:
            state_context = self.state_manager.get_state_prompt_context()
            base_prompt += f"\n\nCurrent Conversation Context: {state_context}"
            
            collected_data = self.state_manager.get_all_data()
            if collected_data:
                base_prompt += f"\n\nCollected Information: {json.dumps(collected_data, indent=2)}"
        
        return base_prompt
    
    async def process_with_openai(
        self,
        user_message: str,
        max_retries: int = 2
    ) -> str:
        """
        Process user message with OpenAI and all enhancements
        
        Args:
            user_message: User's message
            max_retries: Maximum retry attempts
            
        Returns:
            Assistant's response text
        """
        if not user_message or not user_message.strip():
            return "I didn't catch that. Could you please repeat?"
        
        try:
            # Get RAG context if applicable
            rag_context = await self._enhance_with_rag(user_message)
            
            # Add RAG context if available
            if rag_context:
                self.conversation_history.append({
                    "role": "system",
                    "content": rag_context
                })
            
            # Add user message
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Process with retries
            for attempt in range(max_retries):
                try:
                    # Build messages with enhanced system prompt
                    messages = [
                        {"role": "system", "content": self._build_system_prompt()},
                        *self.conversation_history
                    ]
                    
                    # First API call
                    response = await self.openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=messages,
                        tools=OPENAI_FUNCTIONS,
                        tool_choice="auto",
                        temperature=OPENAI_TEMPERATURE,
                        max_tokens=OPENAI_MAX_TOKENS
                    )
                    
                    assistant_message = response.choices[0].message
                    
                    # Handle function calls
                    if assistant_message.tool_calls:
                        response_text = await self._handle_function_calls(assistant_message)
                    else:
                        response_text = assistant_message.content or "I'm not sure how to respond to that."
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": response_text
                        })
                    
                    # Update state
                    self._update_state_from_message(user_message, response_text)
                    
                    # Save conversation if persistence enabled
                    if self.enable_persistence and self.persistence:
                        try:
                            state_value = self.state_manager.get_state().value if self.state_manager else "unknown"
                            collected_data = self.state_manager.get_all_data() if self.state_manager else {}
                            
                            self.persistence.save_conversation(
                                session_id=self.session_id,
                                history=self.conversation_history,
                                state=state_value,
                                collected_data=collected_data
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save conversation: {e}")
                    
                    return response_text
                    
                except RateLimitError as e:
                    logger.warning(f"Rate limit error: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    else:
                        return "I'm experiencing high demand right now. Please try again in a moment."
                        
                except APITimeoutError as e:
                    logger.warning(f"Timeout error: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))
                    else:
                        return "The request took too long. Please try again."
                        
                except APIError as e:
                    logger.error(f"OpenAI API error: {e}")
                    return "I'm having trouble processing your request. Please try rephrasing."
            
            return "I apologize, but I'm having technical difficulties. Please try again."
            
        except Exception as e:
            logger.error(f"Unexpected error in process_with_openai: {e}", exc_info=True)
            return "I encountered an unexpected error. Please try again or contact support if the issue persists."
    
    async def _handle_function_calls(self, assistant_message) -> str:
        """
        Handle function calls from OpenAI response
        
        Args:
            assistant_message: Assistant message with tool calls
            
        Returns:
            Final response text
        """
        try:
            # Convert tool calls to serializable format
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in assistant_message.tool_calls
            ]
            
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": tool_calls_data
            })
            
            # Execute function calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in function arguments: {e}")
                    function_response = {"error": "Invalid function arguments format"}
                else:
                    logger.info(f"Calling function: {function_name}")
                    
                    try:
                        function_response = await asyncio.wait_for(
                            asyncio.to_thread(handle_function_call, function_name, function_args),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"Function {function_name} timed out")
                        function_response = {"error": f"Function {function_name} took too long"}
                    except Exception as e:
                        logger.error(f"Error executing function {function_name}: {e}")
                        function_response = {"error": f"Error executing {function_name}: {str(e)}"}
                
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_response)
                })
            
            # Get final response
            second_response = await self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    *self.conversation_history
                ],
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS
            )
            
            final_message = second_response.choices[0].message.content or "I've processed your request."
            self.conversation_history.append({
                "role": "assistant",
                "content": final_message
            })
            
            return final_message
            
        except Exception as e:
            logger.error(f"Error handling function calls: {e}", exc_info=True)
            return "I encountered an error while processing your request. Please try again."
    
    def reset_conversation(self):
        """Reset conversation history and state"""
        logger.info("Resetting conversation")
        self.conversation_history = []
        if self.state_manager:
            self.state_manager.reset()
    
    def get_conversation_length(self) -> int:
        """Get conversation length"""
        return len(self.conversation_history)
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        if not self.state_manager:
            return {"state": "unknown", "state_management_enabled": False}
        
        return {
            "state": self.state_manager.get_state().value,
            "collected_data": self.state_manager.get_all_data(),
            "valid_next_states": self.state_manager.get_next_states(),
            "should_escalate": self.state_manager.should_escalate(),
            "state_management_enabled": True
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            "session_id": self.session_id,
            "conversation_length": len(self.conversation_history),
            "rag_enabled": self.enable_rag,
            "persistence_enabled": self.enable_persistence,
            "state_management_enabled": self.enable_state_management
        }
        
        if self.state_manager:
            stats["state_info"] = self.get_state_info()
        
        if self.enable_rag and self.rag_service:
            try:
                stats["rag_stats"] = self.rag_service.get_collection_stats()
            except Exception as e:
                logger.warning(f"Could not get RAG stats: {e}")
        
        return stats