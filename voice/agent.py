"""
voice/agent.py
VoiceAgent — integrates TTS, STT, RAG, state machine, and persistence.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from deepgram import DeepgramClient, SpeakOptions
from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from core.config import (
    DEEPGRAM_API_KEY,
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    SYSTEM_INSTRUCTIONS,
    TTS_OPTIONS,
)
from rag.rag_service import get_rag_service
from services.function_handler import OPENAI_FUNCTIONS, handle_function_call
from voice.persistence import ConversationPersistence
from voice.state_manager import ConversationState, StateManager

logger = logging.getLogger(__name__)


class ImprovedVoiceAgent:
    """
    Full voice agent with RAG, state management, and conversation persistence.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        enable_rag: bool = True,
        enable_persistence: bool = True,
        enable_state_management: bool = True,
    ) -> None:
        # Core clients
        self.deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        self.openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY, timeout=30.0, max_retries=2
        )

        self.session_id = session_id or f"session_{int(datetime.now().timestamp())}"
        self.conversation_history: List[Dict[str, Any]] = []
        self.tts_options = TTS_OPTIONS

        # ---- RAG ----
        self.enable_rag = enable_rag
        self.rag_service = None
        if enable_rag:
            try:
                self.rag_service = get_rag_service()
                logger.info("RAG service enabled")
            except Exception as exc:
                logger.warning("RAG service unavailable: %s", exc)
                self.enable_rag = False

        # ---- State management ----
        self.enable_state_management = enable_state_management
        self.state_manager: Optional[StateManager] = (
            StateManager() if enable_state_management else None
        )

        # ---- Persistence ----
        self.enable_persistence = enable_persistence
        self.persistence: Optional[ConversationPersistence] = None
        if enable_persistence:
            try:
                self.persistence = ConversationPersistence()
                loaded = self.persistence.load_conversation(self.session_id)
                if loaded:
                    self.conversation_history = loaded.get("conversation_history", [])
                    if self.state_manager and loaded.get("state"):
                        try:
                            self.state_manager.current_state = ConversationState(loaded["state"])
                            self.state_manager.collected_data = loaded.get("collected_data", {})
                            logger.info("Restored state: %s", loaded["state"])
                        except Exception as exc:
                            logger.warning("Could not restore state: %s", exc)
                    logger.info(
                        "Loaded conversation with %d messages",
                        len(self.conversation_history),
                    )
            except Exception as exc:
                logger.warning("Persistence unavailable: %s", exc)
                self.enable_persistence = False
                self.persistence = None

        logger.info("ImprovedVoiceAgent ready (session=%s)", self.session_id)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def text_to_speech(self, text: str, max_retries: int = 3) -> bytes:
        if not text or not text.strip():
            return b""

        for attempt in range(max_retries):
            try:
                options = SpeakOptions(
                    model=self.tts_options["model"],
                    encoding=self.tts_options["encoding"],
                    sample_rate=self.tts_options["sample_rate"],
                )
                response = self.deepgram.speak.v("1").stream({"text": text}, options)
                audio = b"".join(response.stream_memory)
                if audio:
                    logger.debug("TTS: %d bytes", len(audio))
                    return audio
            except Exception as exc:
                logger.error("TTS attempt %d failed: %s", attempt + 1, exc)
                if attempt < max_retries - 1:
                    await asyncio.sleep(attempt + 1)

        return b""

    # ------------------------------------------------------------------
    # RAG context enrichment
    # ------------------------------------------------------------------

    async def _enhance_with_rag(self, user_message: str) -> Optional[str]:
        if not self.enable_rag or not self.rag_service:
            return None
        try:
            if not await self.rag_service.should_use_rag(user_message):
                return None

            emergency = await self.rag_service.check_emergency(user_message)
            if emergency.get("is_emergency"):
                if self.state_manager:
                    self.state_manager.transition(
                        ConversationState.EMERGENCY_DETECTED,
                        data={"emergency_reason": user_message},
                        force=True,
                    )
                return (
                    f"\n URGENT MEDICAL SITUATION DETECTED:\n{emergency.get('message')}\n\n"
                    f"CRITICAL: {emergency.get('recommendation')}\n\n"
                    "Do NOT schedule a regular appointment. Direct patient to emergency services."
                )

            symptom_words = {"pain", "hurt", "symptom", "feel", "sick"}
            if any(w in user_message.lower() for w in symptom_words):
                rec = await self.rag_service.get_specialty_recommendation(user_message)
                if rec.get("specialties"):
                    context = (
                        f"\nMedical Knowledge Context:\n"
                        f"- Recommended specialties: {', '.join(rec['specialties'])}\n"
                        f"- Urgency: {rec.get('urgency', 'STANDARD')}\n"
                        f"- Confidence: {rec.get('confidence', 'medium')}\n"
                    )
                    if rec.get("supporting_info"):
                        context += f"\nSupporting info: {rec['supporting_info'][0][:200]}..."
                    if self.state_manager and self.state_manager.get_state() == ConversationState.COLLECTING_SYMPTOMS:
                        self.state_manager.add_data("recommended_specialties", rec["specialties"])
                        self.state_manager.add_data("urgency", rec["urgency"])
                    return context

            results = await self.rag_service.search_medical_knowledge(user_message, n_results=2)
            if results:
                return "\nMedical Knowledge Context:\n" + "".join(
                    f"{i}. {r['text'][:200]}...\n" for i, r in enumerate(results, 1)
                )
        except Exception as exc:
            logger.error("RAG enhancement error: %s", exc, exc_info=True)
        return None

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def _update_state(self, user_message: str, assistant_response: str) -> None:
        if not self.state_manager:
            return
        try:
            state = self.state_manager.get_state()
            msg = user_message.lower()
            resp = assistant_response.lower()

            if state == ConversationState.GREETING:
                if any(w in msg for w in ["pain", "hurt", "symptom", "sick", "feel"]):
                    self.state_manager.transition(ConversationState.COLLECTING_SYMPTOMS, data={"initial_complaint": user_message})
                elif any(w in msg for w in ["appointment", "schedule", "book"]):
                    self.state_manager.transition(ConversationState.CHECKING_AVAILABILITY)

            elif state == ConversationState.COLLECTING_SYMPTOMS:
                if "recommend" in resp or "specialist" in resp:
                    self.state_manager.transition(ConversationState.RECOMMENDING_SPECIALTY)

            elif state == ConversationState.RECOMMENDING_SPECIALTY:
                if any(w in msg for w in ["yes", "book", "schedule", "appointment"]):
                    self.state_manager.transition(ConversationState.CHECKING_AVAILABILITY)

            elif state == ConversationState.CHECKING_AVAILABILITY:
                if "name" in resp or "email" in resp:
                    self.state_manager.transition(ConversationState.COLLECTING_PATIENT_INFO)

            elif state == ConversationState.COLLECTING_PATIENT_INFO:
                if "confirm" in resp:
                    self.state_manager.transition(ConversationState.CONFIRMING_APPOINTMENT)

            elif state == ConversationState.CONFIRMING_APPOINTMENT:
                if "confirmed" in resp or "booked" in resp:
                    self.state_manager.transition(ConversationState.COMPLETED)

        except Exception as exc:
            logger.error("State update error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        prompt = SYSTEM_INSTRUCTIONS
        if self.state_manager:
            prompt += f"\n\nCurrent context: {self.state_manager.get_state_prompt_context()}"
            data = self.state_manager.get_all_data()
            if data:
                prompt += f"\n\nCollected info: {json.dumps(data, indent=2)}"
        return prompt

    # ------------------------------------------------------------------
    # Persistence helper — uses correct kwarg names
    # ------------------------------------------------------------------

    def _save_conversation(self) -> None:
        if not self.enable_persistence or not self.persistence:
            return
        try:
            state_str = self.state_manager.get_state().value if self.state_manager else "greeting"
            collected = self.state_manager.get_all_data() if self.state_manager else {}
            self.persistence.save_conversation(
                session_id=self.session_id,
                conversation_history=self.conversation_history,
                state_data={"current_state": state_str, "collected_data": collected},
            )
        except Exception as exc:
            logger.warning("Could not save conversation: %s", exc)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    async def process_with_openai(
        self, user_message: str, max_retries: int = 2
    ) -> str:
        if not user_message or not user_message.strip():
            return "I didn't catch that. Could you please repeat?"

        try:
            rag_context = await self._enhance_with_rag(user_message)
            if rag_context:
                self.conversation_history.append({"role": "system", "content": rag_context})

            self.conversation_history.append({"role": "user", "content": user_message})

            for attempt in range(max_retries):
                try:
                    messages = [
                        {"role": "system", "content": self._build_system_prompt()},
                        *self.conversation_history,
                    ]
                    response = await self.openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=messages,
                        tools=OPENAI_FUNCTIONS,
                        tool_choice="auto",
                        temperature=OPENAI_TEMPERATURE,
                        max_tokens=OPENAI_MAX_TOKENS,
                    )
                    msg = response.choices[0].message

                    if msg.tool_calls:
                        response_text = await self._handle_tool_calls(msg)
                    else:
                        response_text = msg.content or "I'm not sure how to respond to that."
                        self.conversation_history.append({"role": "assistant", "content": response_text})

                    self._update_state(user_message, response_text)
                    self._save_conversation()
                    return response_text

                except RateLimitError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                    else:
                        return "I'm experiencing high demand. Please try again shortly."
                except APITimeoutError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(attempt + 1)
                    else:
                        return "The request timed out. Please try again."
                except APIError as exc:
                    logger.error("OpenAI API error: %s", exc)
                    return "I'm having trouble processing your request. Please rephrase."

            return "I apologise, but I'm having technical difficulties. Please try again."

        except Exception as exc:
            logger.error("Unexpected error: %s", exc, exc_info=True)
            return "An unexpected error occurred. Please try again or contact support."

    async def _handle_tool_calls(self, assistant_message) -> str:
        tool_calls_data = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in assistant_message.tool_calls
        ]
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message.content, "tool_calls": tool_calls_data}
        )

        for tc in assistant_message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
                result = {"error": "Invalid JSON in function arguments"}
            else:
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(handle_function_call, tc.function.name, args),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    result = {"error": f"{tc.function.name} timed out"}
                except Exception as exc:
                    result = {"error": str(exc)}

            self.conversation_history.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
            )

        follow_up = await self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                *self.conversation_history,
            ],
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
        )
        final = follow_up.choices[0].message.content or "I've processed your request."
        self.conversation_history.append({"role": "assistant", "content": final})
        return final

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def reset_conversation(self) -> None:
        self.conversation_history = []
        if self.state_manager:
            self.state_manager.reset()
        logger.info("Conversation reset")

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "session_id": self.session_id,
            "conversation_length": len(self.conversation_history),
            "rag_enabled": self.enable_rag,
            "persistence_enabled": self.enable_persistence,
            "state_management_enabled": self.enable_state_management,
        }
        if self.state_manager:
            stats["state"] = self.state_manager.get_state().value
            stats["collected_data"] = self.state_manager.get_all_data()
        return stats
