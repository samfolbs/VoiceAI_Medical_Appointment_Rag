"""
api/routes.py
FastAPI route definitions — HTTP endpoints and the voice WebSocket.
"""
import asyncio
import base64
import logging
from typing import Optional

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from api.templates import get_html_template
from voice.agent import ImprovedVoiceAgent

logger = logging.getLogger(__name__)

router = APIRouter()

# Injected by the app lifespan
_deepgram_client: Optional[DeepgramClient] = None
_rag_initialized: bool = False


def configure(deepgram_client: DeepgramClient, rag_initialized: bool) -> None:
    """Called at startup to inject shared resources."""
    global _deepgram_client, _rag_initialized
    _deepgram_client = deepgram_client
    _rag_initialized = rag_initialized


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    try:
        return HTMLResponse(content=get_html_template())
    except Exception as exc:
        logger.error("Error serving home page: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load home page")


@router.get("/health")
async def health_check():
    from rag.rag_service import get_rag_service

    status = {
        "status": "healthy",
        "service": "Medical Appointment Voice Assistant",
        "deepgram_initialized": _deepgram_client is not None,
        "rag_enabled": _rag_initialized,
    }
    if _rag_initialized:
        try:
            stats = get_rag_service().get_collection_stats()
            status["rag_stats"] = {
                "total_documents": stats.get("total_documents", 0),
                "embedding_model": stats.get("embedding_model", "unknown"),
            }
        except Exception as exc:
            logger.error("RAG stats error: %s", exc)
    return status


@router.get("/rag/stats")
async def rag_stats():
    from rag.rag_service import get_rag_service

    if not _rag_initialized:
        return {"enabled": False, "message": "RAG service not initialised"}
    try:
        return {"enabled": True, "stats": get_rag_service().get_collection_stats()}
    except Exception as exc:
        logger.error("RAG stats error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to get RAG statistics")


# ---------------------------------------------------------------------------
# WebSocket voice endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket accepted")

    agent: Optional[ImprovedVoiceAgent] = None
    dg_connection = None

    try:
        if _deepgram_client is None:
            raise RuntimeError("Deepgram client not initialised")

        from core.config import STT_OPTIONS

        agent = ImprovedVoiceAgent(enable_rag=_rag_initialized)
        dg_connection = _deepgram_client.listen.websocket.v("1")
        is_connected = True

        async def on_message(self, result, **kwargs):
            nonlocal is_connected
            try:
                sentence = result.channel.alternatives[0].transcript
                if not sentence:
                    return

                logger.info("User: %s", sentence)
                if is_connected:
                    await websocket.send_json({"type": "transcript", "text": sentence})

                rag_used = False
                if agent.enable_rag and agent.rag_service:
                    rag_used = await agent.rag_service.should_use_rag(sentence)
                    if rag_used:
                        await websocket.send_json({"type": "rag_status", "using_rag": True})

                response_text = await agent.process_with_openai(sentence)
                logger.info("Assistant: %s", response_text)

                if is_connected:
                    await websocket.send_json({"type": "response", "text": response_text, "used_rag": rag_used})

                audio = await agent.text_to_speech(response_text)
                if audio and is_connected:
                    await websocket.send_json(
                        {"type": "audio", "audio": base64.b64encode(audio).decode()}
                    )
            except Exception as exc:
                logger.error("Transcription handler error: %s", exc)
                if is_connected:
                    try:
                        await websocket.send_json({"type": "error", "message": "Failed to process audio"})
                    except Exception:
                        pass

        async def on_error(self, error, **kwargs):
            logger.error("Deepgram error: %s", error)
            if is_connected:
                try:
                    await websocket.send_json({"type": "error", "message": "Speech recognition error"})
                except Exception:
                    pass

        async def on_close(self, close, **kwargs):
            logger.info("Deepgram connection closed")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)

        options = LiveOptions(**STT_OPTIONS)
        if not await dg_connection.start(options):
            raise RuntimeError("Failed to start Deepgram STT connection")

        rag_suffix = " with intelligent medical knowledge" if _rag_initialized else ""
        greeting = f"Hello! I'm your medical appointment assistant{rag_suffix}. How can I help you today?"
        await websocket.send_json({"type": "response", "text": greeting, "rag_enabled": _rag_initialized})
        greeting_audio = await agent.text_to_speech(greeting)
        if greeting_audio:
            await websocket.send_json(
                {"type": "audio", "audio": base64.b64encode(greeting_audio).decode()}
            )

        while is_connected:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                if dg_connection and data:
                    dg_connection.send(data)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                is_connected = False
                break
            except Exception as exc:
                logger.error("Audio receive error: %s", exc)
                is_connected = False
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    except Exception as exc:
        logger.error("Unexpected WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": "An unexpected error occurred"})
        except Exception:
            pass
    finally:
        if dg_connection:
            try:
                await dg_connection.finish()
            except Exception:
                pass
        if agent:
            agent.reset_conversation()
        try:
            await websocket.close()
        except Exception:
            pass
