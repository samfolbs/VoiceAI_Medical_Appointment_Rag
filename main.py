"""
Main Server 
FastAPI server with WebSocket support for voice interaction
Enhanced with RAG integration for medical knowledge
"""
import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    DeepgramClientOptions
)

from config import SERVER_HOST, SERVER_PORT, STT_OPTIONS, DEEPGRAM_API_KEY
from improved_voice_agent import ImprovedVoiceAgent
from rag_service import get_rag_service, SAMPLE_MEDICAL_KNOWLEDGE
from templates import get_html_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global clients
deepgram_client: Optional[DeepgramClient] = None
rag_initialized: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown with RAG initialization"""
    global deepgram_client, rag_initialized
    
    # Startup
    logger.info("Starting Medical Appointment Voice Assistant with RAG...")
    try:
        # Initialize Deepgram client
        config = DeepgramClientOptions(
            options={"keepalive": "true"}
        )
        deepgram_client = DeepgramClient(DEEPGRAM_API_KEY, config)
        logger.info("Deepgram client initialized successfully")
        
        # Initialize RAG service
        try:
            logger.info("Initializing RAG service...")
            rag_service = get_rag_service()
            
            # Check if knowledge base is empty and initialize if needed
            stats = rag_service.get_collection_stats()
            if stats.get('total_documents', 0) == 0:
                logger.info("Knowledge base empty, initializing with sample data...")
                success = await rag_service.add_medical_knowledge(SAMPLE_MEDICAL_KNOWLEDGE)
                if success:
                    logger.info("RAG knowledge base initialized successfully")
                    rag_initialized = True
                else:
                    logger.warning("Failed to initialize RAG knowledge base")
            else:
                logger.info(f"RAG knowledge base loaded: {stats.get('total_documents')} documents")
                rag_initialized = True
                
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            logger.info("Continuing without RAG support")
            rag_initialized = False
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Medical Appointment Voice Assistant...")


app = FastAPI(
    title="Medical Appointment Voice Assistant with RAG",
    description="Voice-enabled medical appointment scheduling with intelligent knowledge retrieval",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_home():
    """Serve the home page with voice interface"""
    try:
        return HTMLResponse(content=get_html_template())
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        raise HTTPException(status_code=500, detail="Failed to load home page")


@app.get("/health")
async def health_check():
    """Health check endpoint with RAG status"""
    health_status = {
        "status": "healthy",
        "service": "Medical Appointment Voice Assistant",
        "deepgram_initialized": deepgram_client is not None,
        "rag_enabled": rag_initialized
    }
    
    # Add RAG statistics if available
    if rag_initialized:
        try:
            rag_service = get_rag_service()
            stats = rag_service.get_collection_stats()
            health_status["rag_stats"] = {
                "total_documents": stats.get('total_documents', 0),
                "embedding_model": stats.get('embedding_model', 'unknown')
            }
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
    
    return health_status


@app.get("/rag/stats")
async def get_rag_stats():
    """Get RAG service statistics"""
    if not rag_initialized:
        return {
            "enabled": False,
            "message": "RAG service not initialized"
        }
    
    try:
        rag_service = get_rag_service()
        stats = rag_service.get_collection_stats()
        return {
            "enabled": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RAG statistics")


@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice communication with RAG support
    Handles bidirectional audio streaming with Deepgram, OpenAI, and RAG
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    agent: Optional[ImprovedVoiceAgent] = None
    dg_connection = None
    
    try:
        # Validate Deepgram client
        if deepgram_client is None:
            raise RuntimeError("Deepgram client not initialized")
        
        # Initialize voice agent with RAG enabled
        agent = ImprovedVoiceAgent(enable_rag=rag_initialized)
        logger.info(f"Voice agent initialized (RAG: {rag_initialized})")
        
        # Initialize Deepgram STT connection
        dg_connection = deepgram_client.listen.websocket.v("1")
        
        # Flag to track if connection is active
        is_connected = True
        
        async def on_message(self, result, **kwargs):
            """Handle transcription results from Deepgram"""
            try:
                sentence = result.channel.alternatives[0].transcript
                
                if len(sentence) == 0:
                    return
                
                logger.info(f"User said: {sentence}")
                
                # Send transcript to client
                if is_connected:
                    await websocket.send_json({
                        "type": "transcript",
                        "text": sentence
                    })
                
                # Check if RAG will be used (for UI indication)
                rag_will_be_used = False
                if agent.enable_rag and agent.rag_service:
                    rag_will_be_used = await agent.rag_service.should_use_rag(sentence)
                    if rag_will_be_used:
                        logger.info("RAG will be used for this query")
                        # Notify client that RAG is being used
                        await websocket.send_json({
                            "type": "rag_status",
                            "using_rag": True
                        })
                
                # Process with OpenAI (RAG integration happens inside)
                response_text = await agent.process_with_openai(sentence)
                logger.info(f"Assistant response: {response_text}")
                
                # Send text response
                if is_connected:
                    await websocket.send_json({
                        "type": "response",
                        "text": response_text,
                        "used_rag": rag_will_be_used
                    })
                
                # Convert to speech
                audio_data = await agent.text_to_speech(response_text)
                
                # Send audio
                if audio_data and is_connected:
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    await websocket.send_json({
                        "type": "audio",
                        "audio": audio_base64
                    })
                    
            except Exception as e:
                logger.error(f"Error in transcription handler: {e}")
                if is_connected:
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to process audio"
                        })
                    except:
                        pass
        
        async def on_error(self, error, **kwargs):
            """Handle Deepgram errors"""
            logger.error(f"Deepgram error: {error}")
            if is_connected:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Speech recognition error"
                    })
                except:
                    pass
        
        async def on_close(self, close, **kwargs):
            """Handle Deepgram connection close"""
            logger.info("Deepgram connection closed")
        
        # Register event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        
        # Configure STT options
        options = LiveOptions(**STT_OPTIONS)
        
        # Start Deepgram connection
        if not await dg_connection.start(options):
            raise RuntimeError("Failed to start Deepgram connection")
        
        logger.info("Deepgram connection started successfully")
        
        # Send initial greeting with RAG status
        try:
            rag_status = " with intelligent medical knowledge" if rag_initialized else ""
            greeting = f"Hello! I'm your medical appointment assistant{rag_status}. How can I help you today?"
            await websocket.send_json({
                "type": "response",
                "text": greeting,
                "rag_enabled": rag_initialized
            })
            
            greeting_audio = await agent.text_to_speech(greeting)
            if greeting_audio:
                audio_base64 = base64.b64encode(greeting_audio).decode('utf-8')
                await websocket.send_json({
                    "type": "audio",
                    "audio": audio_base64
                })
        except Exception as e:
            logger.error(f"Error sending greeting: {e}")
        
        # Receive and forward audio data
        while is_connected:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                
                if dg_connection and len(data) > 0:
                    dg_connection.send(data)
                    
            except asyncio.TimeoutError:
                # Send keepalive
                logger.debug("Keepalive timeout - connection still active")
                continue
                
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                is_connected = False
                break
                
            except Exception as e:
                logger.error(f"Error receiving audio data: {e}")
                is_connected = False
                break
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
        
    except RuntimeError as e:
        logger.error(f"Runtime error in WebSocket: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
            
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": "An unexpected error occurred"
            })
        except:
            pass
            
    finally:
        # Cleanup
        logger.info("Cleaning up WebSocket connection")
        
        if dg_connection:
            try:
                await dg_connection.finish()
                logger.info("Deepgram connection finished")
            except Exception as e:
                logger.error(f"Error finishing Deepgram connection: {e}")
        
        if agent:
            try:
                agent.reset_conversation()
            except Exception as e:
                logger.error(f"Error resetting agent: {e}")
        
        try:
            await websocket.close()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"\n{'='*60}")
    print(f"  Medical Appointment Voice Assistant with RAG")
    print(f"{'='*60}")
    print(f"\n  Open http://{SERVER_HOST}:{SERVER_PORT} in your browser")
    print(f"  Ready to schedule appointments")
    print(f"  RAG-powered medical knowledge")
    print(f"\n  Health check: http://{SERVER_HOST}:{SERVER_PORT}/health")
    print(f"  RAG stats: http://{SERVER_HOST}:{SERVER_PORT}/rag/stats\n")
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        access_log=True
    )