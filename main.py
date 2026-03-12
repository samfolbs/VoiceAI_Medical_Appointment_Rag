"""
main.py
Application entry point.
"""
import logging
import logging.config
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from deepgram import DeepgramClient, DeepgramClientOptions
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import DEEPGRAM_API_KEY, SERVER_HOST, SERVER_PORT, validate_config
from api.routes import configure as configure_routes, router
from rag.knowledge_base import SAMPLE_MEDICAL_KNOWLEDGE
from rag.rag_service import get_rag_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    logger.info("Starting Medical Appointment Voice Assistant with RAG...")

    # Validate API keys — raises ValueError with clear message if missing
    validate_config()

    deepgram_client: Optional[DeepgramClient] = None
    rag_initialized = False

    try:
        deepgram_client = DeepgramClient(
            DEEPGRAM_API_KEY,
            DeepgramClientOptions(options={"keepalive": "true"}),
        )
        logger.info("Deepgram client initialised")

        # RAG initialisation
        try:
            logger.info("Initialising RAG service...")
            rag = get_rag_service()
            stats = rag.get_collection_stats()
            if stats.get("total_documents", 0) == 0:
                logger.info("Seeding knowledge base with sample data...")
                ok = await rag.add_medical_knowledge(SAMPLE_MEDICAL_KNOWLEDGE)
                rag_initialized = ok
                logger.info("Knowledge base seeded: %s", ok)
            else:
                rag_initialized = True
                logger.info("Knowledge base loaded: %d docs", stats["total_documents"])
        except Exception as exc:
            logger.error("RAG init failed: %s — continuing without RAG", exc)

        configure_routes(deepgram_client, rag_initialized)

    except Exception as exc:
        logger.error("Startup error: %s", exc)
        raise

    yield

    # ---- Shutdown ----
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Appointment Voice Assistant",
    description="Voice-enabled appointment scheduling with RAG medical knowledge",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Medical Appointment Voice Assistant with RAG")
    print(f"{'='*60}")
    print(f"\n  Open http://{SERVER_HOST}:{SERVER_PORT} in your browser")
    print(f"  Health:   http://{SERVER_HOST}:{SERVER_PORT}/health")
    print(f"  RAG stats: http://{SERVER_HOST}:{SERVER_PORT}/rag/stats\n")

    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        access_log=True,
    )
