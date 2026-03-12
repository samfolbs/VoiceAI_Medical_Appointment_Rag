"""
core/config.py
Centralised, validated configuration.
"""
import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv

# Load .env file 
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

# ---------------------------------------------------------------------------
# Deepgram TTS / STT
# ---------------------------------------------------------------------------
TTS_OPTIONS: Dict[str, Any] = {
    "model": "aura-asteria-en",
    "encoding": "linear16",
    "sample_rate": 24000,
}

STT_OPTIONS: Dict[str, Any] = {
    "model": "nova-2",
    "language": "en-US",
    "smart_format": True,
    "interim_results": False,
    "utterance_end_ms": 1000,
    "vad_events": True,
    "endpointing": 300,
}

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_MODEL: str = "gpt-4o"
OPENAI_TEMPERATURE: float = 0.7
OPENAI_MAX_TOKENS: int = 500

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
RAG_CONFIG: Dict[str, Any] = {
    "collection_name": "medical_knowledge",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "max_results": 3,
    "similarity_threshold": 0.7,
    # Cache sub-keys used by MultiLevelCache
    "l1_size": 50,
    "l2_size": 500,
    "ttl_seconds": 3600,
}

# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------
PERSISTENCE_CONFIG: Dict[str, Any] = {
    "storage_path": "./conversations",
    "cleanup_days": 30,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS: str = """You are a professional medical appointment scheduling assistant.

Your responsibilities:
1. Greet patients warmly and professionally
2. Collect symptom information carefully and empathetically
3. CRITICAL: Detect medical emergencies and direct to 911/ER immediately
4. Recommend appropriate medical specialties based on symptoms
5. Check doctor availability and schedule appointments
6. Collect required patient information (name, email, phone, DOB, insurance)
7. Confirm all appointment details before finalising

Emergency Detection:
- Chest pain, difficulty breathing, stroke symptoms → IMMEDIATE 911
- Severe bleeding, unconsciousness → EMERGENCY ROOM
- DO NOT schedule appointments for emergencies

Appointment Booking Process:
1. Understand patient's medical need
2. Recommend appropriate specialist if needed
3. Check doctor availability
4. Collect ALL required patient information
5. Confirm details before booking
6. Provide confirmation code and preparation instructions

Required Information Before Booking:
- Patient name, email, phone, date of birth, insurance provider
- Appointment date, time, doctor selection, reason for visit

Communication Style:
- Professional yet warm and empathetic
- Clear and concise; patient-focused
- Respectful of medical privacy
- Reassuring but not providing medical advice

When using medical knowledge context:
- Use the provided information to make better specialty recommendations
- Reference relevant medical information naturally
- Always prioritise patient safety and appropriate care level
"""

# ---------------------------------------------------------------------------
# Mock database
# ---------------------------------------------------------------------------
MOCK_DATABASE: Dict[str, Any] = {
    "doctors": [
        {
            "id": "dr_smith",
            "name": "Dr. Sarah Smith",
            "specialty": "Cardiology",
            "available_days": ["Monday", "Tuesday", "Thursday", "Friday"],
            "available_hours": ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"],
        },
        {
            "id": "dr_johnson",
            "name": "Dr. Michael Johnson",
            "specialty": "Pediatrics",
            "available_days": ["Monday", "Wednesday", "Thursday", "Friday"],
            "available_hours": ["08:00", "09:00", "10:00", "13:00", "14:00", "15:00"],
        },
        {
            "id": "dr_williams",
            "name": "Dr. Emily Williams",
            "specialty": "Orthopedics",
            "available_days": ["Tuesday", "Wednesday", "Thursday"],
            "available_hours": ["09:00", "10:00", "11:00", "13:00", "14:00"],
        },
        {
            "id": "dr_brown",
            "name": "Dr. James Brown",
            "specialty": "Neurology",
            "available_days": ["Monday", "Tuesday", "Wednesday", "Friday"],
            "available_hours": ["10:00", "11:00", "13:00", "14:00", "15:00"],
        },
        {
            "id": "dr_davis",
            "name": "Dr. Jennifer Davis",
            "specialty": "General Practice",
            "available_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "available_hours": ["08:00", "09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"],
        },
        {
            "id": "dr_martinez",
            "name": "Dr. Carlos Martinez",
            "specialty": "Dermatology",
            "available_days": ["Tuesday", "Thursday", "Friday"],
            "available_hours": ["09:00", "10:00", "14:00", "15:00", "16:00"],
        },
    ],
    "appointment_types": [
        {"id": "consultation", "name": "Initial Consultation", "duration_minutes": 30, "base_price": 150.00},
        {"id": "follow_up", "name": "Follow-up Visit", "duration_minutes": 20, "base_price": 100.00},
        {"id": "annual_checkup", "name": "Annual Checkup", "duration_minutes": 45, "base_price": 200.00},
        {"id": "urgent_care", "name": "Urgent Care", "duration_minutes": 30, "base_price": 175.00},
        {"id": "telehealth", "name": "Telehealth Consultation", "duration_minutes": 20, "base_price": 75.00},
    ],
}


def validate_config() -> None:
    """
    Raise a clear ValueError if required API keys are missing.
    Call this at application startup before initialising any services.
    """
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not DEEPGRAM_API_KEY:
        missing.append("DEEPGRAM_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your API keys."
        )
    logger.info("Configuration validated successfully.")
