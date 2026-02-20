"""
Configuration Module
Centralized configuration for the Medical Appointment RAG system
"""
import os
from typing import Dict, Any

# API Keys (set these as environment variables in production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Deepgram TTS Options
TTS_OPTIONS = {
    "model": "aura-asteria-en",
    "encoding": "linear16",
    "sample_rate": 24000
}

# Deepgram STT Options
STT_OPTIONS = {
    "model": "nova-2",
    "language": "en-US",
    "smart_format": True,
    "interim_results": False,
    "utterance_end_ms": 1000,
    "vad_events": True,
    "endpointing": 300
}

# OpenAI Model Configuration
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.7
OPENAI_MAX_TOKENS = 500

# RAG Configuration
RAG_CONFIG = {
    "collection_name": "medical_knowledge",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "max_results": 3,
    "similarity_threshold": 0.7
}

# Cache Configuration
CACHE_CONFIG = {
    "l1_size": 50,
    "l2_size": 500,
    "ttl_seconds": 3600
}

# Persistence Configuration
PERSISTENCE_CONFIG = {
    "db_path": "conversations.db",
    "max_sessions": 1000,
    "session_timeout_hours": 24
}

# System Instructions
SYSTEM_INSTRUCTIONS = """You are a professional medical appointment scheduling assistant.

Your responsibilities:
1. Greet patients warmly and professionally
2. Collect symptom information carefully and empathetically
3. CRITICAL: Detect medical emergencies and direct to 911/ER immediately
4. Recommend appropriate medical specialties based on symptoms
5. Check doctor availability and schedule appointments
6. Collect required patient information (name, email, phone, DOB, insurance)
7. Confirm all appointment details before finalizing

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
- Patient name
- Email address
- Phone number
- Date of birth
- Insurance provider
- Appointment date and time
- Doctor selection
- Reason for visit

Communication Style:
- Professional yet warm and empathetic
- Clear and concise
- Patient-focused
- Respectful of medical privacy
- Reassuring but not providing medical advice

When using medical knowledge context:
- Use the provided information to make better specialty recommendations
- Reference relevant medical information naturally
- Always prioritize patient safety and appropriate care level
"""

# Mock Database Configuration 
MOCK_DATABASE = {
    "doctors": [
        {
            "id": "dr_smith",
            "name": "Dr. Sarah Smith",
            "specialty": "Cardiology",
            "available_days": ["Monday", "Tuesday", "Thursday", "Friday"],
            "available_hours": ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]
        },
        {
            "id": "dr_johnson",
            "name": "Dr. Michael Johnson",
            "specialty": "Pediatrics",
            "available_days": ["Monday", "Wednesday", "Thursday", "Friday"],
            "available_hours": ["08:00", "09:00", "10:00", "13:00", "14:00", "15:00"]
        },
        {
            "id": "dr_williams",
            "name": "Dr. Emily Williams",
            "specialty": "Orthopedics",
            "available_days": ["Tuesday", "Wednesday", "Thursday"],
            "available_hours": ["09:00", "10:00", "11:00", "13:00", "14:00"]
        },
        {
            "id": "dr_brown",
            "name": "Dr. James Brown",
            "specialty": "Neurology",
            "available_days": ["Monday", "Tuesday", "Wednesday", "Friday"],
            "available_hours": ["10:00", "11:00", "13:00", "14:00", "15:00"]
        },
        {
            "id": "dr_davis",
            "name": "Dr. Jennifer Davis",
            "specialty": "General Practice",
            "available_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "available_hours": ["08:00", "09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]
        },
        {
            "id": "dr_martinez",
            "name": "Dr. Carlos Martinez",
            "specialty": "Dermatology",
            "available_days": ["Tuesday", "Thursday", "Friday"],
            "available_hours": ["09:00", "10:00", "14:00", "15:00", "16:00"]
        }
    ],
    "appointment_types": [
        {
            "id": "consultation",
            "name": "Initial Consultation",
            "duration_minutes": 30,
            "base_price": 150.00
        },
        {
            "id": "follow_up",
            "name": "Follow-up Visit",
            "duration_minutes": 20,
            "base_price": 100.00
        },
        {
            "id": "annual_checkup",
            "name": "Annual Checkup",
            "duration_minutes": 45,
            "base_price": 200.00
        },
        {
            "id": "urgent_care",
            "name": "Urgent Care",
            "duration_minutes": 30,
            "base_price": 175.00
        },
        {
            "id": "telehealth",
            "name": "Telehealth Consultation",
            "duration_minutes": 20,
            "base_price": 75.00
        }
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "medical_appointment.log",
            "formatter": "default",
            "level": "DEBUG"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}