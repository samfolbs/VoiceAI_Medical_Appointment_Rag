"""core — shared configuration and cross-cutting utilities."""
from .config import (
    OPENAI_API_KEY,
    DEEPGRAM_API_KEY,
    SERVER_HOST,
    SERVER_PORT,
    TTS_OPTIONS,
    STT_OPTIONS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OPENAI_MAX_TOKENS,
    RAG_CONFIG,
    PERSISTENCE_CONFIG,
    SYSTEM_INSTRUCTIONS,
    MOCK_DATABASE,
    validate_config,
)

__all__ = [
    "OPENAI_API_KEY",
    "DEEPGRAM_API_KEY",
    "SERVER_HOST",
    "SERVER_PORT",
    "TTS_OPTIONS",
    "STT_OPTIONS",
    "OPENAI_MODEL",
    "OPENAI_TEMPERATURE",
    "OPENAI_MAX_TOKENS",
    "RAG_CONFIG",
    "PERSISTENCE_CONFIG",
    "SYSTEM_INSTRUCTIONS",
    "MOCK_DATABASE",
    "validate_config",
]
