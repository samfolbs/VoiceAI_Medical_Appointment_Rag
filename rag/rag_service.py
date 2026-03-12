"""
rag/rag_service.py
Medical knowledge retrieval using ChromaDB and OpenAI embeddings.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI

from core.config import OPENAI_API_KEY, RAG_CONFIG
from rag.cache import MultiLevelCache
from rag.knowledge_base import SAMPLE_MEDICAL_KNOWLEDGE

logger = logging.getLogger(__name__)

_instance: Optional["MedicalRAGService"] = None


class MedicalRAGService:
    """RAG service for medical knowledge retrieval."""

    def __init__(self, enable_cache: bool = True) -> None:
        try:
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self.collection_name: str = RAG_CONFIG["collection_name"]
            self.embedding_model: str = RAG_CONFIG["embedding_model"]

            # Get or create ChromaDB collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info("Loaded existing collection: %s", self.collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Medical knowledge base"},
                )
                logger.info("Created new collection: %s", self.collection_name)

            #      core/config.py calls load_dotenv() at import time.
            self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

            self.enable_cache = enable_cache
            self.cache: Optional[MultiLevelCache] = None
            if enable_cache:
                self.cache = MultiLevelCache(
                    l1_size=RAG_CONFIG.get("l1_size", 50),
                    l2_size=RAG_CONFIG.get("l2_size", 500),
                    ttl_seconds=RAG_CONFIG.get("ttl_seconds", 3600),
                )
                logger.info("Embedding cache enabled")

            logger.info("MedicalRAGService initialised successfully")

        except Exception as exc:
            logger.error("Failed to initialise RAG service: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    async def _get_embedding(self, text: str) -> List[float]:
        """Return an embedding vector, using cache when possible."""
        if self.enable_cache and self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                logger.debug("Embedding cache hit")
                return cached

        response = await self.openai_client.embeddings.create(
            input=text, model=self.embedding_model
        )
        embedding: List[float] = response.data[0].embedding

        if self.enable_cache and self.cache:
            self.cache.set(text, embedding)

        return embedding

    # ------------------------------------------------------------------
    # Knowledge-base management
    # ------------------------------------------------------------------

    async def add_medical_knowledge(
        self, knowledge_items: List[Dict[str, Any]]
    ) -> bool:
        """Add a list of knowledge items to the ChromaDB collection."""
        if not knowledge_items:
            logger.warning("No knowledge items to add")
            return False

        ids, documents, metadatas, embeddings = [], [], [], []

        for item in knowledge_items:
            ids.append(item["id"])
            documents.append(item["content"])
            metadatas.append(
                {
                    "specialty": item.get("specialty", "General"),
                    "urgency": item.get("urgency", "STANDARD"),
                    "keywords": ",".join(item.get("keywords", [])),
                }
            )
            embeddings.append(await self._get_embedding(item["content"]))

        self.collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        logger.info("Added %d items to knowledge base", len(knowledge_items))
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def search_medical_knowledge(
        self,
        query: str,
        n_results: int = 3,
        specialty_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base and return ranked results."""
        try:
            query_embedding = await self._get_embedding(query)
            where_clause = {"specialty": specialty_filter} if specialty_filter else None

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
            )

            formatted: List[Dict[str, Any]] = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    formatted.append(
                        {
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": (
                                results["distances"][0][i]
                                if "distances" in results
                                else None
                            ),
                        }
                    )
            return formatted

        except Exception as exc:
            logger.error("Error searching knowledge: %s", exc, exc_info=True)
            return []

    async def check_emergency(self, symptoms: str) -> Dict[str, Any]:
        """Return emergency status dict for a symptom description."""
        try:
            results = await self.search_medical_knowledge(symptoms, n_results=3)
            for result in results:
                if result["metadata"].get("urgency") == "EMERGENCY":
                    return {
                        "is_emergency": True,
                        "message": result["text"],
                        "recommendation": "CALL 911 IMMEDIATELY or go to the nearest emergency room.",
                    }
            return {
                "is_emergency": False,
                "message": "No immediate emergency detected, but please consult a healthcare provider.",
            }
        except Exception as exc:
            logger.error("Error checking emergency: %s", exc, exc_info=True)
            return {"is_emergency": False, "error": "Unable to determine emergency status"}

    async def get_specialty_recommendation(
        self, symptoms: str, n_results: int = 3
    ) -> Dict[str, Any]:
        """Return specialty recommendations and urgency level for given symptoms."""
        try:
            results = await self.search_medical_knowledge(symptoms, n_results=n_results)
            if not results:
                return {
                    "specialties": ["General Practice"],
                    "urgency": "STANDARD",
                    "confidence": "low",
                    "message": "Consider starting with a general practitioner.",
                }

            specialties = set()
            max_urgency = "ROUTINE"
            urgency_rank = {"EMERGENCY": 4, "URGENT": 3, "STANDARD": 2, "ROUTINE": 1}

            for result in results:
                specialties.add(result["metadata"].get("specialty", "General Practice"))
                urgency = result["metadata"].get("urgency", "STANDARD")
                if urgency_rank.get(urgency, 0) > urgency_rank.get(max_urgency, 0):
                    max_urgency = urgency

            if max_urgency == "EMERGENCY":
                return {
                    "specialties": list(specialties),
                    "urgency": "EMERGENCY",
                    "confidence": "high",
                    "message": "EMERGENCY: Call 911 or go to emergency room immediately!",
                    "supporting_info": [results[0]["text"]],
                }

            return {
                "specialties": list(specialties),
                "urgency": max_urgency,
                "confidence": "medium" if len(results) > 1 else "low",
                "message": f"Recommended specialist(s): {', '.join(specialties)}",
                "supporting_info": [r["text"] for r in results[:2]],
            }

        except Exception as exc:
            logger.error("Error getting specialty recommendation: %s", exc, exc_info=True)
            return {
                "specialties": ["General Practice"],
                "urgency": "STANDARD",
                "confidence": "low",
                "error": str(exc),
            }

    async def should_use_rag(self, message: str) -> bool:
        """Return True if the message is likely a medical/symptom query."""
        medical_keywords = [
            "pain", "hurt", "symptom", "sick", "fever", "cough", "ache",
            "specialist", "doctor", "medical", "condition", "diagnosis",
            "treatment", "medication", "injury", "illness",
        ]
        return any(kw in message.lower() for kw in medical_keywords)

    # ------------------------------------------------------------------
    # Stats / admin
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            stats: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "total_documents": self.collection.count(),
                "embedding_model": self.embedding_model,
                "cache_enabled": self.enable_cache,
            }
            if self.enable_cache and self.cache:
                stats["cache_stats"] = self.cache.get_stats()
            return stats
        except Exception as exc:
            logger.error("Error getting stats: %s", exc, exc_info=True)
            return {"error": str(exc)}

    def clear_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Medical knowledge base"},
            )
            logger.info("Collection cleared")
            return True
        except Exception as exc:
            logger.error("Error clearing collection: %s", exc, exc_info=True)
            return False


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_rag_service() -> MedicalRAGService:
    """Return (or create) the global RAG service singleton."""
    global _instance
    if _instance is None:
        _instance = MedicalRAGService(enable_cache=True)
    return _instance
