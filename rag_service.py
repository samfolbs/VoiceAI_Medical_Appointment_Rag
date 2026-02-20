"""
RAG Service 
Medical knowledge retrieval using ChromaDB and OpenAI embeddings
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI
import asyncio

from config import OPENAI_API_KEY, RAG_CONFIG
from cache_manager import MultiLevelCache

logger = logging.getLogger(__name__)

# Global RAG service instance
_rag_service_instance: Optional['MedicalRAGService'] = None


# Sample medical knowledge base
SAMPLE_MEDICAL_KNOWLEDGE = [
    {
        "id": "cardio_001",
        "specialty": "Cardiology",
        "content": "Chest pain, especially when accompanied by shortness of breath, sweating, or pain radiating to the arm or jaw, requires immediate emergency evaluation. Call 911 immediately.",
        "urgency": "EMERGENCY",
        "keywords": ["chest pain", "heart attack", "cardiac", "breathing difficulty"]
    },
    {
        "id": "cardio_002",
        "specialty": "Cardiology",
        "content": "For routine cardiac care, annual checkups, or follow-up for controlled heart conditions, schedule with a cardiologist during regular business hours.",
        "urgency": "ROUTINE",
        "keywords": ["heart checkup", "blood pressure", "cholesterol", "cardiac screening"]
    },
    {
        "id": "ortho_001",
        "specialty": "Orthopedics",
        "content": "Back pain after lifting heavy objects typically indicates muscle strain. If pain persists beyond 72 hours, worsens, or involves numbness/tingling in legs, see an orthopedist or spine specialist.",
        "urgency": "STANDARD",
        "keywords": ["back pain", "lifting injury", "muscle strain", "spine"]
    },
    {
        "id": "ortho_002",
        "specialty": "Orthopedics",
        "content": "Joint pain, swelling, or limited range of motion should be evaluated by an orthopedic specialist, especially if it interferes with daily activities.",
        "urgency": "STANDARD",
        "keywords": ["joint pain", "knee pain", "shoulder pain", "arthritis", "swelling"]
    },
    {
        "id": "peds_001",
        "specialty": "Pediatrics",
        "content": "High fever (>103°F/39.4°C) in children under 3 months requires immediate medical attention. For older children, persistent fever beyond 3 days warrants a pediatric visit.",
        "urgency": "URGENT",
        "keywords": ["child fever", "infant", "pediatric", "high temperature"]
    },
    {
        "id": "peds_002",
        "specialty": "Pediatrics",
        "content": "Routine well-child visits, vaccinations, and developmental screenings should be scheduled with a pediatrician according to the recommended schedule.",
        "urgency": "ROUTINE",
        "keywords": ["well-child", "vaccination", "checkup", "development", "immunization"]
    },
    {
        "id": "neuro_001",
        "specialty": "Neurology",
        "content": "Sudden severe headache, worst headache of life, headache with vision changes, confusion, or weakness requires immediate emergency evaluation.",
        "urgency": "EMERGENCY",
        "keywords": ["severe headache", "migraine", "vision problems", "confusion", "stroke"]
    },
    {
        "id": "neuro_002",
        "specialty": "Neurology",
        "content": "Chronic headaches, migraines, or neurological symptoms like numbness or tingling should be evaluated by a neurologist.",
        "urgency": "STANDARD",
        "keywords": ["chronic headache", "migraine", "numbness", "tingling", "neurological"]
    },
    {
        "id": "derm_001",
        "specialty": "Dermatology",
        "content": "Skin changes, new or changing moles, persistent rashes, or skin lesions should be evaluated by a dermatologist. Rapidly changing moles warrant urgent evaluation.",
        "urgency": "STANDARD",
        "keywords": ["mole", "skin", "rash", "lesion", "dermatology", "melanoma"]
    },
    {
        "id": "gp_001",
        "specialty": "General Practice",
        "content": "General medical concerns, annual physicals, medication management, and minor illnesses can be addressed by a general practitioner.",
        "urgency": "ROUTINE",
        "keywords": ["checkup", "physical", "general health", "primary care", "medication"]
    },
    {
        "id": "emergency_001",
        "specialty": "Emergency",
        "content": "CALL 911 IMMEDIATELY for: chest pain, difficulty breathing, loss of consciousness, severe bleeding, stroke symptoms (face drooping, arm weakness, speech difficulty), or suspected heart attack.",
        "urgency": "EMERGENCY",
        "keywords": ["911", "emergency", "life-threatening", "ambulance", "ER"]
    },
    {
        "id": "emergency_002",
        "specialty": "Emergency",
        "content": "Go to emergency room for: severe injuries, broken bones, deep cuts requiring stitches, high fever with stiff neck, severe abdominal pain, or poisoning.",
        "urgency": "EMERGENCY",
        "keywords": ["emergency room", "severe injury", "broken bone", "poisoning", "severe pain"]
    }
]


class MedicalRAGService:
    """RAG service for medical knowledge retrieval"""
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize RAG service
        
        Args:
            enable_cache: Whether to enable embedding caching
        """
        try:
            self.client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            
            self.collection_name = RAG_CONFIG["collection_name"]
            self.embedding_model = RAG_CONFIG["embedding_model"]
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Medical knowledge base"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize OpenAI client
            self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            
            # Initialize cache
            self.enable_cache = enable_cache
            if enable_cache:
                self.cache = MultiLevelCache(
                    l1_size=RAG_CONFIG.get("l1_size", 50),
                    l2_size=RAG_CONFIG.get("l2_size", 500),
                    ttl_seconds=RAG_CONFIG.get("ttl_seconds", 3600)
                )
                logger.info("Embedding cache enabled")
            else:
                self.cache = None
            
            logger.info("MedicalRAGService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
            raise
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text with caching
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Check cache first
            if self.enable_cache and self.cache:
                cached_embedding = self.cache.get(text)
                if cached_embedding is not None:
                    logger.debug("Cache hit for embedding")
                    return cached_embedding
            
            # Get embedding from OpenAI
            response = await self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            
            embedding = response.data[0].embedding
            
            # Cache embedding
            if self.enable_cache and self.cache:
                self.cache.set(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}", exc_info=True)
            raise
    
    async def add_medical_knowledge(
        self,
        knowledge_items: List[Dict[str, Any]]
    ) -> bool:
        """
        Add medical knowledge to the collection
        
        Args:
            knowledge_items: List of knowledge items with id, content, metadata
            
        Returns:
            True if successful
        """
        try:
            if not knowledge_items:
                logger.warning("No knowledge items to add")
                return False
            
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for item in knowledge_items:
                ids.append(item["id"])
                documents.append(item["content"])
                
                # Prepare metadata
                metadata = {
                    "specialty": item.get("specialty", "General"),
                    "urgency": item.get("urgency", "STANDARD"),
                    "keywords": ",".join(item.get("keywords", []))
                }
                metadatas.append(metadata)
                
                # Get embedding
                embedding = await self._get_embedding(item["content"])
                embeddings.append(embedding)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Added {len(knowledge_items)} items to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding medical knowledge: {e}", exc_info=True)
            return False
    
    async def search_medical_knowledge(
        self,
        query: str,
        n_results: int = 3,
        specialty_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search medical knowledge base
        
        Args:
            query: Search query
            n_results: Number of results to return
            specialty_filter: Optional specialty filter
            
        Returns:
            List of matching knowledge items
        """
        try:
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = None
            if specialty_filter:
                where_clause = {"specialty": specialty_filter}
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching medical knowledge: {e}", exc_info=True)
            return []
    
    async def check_emergency(self, symptoms: str) -> Dict[str, Any]:
        """
        Check if symptoms indicate emergency
        
        Args:
            symptoms: Symptom description
            
        Returns:
            Dictionary with emergency status and recommendation
        """
        try:
            # Search for matching emergency conditions
            results = await self.search_medical_knowledge(
                query=symptoms,
                n_results=3
            )
            
            # Check for emergency markers
            is_emergency = False
            emergency_message = ""
            
            for result in results:
                urgency = result["metadata"].get("urgency", "STANDARD")
                if urgency == "EMERGENCY":
                    is_emergency = True
                    emergency_message = result["text"]
                    break
            
            if is_emergency:
                return {
                    "is_emergency": True,
                    "message": emergency_message,
                    "recommendation": "CALL 911 IMMEDIATELY or go to the nearest emergency room."
                }
            
            return {
                "is_emergency": False,
                "message": "No immediate emergency detected, but please consult with a healthcare provider."
            }
            
        except Exception as e:
            logger.error(f"Error checking emergency status: {e}", exc_info=True)
            return {
                "is_emergency": False,
                "error": "Unable to determine emergency status"
            }
    
    async def get_specialty_recommendation(
        self,
        symptoms: str,
        n_results: int = 3
    ) -> Dict[str, Any]:
        """
        Get specialty recommendation based on symptoms
        
        Args:
            symptoms: Symptom description
            n_results: Number of results to consider
            
        Returns:
            Dictionary with recommended specialties and urgency
        """
        try:
            # Search knowledge base
            results = await self.search_medical_knowledge(
                query=symptoms,
                n_results=n_results
            )
            
            if not results:
                return {
                    "specialties": ["General Practice"],
                    "urgency": "STANDARD",
                    "confidence": "low",
                    "message": "Consider starting with a general practitioner."
                }
            
            # Extract specialties and urgency
            specialties = set()
            max_urgency = "ROUTINE"
            urgency_levels = {"EMERGENCY": 4, "URGENT": 3, "STANDARD": 2, "ROUTINE": 1}
            
            for result in results:
                specialty = result["metadata"].get("specialty", "General Practice")
                specialties.add(specialty)
                
                urgency = result["metadata"].get("urgency", "STANDARD")
                if urgency_levels.get(urgency, 0) > urgency_levels.get(max_urgency, 0):
                    max_urgency = urgency
            
            # Check for emergency
            if max_urgency == "EMERGENCY":
                return {
                    "specialties": list(specialties),
                    "urgency": "EMERGENCY",
                    "confidence": "high",
                    "message": "EMERGENCY: Call 911 or go to emergency room immediately!",
                    "supporting_info": [r["text"] for r in results[:1]]
                }
            
            return {
                "specialties": list(specialties),
                "urgency": max_urgency,
                "confidence": "medium" if len(results) > 1 else "low",
                "message": f"Recommended specialist(s): {', '.join(specialties)}",
                "supporting_info": [r["text"] for r in results[:2]]
            }
            
        except Exception as e:
            logger.error(f"Error getting specialty recommendation: {e}", exc_info=True)
            return {
                "specialties": ["General Practice"],
                "urgency": "STANDARD",
                "confidence": "low",
                "error": str(e)
            }
    
    async def should_use_rag(self, message: str) -> bool:
        """
        Determine if RAG should be used for this message
        
        Args:
            message: User message
            
        Returns:
            True if RAG should be used
        """
        # Keywords indicating medical/symptom queries
        medical_keywords = [
            "pain", "hurt", "symptom", "sick", "fever", "cough", "ache",
            "specialist", "doctor", "medical", "condition", "diagnosis",
            "treatment", "medication", "injury", "illness"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in medical_keywords)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            count = self.collection.count()
            
            stats = {
                "collection_name": self.collection_name,
                "total_documents": count,
                "embedding_model": self.embedding_model,
                "cache_enabled": self.enable_cache
            }
            
            if self.enable_cache and self.cache:
                stats["cache_stats"] = self.cache.get_stats()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}", exc_info=True)
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from collection
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Medical knowledge base"}
            )
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}", exc_info=True)
            return False


def get_rag_service() -> MedicalRAGService:
    """
    Get or create global RAG service instance
    
    Returns:
        MedicalRAGService instance
    """
    global _rag_service_instance
    
    if _rag_service_instance is None:
        _rag_service_instance = MedicalRAGService(enable_cache=True)
    
    return _rag_service_instance