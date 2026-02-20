"""
Initialize RAG Knowledge Base: script to populate the medical knowledge base
"""
import asyncio
import logging

from rag_service import get_rag_service, SAMPLE_MEDICAL_KNOWLEDGE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_rag():
    """Initialize RAG service with medical knowledge"""
    try:
        logger.info("Initializing RAG service...")
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Check if already initialized
        stats = rag_service.get_collection_stats()
        if stats.get('total_documents', 0) > 0:
            logger.info(f"Knowledge base already has {stats['total_documents']} documents")
            response = input("Do you want to clear and reinitialize? (yes/no): ")
            if response.lower() == 'yes':
                logger.info("Clearing existing knowledge base...")
                rag_service.clear_collection()
            else:
                logger.info("Keeping existing knowledge base")
                return
        
        # Add sample medical knowledge
        logger.info(f"Adding {len(SAMPLE_MEDICAL_KNOWLEDGE)} medical knowledge items...")
        success = await rag_service.add_medical_knowledge(SAMPLE_MEDICAL_KNOWLEDGE)
        
        if success:
            logger.info(" RAG knowledge base initialized successfully!")
            
            # Show stats
            final_stats = rag_service.get_collection_stats()
            logger.info(f"Collection: {final_stats['collection_name']}")
            logger.info(f"Documents: {final_stats['total_documents']}")
            logger.info(f"Embedding model: {final_stats['embedding_model']}")
            
            # Test search
            logger.info("\nTesting search functionality...")
            test_queries = [
                "I have chest pain",
                "My child has a fever",
                "Back pain after lifting"
            ]
            
            for query in test_queries:
                logger.info(f"\nQuery: '{query}'")
                results = await rag_service.search_medical_knowledge(query, n_results=2)
                for i, result in enumerate(results, 1):
                    logger.info(f"  {i}. {result['text'][:100]}...")
                    logger.info(f"     Specialty: {result['metadata'].get('specialty')}")
                    logger.info(f"     Urgency: {result['metadata'].get('urgency')}")
        else:
            logger.error(" Failed to initialize RAG knowledge base")
            
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}", exc_info=True)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Medical Appointment RAG - Knowledge Base Initialization")
    print("="*60 + "\n")
    
    try:
        asyncio.run(initialize_rag())
    except KeyboardInterrupt:
        print("\n\nInitialization cancelled by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n Initialization failed: {e}")