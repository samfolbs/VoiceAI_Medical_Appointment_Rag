"""
rag/knowledge_base.py
Static sample medical knowledge for seeding the ChromaDB collection.
This could be swapped for a real data source.
"""
from typing import Any, Dict, List

SAMPLE_MEDICAL_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "id": "cardio_001",
        "specialty": "Cardiology",
        "content": (
            "Chest pain, especially when accompanied by shortness of breath, sweating, "
            "or pain radiating to the arm or jaw, requires immediate emergency evaluation. "
            "Call 911 immediately."
        ),
        "urgency": "EMERGENCY",
        "keywords": ["chest pain", "heart attack", "cardiac", "breathing difficulty"],
    },
    {
        "id": "cardio_002",
        "specialty": "Cardiology",
        "content": (
            "For routine cardiac care, annual checkups, or follow-up for controlled heart "
            "conditions, schedule with a cardiologist during regular business hours."
        ),
        "urgency": "ROUTINE",
        "keywords": ["heart checkup", "blood pressure", "cholesterol", "cardiac screening"],
    },
    {
        "id": "ortho_001",
        "specialty": "Orthopedics",
        "content": (
            "Back pain after lifting heavy objects typically indicates muscle strain. "
            "If pain persists beyond 72 hours, worsens, or involves numbness/tingling "
            "in legs, see an orthopedist or spine specialist."
        ),
        "urgency": "STANDARD",
        "keywords": ["back pain", "lifting injury", "muscle strain", "spine"],
    },
    {
        "id": "ortho_002",
        "specialty": "Orthopedics",
        "content": (
            "Joint pain, swelling, or limited range of motion should be evaluated by an "
            "orthopedic specialist, especially if it interferes with daily activities."
        ),
        "urgency": "STANDARD",
        "keywords": ["joint pain", "knee pain", "shoulder pain", "arthritis", "swelling"],
    },
    {
        "id": "peds_001",
        "specialty": "Pediatrics",
        "content": (
            "High fever (>103°F/39.4°C) in children under 3 months requires immediate "
            "medical attention. For older children, persistent fever beyond 3 days "
            "warrants a pediatric visit."
        ),
        "urgency": "URGENT",
        "keywords": ["child fever", "infant", "pediatric", "high temperature"],
    },
    {
        "id": "peds_002",
        "specialty": "Pediatrics",
        "content": (
            "Routine well-child visits, vaccinations, and developmental screenings should "
            "be scheduled with a pediatrician according to the recommended schedule."
        ),
        "urgency": "ROUTINE",
        "keywords": ["well-child", "vaccination", "checkup", "development", "immunization"],
    },
    {
        "id": "neuro_001",
        "specialty": "Neurology",
        "content": (
            "Sudden severe headache, worst headache of life, headache with vision changes, "
            "confusion, or weakness requires immediate emergency evaluation."
        ),
        "urgency": "EMERGENCY",
        "keywords": ["severe headache", "migraine", "vision problems", "confusion", "stroke"],
    },
    {
        "id": "neuro_002",
        "specialty": "Neurology",
        "content": (
            "Chronic headaches, migraines, or neurological symptoms like numbness or "
            "tingling should be evaluated by a neurologist."
        ),
        "urgency": "STANDARD",
        "keywords": ["chronic headache", "migraine", "numbness", "tingling", "neurological"],
    },
    {
        "id": "derm_001",
        "specialty": "Dermatology",
        "content": (
            "Skin changes, new or changing moles, persistent rashes, or skin lesions should "
            "be evaluated by a dermatologist. Rapidly changing moles warrant urgent evaluation."
        ),
        "urgency": "STANDARD",
        "keywords": ["mole", "skin", "rash", "lesion", "dermatology", "melanoma"],
    },
    {
        "id": "gp_001",
        "specialty": "General Practice",
        "content": (
            "General medical concerns, annual physicals, medication management, and minor "
            "illnesses can be addressed by a general practitioner."
        ),
        "urgency": "ROUTINE",
        "keywords": ["checkup", "physical", "general health", "primary care", "medication"],
    },
    {
        "id": "emergency_001",
        "specialty": "Emergency",
        "content": (
            "CALL 911 IMMEDIATELY for: chest pain, difficulty breathing, loss of consciousness, "
            "severe bleeding, stroke symptoms (face drooping, arm weakness, speech difficulty), "
            "or suspected heart attack."
        ),
        "urgency": "EMERGENCY",
        "keywords": ["911", "emergency", "life-threatening", "ambulance", "ER"],
    },
    {
        "id": "emergency_002",
        "specialty": "Emergency",
        "content": (
            "Go to emergency room for: severe injuries, broken bones, deep cuts requiring "
            "stitches, high fever with stiff neck, severe abdominal pain, or poisoning."
        ),
        "urgency": "EMERGENCY",
        "keywords": ["emergency room", "severe injury", "broken bone", "poisoning", "severe pain"],
    },
]
