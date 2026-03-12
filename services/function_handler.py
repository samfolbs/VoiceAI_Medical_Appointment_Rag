"""
services/function_handler.py
Dispatches OpenAI tool-call requests to the appropriate service methods.
"""
import logging
from typing import Any, Callable, Dict

from .appointment_services import (
    AppointmentService,
    AppointmentTypeService,
    DoctorService,
)

logger = logging.getLogger(__name__)

_FUNCTION_MAP: Dict[str, Callable] = {
    "get_doctors_list": DoctorService.get_doctors_list,
    "get_specialties": DoctorService.get_specialties,
    "get_appointment_types": AppointmentTypeService.get_appointment_types,
    "check_doctor_availability": AppointmentService.check_doctor_availability,
    "get_available_slots": AppointmentService.get_available_slots,
    "book_appointment": AppointmentService.book_appointment,
    "lookup_appointment": AppointmentService.lookup_appointment,
    "cancel_appointment": AppointmentService.cancel_appointment,
    "get_patient_history": AppointmentService.get_patient_history,
}


def handle_function_call(
    function_name: str, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a named function and return its result dict."""
    if function_name not in _FUNCTION_MAP:
        logger.error("Unknown function: %s", function_name)
        return {
            "error": f"Unknown function: {function_name}",
            "available_functions": list(_FUNCTION_MAP.keys()),
        }

    if not isinstance(parameters, dict):
        logger.error("Invalid parameters type for %s: %s", function_name, type(parameters))
        return {"error": "Parameters must be a dictionary"}

    try:
        result = _FUNCTION_MAP[function_name](**parameters)
        if not isinstance(result, dict):
            result = {"result": result}
        logger.info("Function %s executed successfully", function_name)
        return result

    except TypeError as exc:
        logger.error("Parameter error in %s: %s", function_name, exc)
        return {
            "error": f"Invalid parameters for {function_name}",
            "details": str(exc),
            "hint": "Check all required parameters are present with correct types",
        }
    except (ValueError, KeyError) as exc:
        logger.error("Data error in %s: %s", function_name, exc)
        return {"error": f"Data error in {function_name}", "details": str(exc)}
    except Exception as exc:
        logger.error("Unexpected error in %s: %s", function_name, exc, exc_info=True)
        return {
            "error": f"Error executing {function_name}",
            "details": str(exc),
            "type": type(exc).__name__,
        }


# ---------------------------------------------------------------------------
# OpenAI tool schema
# ---------------------------------------------------------------------------
OPENAI_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_doctors_list",
            "description": "Get available doctors, optionally filtered by specialty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": "Medical specialty (e.g. 'Cardiology'). Omit for all doctors.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_specialties",
            "description": "List all medical specialties available at the clinic.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointment_types",
            "description": "List appointment types with durations and pricing.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_doctor_availability",
            "description": "Check if a doctor is available at a specific date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "enum": ["dr_smith", "dr_johnson", "dr_williams", "dr_brown", "dr_davis", "dr_martinez"],
                    },
                    "appointment_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "appointment_time": {"type": "string", "description": "HH:MM (24h)"},
                },
                "required": ["doctor_id", "appointment_date", "appointment_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": "Get all open time slots for a doctor on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "enum": ["dr_smith", "dr_johnson", "dr_williams", "dr_brown", "dr_davis", "dr_martinez"],
                    },
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["doctor_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book a medical appointment. Collect ALL required fields before calling: "
                "name, email, phone, DOB, insurance. Confirm with patient first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "doctor_id": {
                        "type": "string",
                        "enum": ["dr_smith", "dr_johnson", "dr_williams", "dr_brown", "dr_davis", "dr_martinez"],
                    },
                    "appointment_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "appointment_time": {"type": "string", "description": "HH:MM (24h)"},
                    "appointment_type": {
                        "type": "string",
                        "enum": ["consultation", "follow_up", "annual_checkup", "urgent_care", "telehealth"],
                        "default": "consultation",
                    },
                    "reason": {"type": "string"},
                    "patient_email": {"type": "string"},
                    "patient_phone": {"type": "string"},
                    "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
                    "insurance_provider": {
                        "type": "string",
                        "enum": [
                            "Blue Cross Blue Shield", "UnitedHealthcare", "Aetna",
                            "Cigna", "Medicare", "Medicaid", "Self-Pay", "Other",
                        ],
                        "default": "Self-Pay",
                    },
                },
                "required": [
                    "patient_name", "doctor_id", "appointment_date", "appointment_time",
                    "patient_email", "patient_phone", "date_of_birth", "insurance_provider",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_appointment",
            "description": "Look up an appointment by ID or confirmation code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "integer"},
                    "confirmation_code": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an appointment. Explain 24-hour policy and get explicit confirmation first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_history",
            "description": "Retrieve appointment history for a patient. Verify identity first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "patient_email": {"type": "string"},
                },
            },
        },
    },
]
