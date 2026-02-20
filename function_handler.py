"""
Function Handler: Handles function calls from AI agent 
"""
import logging
from typing import Dict, Any, Callable

from services import (
    DoctorService,
    AppointmentTypeService,
    AppointmentService
)

logger = logging.getLogger(__name__)

# Function mapping
FUNCTION_MAP: Dict[str, Callable] = {
    'get_doctors_list': DoctorService.get_doctors_list,
    'get_specialties': DoctorService.get_specialties,
    'get_appointment_types': AppointmentTypeService.get_appointment_types,
    'check_doctor_availability': AppointmentService.check_doctor_availability,
    'get_available_slots': AppointmentService.get_available_slots,
    'book_appointment': AppointmentService.book_appointment,
    'lookup_appointment': AppointmentService.lookup_appointment,
    'cancel_appointment': AppointmentService.cancel_appointment,
    'get_patient_history': AppointmentService.get_patient_history
}


def handle_function_call(function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle function calls from the voice agent 
    
    Args:
        function_name: Name of the function to call
        parameters: Dictionary of parameters to pass to the function
        
    Returns:
        Dictionary containing the function result or error information
    """
    try:
        # Validate function exists
        if function_name not in FUNCTION_MAP:
            logger.error(f"Unknown function requested: {function_name}")
            return {
                "error": f"Unknown function: {function_name}",
                "available_functions": list(FUNCTION_MAP.keys())
            }
        
        # Sanitize parameters
        if not isinstance(parameters, dict):
            logger.error(f"Invalid parameters type for {function_name}: {type(parameters)}")
            return {
                "error": "Invalid parameters format",
                "expected": "dictionary"
            }
        
        # Get function
        func = FUNCTION_MAP[function_name]
        
        logger.info(f"Executing function: {function_name} with parameters: {parameters}")
        
        # Execute function with parameter validation
        try:
            result = func(**parameters)
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                logger.warning(f"Function {function_name} returned non-dict result")
                result = {"result": result}
            
            logger.info(f"Function {function_name} executed successfully")
            return result
            
        except TypeError as e:
            # Parameter mismatch
            logger.error(f"Parameter error in {function_name}: {e}")
            return {
                "error": f"Invalid parameters for {function_name}",
                "details": str(e),
                "hint": "Check that all required parameters are provided with correct types"
            }
            
        except ValueError as e:
            # Value validation error
            logger.error(f"Value error in {function_name}: {e}")
            return {
                "error": f"Invalid value in {function_name}",
                "details": str(e)
            }
            
        except KeyError as e:
            # Missing required data
            logger.error(f"Key error in {function_name}: {e}")
            return {
                "error": f"Missing required data in {function_name}",
                "details": str(e)
            }
            
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error executing {function_name}: {e}", exc_info=True)
        return {
            "error": f"Error executing {function_name}",
            "details": str(e),
            "type": type(e).__name__
        }


# Function definitions for OpenAI with enhanced descriptions
OPENAI_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_doctors_list",
            "description": "Get list of available doctors, optionally filtered by medical specialty. Use this when patient asks about available doctors or specialists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": "Medical specialty to filter by (e.g., 'Cardiology', 'Pediatrics', 'General Practice'). Leave empty to get all doctors."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_specialties",
            "description": "Get list of all medical specialties available at the clinic. Use this when patient asks what types of doctors are available.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointment_types",
            "description": "Get list of available appointment types with durations and any applicable discounts. Use when patient asks about types of visits or pricing.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_doctor_availability",
            "description": "Check if a specific doctor is available at a given date and time. Use this before attempting to book an appointment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "description": "Doctor identifier (e.g., 'dr_smith', 'dr_johnson')",
                        "enum": ["dr_smith", "dr_johnson", "dr_williams", "dr_brown", "dr_davis", "dr_martinez"]
                    },
                    "appointment_date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (e.g., '2024-12-25')"
                    },
                    "appointment_time": {
                        "type": "string",
                        "description": "Time in HH:MM format using 24-hour clock (e.g., '14:30' for 2:30 PM)"
                    }
                },
                "required": ["doctor_id", "appointment_date", "appointment_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": "Get all available time slots for a doctor on a specific date. Use this to show patients their options for scheduling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "description": "Doctor identifier",
                        "enum": ["dr_smith", "dr_johnson", "dr_williams", "dr_brown", "dr_davis", "dr_martinez"]
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format"
                    }
                },
                "required": ["doctor_id", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a medical appointment. CRITICAL: Always collect ALL required information before calling this function: patient name, email, phone, date of birth, and insurance provider. Confirm all details with patient before booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full name of the patient"
                    },
                    "doctor_id": {
                        "type": "string",
                        "description": "Doctor to book with",
                        "enum": ["dr_smith", "dr_johnson", "dr_williams", "dr_brown", "dr_davis", "dr_martinez"]
                    },
                    "appointment_date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format"
                    },
                    "appointment_time": {
                        "type": "string",
                        "description": "Time in HH:MM format (24-hour)"
                    },
                    "appointment_type": {
                        "type": "string",
                        "description": "Type of appointment",
                        "enum": ["consultation", "follow_up", "annual_checkup", "urgent_care", "telehealth"],
                        "default": "consultation"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for visit (keep confidential, max 500 characters)"
                    },
                    "patient_email": {
                        "type": "string",
                        "description": "Patient email for confirmation (REQUIRED)"
                    },
                    "patient_phone": {
                        "type": "string",
                        "description": "Patient phone number (REQUIRED)"
                    },
                    "date_of_birth": {
                        "type": "string",
                        "description": "Patient date of birth in YYYY-MM-DD format (REQUIRED)"
                    },
                    "insurance_provider": {
                        "type": "string",
                        "description": "Insurance provider name (REQUIRED)",
                        "enum": ["Blue Cross Blue Shield", "UnitedHealthcare", "Aetna", "Cigna", "Medicare", "Medicaid", "Self-Pay", "Other"],
                        "default": "Self-Pay"
                    }
                },
                "required": ["patient_name", "doctor_id", "appointment_date", "appointment_time", "patient_email", "patient_phone", "date_of_birth", "insurance_provider"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_appointment",
            "description": "Look up an existing appointment by ID or confirmation code. Use when patient wants to check their appointment details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "integer",
                        "description": "Numeric appointment ID"
                    },
                    "confirmation_code": {
                        "type": "string",
                        "description": "Appointment confirmation code (e.g., 'APT12345')"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment. IMPORTANT: Only use after confirming with patient and explaining the 24-hour cancellation policy. Get explicit confirmation before canceling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "integer",
                        "description": "The appointment ID to cancel"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for cancellation"
                    }
                },
                "required": ["appointment_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_history",
            "description": "Retrieve appointment history for a patient. IMPORTANT: Only use with explicit patient consent. Verify patient identity first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Patient's full name"
                    },
                    "patient_email": {
                        "type": "string",
                        "description": "Patient's email address"
                    }
                }
            }
        }
    }
]