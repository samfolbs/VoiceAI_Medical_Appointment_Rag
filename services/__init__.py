"""
services/__init__.py
Exports service classes and the OpenAI function-calling schema.
"""
from .appointment_services import (
    AppointmentService,
    AppointmentTypeService,
    DoctorService,
)
from .function_handler import OPENAI_FUNCTIONS, handle_function_call

__all__ = [
    "DoctorService",
    "AppointmentTypeService",
    "AppointmentService",
    "OPENAI_FUNCTIONS",
    "handle_function_call",
]
