"""
Services: Business logic for doctor, appointment type, and appointment management
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import random

from config import MOCK_DATABASE
from utils import (
    parse_date,
    generate_confirmation_code,
    format_datetime_friendly,
    validate_email,
    validate_phone,
    sanitize_string
)

logger = logging.getLogger(__name__)

# In-memory storage for appointments 
appointments_db: List[Dict[str, Any]] = []
appointment_id_counter = 1


class DoctorService:
    """Service for doctor-related operations"""
    
    @staticmethod
    def get_doctors_list(specialty: Optional[str] = None) -> Dict[str, Any]:
        """
        Get list of doctors, optionally filtered by specialty
        
        Args:
            specialty: Medical specialty to filter by
            
        Returns:
            Dictionary with doctors list
        """
        try:
            doctors = MOCK_DATABASE["doctors"]
            
            if specialty:
                specialty_lower = specialty.lower()
                doctors = [
                    d for d in doctors 
                    if d["specialty"].lower() == specialty_lower
                ]
                
                if not doctors:
                    logger.info(f"No doctors found for specialty: {specialty}")
                    return {
                        "success": False,
                        "message": f"No doctors found for specialty: {specialty}",
                        "available_specialties": DoctorService.get_specialties()["specialties"]
                    }
            
            logger.info(f"Retrieved {len(doctors)} doctors")
            return {
                "success": True,
                "doctors": doctors,
                "count": len(doctors)
            }
            
        except Exception as e:
            logger.error(f"Error getting doctors list: {e}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to retrieve doctors list"
            }
    
    @staticmethod
    def get_specialties() -> Dict[str, Any]:
        """
        Get list of all available specialties
        
        Returns:
            Dictionary with specialties list
        """
        try:
            specialties = list(set(d["specialty"] for d in MOCK_DATABASE["doctors"]))
            specialties.sort()
            
            logger.info(f"Retrieved {len(specialties)} specialties")
            return {
                "success": True,
                "specialties": specialties,
                "count": len(specialties)
            }
            
        except Exception as e:
            logger.error(f"Error getting specialties: {e}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to retrieve specialties"
            }


class AppointmentTypeService:
    """Service for appointment type operations"""
    
    @staticmethod
    def get_appointment_types() -> Dict[str, Any]:
        """
        Get all available appointment types
        
        Returns:
            Dictionary with appointment types
        """
        try:
            types = MOCK_DATABASE["appointment_types"]
            
            logger.info(f"Retrieved {len(types)} appointment types")
            return {
                "success": True,
                "appointment_types": types,
                "count": len(types)
            }
            
        except Exception as e:
            logger.error(f"Error getting appointment types: {e}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to retrieve appointment types"
            }


class AppointmentService:
    """Service for appointment management operations"""
    
    @staticmethod
    def _find_doctor(doctor_id: str) -> Optional[Dict[str, Any]]:
        """Find doctor by ID"""
        return next(
            (d for d in MOCK_DATABASE["doctors"] if d["id"] == doctor_id),
            None
        )
    
    @staticmethod
    def check_doctor_availability(
        doctor_id: str,
        appointment_date: str,
        appointment_time: str
    ) -> Dict[str, Any]:
        """
        Check if doctor is available at specified date and time
        
        Args:
            doctor_id: Doctor identifier
            appointment_date: Date in YYYY-MM-DD format
            appointment_time: Time in HH:MM format
            
        Returns:
            Dictionary with availability status
        """
        try:
            # Find doctor
            doctor = AppointmentService._find_doctor(doctor_id)
            if not doctor:
                return {
                    "success": False,
                    "available": False,
                    "error": f"Doctor not found: {doctor_id}"
                }
            
            # Parse date
            dt = parse_date(f"{appointment_date} {appointment_time}")
            if not dt:
                return {
                    "success": False,
                    "available": False,
                    "error": "Invalid date or time format"
                }
            
            # Check if date is in the past
            if dt < datetime.now():
                return {
                    "success": True,
                    "available": False,
                    "message": "Cannot book appointments in the past"
                }
            
            # Check day of week
            day_name = dt.strftime("%A")
            if day_name not in doctor["available_days"]:
                return {
                    "success": True,
                    "available": False,
                    "message": f"Dr. {doctor['name']} is not available on {day_name}",
                    "available_days": doctor["available_days"]
                }
            
            # Check time
            time_str = dt.strftime("%H:%M")
            if time_str not in doctor["available_hours"]:
                return {
                    "success": True,
                    "available": False,
                    "message": f"Time {appointment_time} is not available",
                    "available_hours": doctor["available_hours"]
                }
            
            # Check for conflicts with existing appointments
            for appt in appointments_db:
                if (appt["doctor_id"] == doctor_id and 
                    appt["appointment_date"] == appointment_date and 
                    appt["appointment_time"] == appointment_time and
                    appt["status"] != "cancelled"):
                    return {
                        "success": True,
                        "available": False,
                        "message": "This time slot is already booked"
                    }
            
            return {
                "success": True,
                "available": True,
                "doctor": doctor,
                "appointment_datetime": format_datetime_friendly(dt)
            }
            
        except Exception as e:
            logger.error(f"Error checking availability: {e}", exc_info=True)
            return {
                "success": False,
                "available": False,
                "error": "Failed to check availability"
            }
    
    @staticmethod
    def get_available_slots(doctor_id: str, date: str) -> Dict[str, Any]:
        """
        Get all available time slots for a doctor on a specific date
        
        Args:
            doctor_id: Doctor identifier
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with available slots
        """
        try:
            doctor = AppointmentService._find_doctor(doctor_id)
            if not doctor:
                return {
                    "success": False,
                    "error": f"Doctor not found: {doctor_id}"
                }
            
            # Parse date
            dt = parse_date(date)
            if not dt:
                return {
                    "success": False,
                    "error": "Invalid date format"
                }
            
            # Check if date is in the past
            if dt.date() < datetime.now().date():
                return {
                    "success": False,
                    "error": "Cannot check availability for past dates"
                }
            
            # Check day of week
            day_name = dt.strftime("%A")
            if day_name not in doctor["available_days"]:
                return {
                    "success": True,
                    "available_slots": [],
                    "message": f"Dr. {doctor['name']} is not available on {day_name}",
                    "available_days": doctor["available_days"]
                }
            
            # Get booked slots
            booked_slots = set(
                appt["appointment_time"]
                for appt in appointments_db
                if (appt["doctor_id"] == doctor_id and 
                    appt["appointment_date"] == date and
                    appt["status"] != "cancelled")
            )
            
            # Calculate available slots
            available_slots = [
                time_slot for time_slot in doctor["available_hours"]
                if time_slot not in booked_slots
            ]
            
            return {
                "success": True,
                "doctor": doctor,
                "date": date,
                "day_of_week": day_name,
                "available_slots": available_slots,
                "total_slots": len(available_slots)
            }
            
        except Exception as e:
            logger.error(f"Error getting available slots: {e}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to get available slots"
            }
    
    @staticmethod
    def book_appointment(
        patient_name: str,
        doctor_id: str,
        appointment_date: str,
        appointment_time: str,
        patient_email: str,
        patient_phone: str,
        date_of_birth: str,
        insurance_provider: str,
        appointment_type: str = "consultation",
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Book a medical appointment
        
        Args:
            patient_name: Patient's full name
            doctor_id: Doctor identifier
            appointment_date: Date in YYYY-MM-DD format
            appointment_time: Time in HH:MM format
            patient_email: Patient's email
            patient_phone: Patient's phone number
            date_of_birth: Patient's date of birth
            insurance_provider: Insurance provider name
            appointment_type: Type of appointment
            reason: Reason for visit
            
        Returns:
            Dictionary with booking confirmation
        """
        global appointment_id_counter
        
        try:
            # Validate required fields
            if not all([patient_name, doctor_id, appointment_date, appointment_time,
                       patient_email, patient_phone, date_of_birth, insurance_provider]):
                return {
                    "success": False,
                    "error": "Missing required fields"
                }
            
            # Sanitize inputs
            patient_name = sanitize_string(patient_name, 200)
            reason = sanitize_string(reason, 500) if reason else "Not specified"
            
            # Validate email
            if not validate_email(patient_email):
                return {
                    "success": False,
                    "error": "Invalid email address"
                }
            
            # Validate phone
            if not validate_phone(patient_phone):
                return {
                    "success": False,
                    "error": "Invalid phone number"
                }
            
            # Check availability
            availability = AppointmentService.check_doctor_availability(
                doctor_id, appointment_date, appointment_time
            )
            
            if not availability.get("available", False):
                return {
                    "success": False,
                    "error": availability.get("message", "Time slot not available"),
                    "available_slots": AppointmentService.get_available_slots(
                        doctor_id, appointment_date
                    ).get("available_slots", [])
                }
            
            # Generate confirmation code
            confirmation_code = generate_confirmation_code()
            
            # Create appointment
            appointment = {
                "id": appointment_id_counter,
                "confirmation_code": confirmation_code,
                "patient_name": patient_name,
                "patient_email": patient_email,
                "patient_phone": patient_phone,
                "date_of_birth": date_of_birth,
                "insurance_provider": insurance_provider,
                "doctor_id": doctor_id,
                "doctor_name": availability["doctor"]["name"],
                "specialty": availability["doctor"]["specialty"],
                "appointment_date": appointment_date,
                "appointment_time": appointment_time,
                "appointment_type": appointment_type,
                "reason": reason,
                "status": "confirmed",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            appointments_db.append(appointment)
            appointment_id_counter += 1
            
            logger.info(f"Appointment booked: ID={appointment['id']}, Code={confirmation_code}")
            
            return {
                "success": True,
                "message": "Appointment booked successfully",
                "appointment": appointment,
                "confirmation_code": confirmation_code,
                "appointment_datetime": format_datetime_friendly(
                    f"{appointment_date} {appointment_time}"
                )
            }
            
        except Exception as e:
            logger.error(f"Error booking appointment: {e}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to book appointment"
            }
    
    @staticmethod
    def lookup_appointment(
        appointment_id: Optional[int] = None,
        confirmation_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Look up an appointment by ID or confirmation code
        
        Args:
            appointment_id: Appointment ID
            confirmation_code: Confirmation code
            
        Returns:
            Dictionary with appointment details
        """
        try:
            if not appointment_id and not confirmation_code:
                return {
                    "success": False,
                    "error": "Must provide appointment_id or confirmation_code"
                }
            
            appointment = None
            
            if appointment_id:
                appointment = next(
                    (a for a in appointments_db if a["id"] == appointment_id),
                    None
                )
            elif confirmation_code:
                appointment = next(
                    (a for a in appointments_db if a["confirmation_code"] == confirmation_code),
                    None
                )
            
            if not appointment:
                return {
                    "success": False,
                    "error": "Appointment not found"
                }
            
            return {
                "success": True,
                "appointment": appointment,
                "appointment_datetime": format_datetime_friendly(
                    f"{appointment['appointment_date']} {appointment['appointment_time']}"
                )
            }
            
        except Exception as e:
            logger.error(f"Error looking up appointment: {e}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to lookup appointment"
            }
    
    @staticmethod
    def cancel_appointment(
        appointment_id: int,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an appointment
        
        Args:
            appointment_id: Appointment ID to cancel
            reason: Optional cancellation reason
            
        Returns:
            Dictionary with cancellation status
        """
        try:
            appointment = next(
                (a for a in appointments_db if a["id"] == appointment_id),
                None
            )
            
            if not appointment:
                return {
                    "success": False,
                    "error": "Appointment not found"
                }
            
            if appointment["status"] == "cancelled":
                return {
                    "success": False,
                    "error": "Appointment already cancelled"
                }
            
            # Check 24-hour cancellation policy
            appt_datetime = parse_date(
                f"{appointment['appointment_date']} {appointment['appointment_time']}"
            )
            
            if appt_datetime:
                hours_until = (appt_datetime - datetime.now()).total_seconds() / 3600
                if hours_until < 24:
                    return {
                        "success": False,
                        "error": "Cannot cancel within 24 hours of appointment",
                        "policy": "Please call our office for late cancellations"
                    }
            
            # Cancel appointment
            appointment["status"] = "cancelled"
            appointment["cancellation_reason"] = reason or "Not specified"
            appointment["cancelled_at"] = datetime.now().isoformat()
            appointment["updated_at"] = datetime.now().isoformat()
            
            logger.info(f"Appointment cancelled: ID={appointment_id}")
            
            return {
                "success": True,
                "message": "Appointment cancelled successfully",
                "appointment": appointment
            }
            
        except Exception as e:
            logger.error(f"Error cancelling appointment: {e}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to cancel appointment"
            }
    
    @staticmethod
    def get_patient_history(
        patient_name: Optional[str] = None,
        patient_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get appointment history for a patient
        
        Args:
            patient_name: Patient's name
            patient_email: Patient's email
            
        Returns:
            Dictionary with appointment history
        """
        try:
            if not patient_name and not patient_email:
                return {
                    "success": False,
                    "error": "Must provide patient_name or patient_email"
                }
            
            # Find matching appointments
            patient_appointments = []
            
            for appt in appointments_db:
                if patient_email and appt["patient_email"].lower() == patient_email.lower():
                    patient_appointments.append(appt)
                elif patient_name and appt["patient_name"].lower() == patient_name.lower():
                    patient_appointments.append(appt)
            
            if not patient_appointments:
                return {
                    "success": True,
                    "appointments": [],
                    "count": 0,
                    "message": "No appointments found for this patient"
                }
            
            # Sort by date (most recent first)
            patient_appointments.sort(
                key=lambda x: (x["appointment_date"], x["appointment_time"]),
                reverse=True
            )
            
            return {
                "success": True,
                "appointments": patient_appointments,
                "count": len(patient_appointments)
            }
            
        except Exception as e:
            logger.error(f"Error getting patient history: {e}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to retrieve patient history"
            }