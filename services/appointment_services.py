"""
services/appointment_services.py
Business logic for doctors, appointment types, and bookings.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.config import MOCK_DATABASE
from utils.helpers import (
    format_datetime_friendly,
    generate_confirmation_code,
    parse_date,
    sanitize_string,
    validate_email,
    validate_phone,
)

logger = logging.getLogger(__name__)

# In-memory appointment store 
_appointments: List[Dict[str, Any]] = []
_id_counter = 1


class DoctorService:
    """Read-only queries against the doctor catalogue."""

    @staticmethod
    def get_doctors_list(specialty: Optional[str] = None) -> Dict[str, Any]:
        try:
            doctors = MOCK_DATABASE["doctors"]
            if specialty:
                doctors = [d for d in doctors if d["specialty"].lower() == specialty.lower()]
                if not doctors:
                    return {
                        "success": False,
                        "message": f"No doctors found for specialty: {specialty}",
                        "available_specialties": DoctorService.get_specialties()["specialties"],
                    }
            return {"success": True, "doctors": doctors, "count": len(doctors)}
        except Exception as exc:
            logger.error("get_doctors_list error: %s", exc, exc_info=True)
            return {"success": False, "error": "Failed to retrieve doctors list"}

    @staticmethod
    def get_specialties() -> Dict[str, Any]:
        try:
            specialties = sorted({d["specialty"] for d in MOCK_DATABASE["doctors"]})
            return {"success": True, "specialties": specialties, "count": len(specialties)}
        except Exception as exc:
            logger.error("get_specialties error: %s", exc, exc_info=True)
            return {"success": False, "error": "Failed to retrieve specialties"}


class AppointmentTypeService:
    """Queries for appointment type catalogue."""

    @staticmethod
    def get_appointment_types() -> Dict[str, Any]:
        try:
            types = MOCK_DATABASE["appointment_types"]
            return {"success": True, "appointment_types": types, "count": len(types)}
        except Exception as exc:
            logger.error("get_appointment_types error: %s", exc, exc_info=True)
            return {"success": False, "error": "Failed to retrieve appointment types"}


class AppointmentService:
    """CRUD operations for appointments."""

    @staticmethod
    def _find_doctor(doctor_id: str) -> Optional[Dict[str, Any]]:
        return next((d for d in MOCK_DATABASE["doctors"] if d["id"] == doctor_id), None)

    @staticmethod
    def check_doctor_availability(
        doctor_id: str, appointment_date: str, appointment_time: str
    ) -> Dict[str, Any]:
        try:
            doctor = AppointmentService._find_doctor(doctor_id)
            if not doctor:
                return {"success": False, "available": False, "error": f"Doctor not found: {doctor_id}"}

            dt = parse_date(f"{appointment_date} {appointment_time}")
            if not dt:
                return {"success": False, "available": False, "error": "Invalid date/time format"}

            if dt < datetime.now():
                return {"success": True, "available": False, "message": "Cannot book appointments in the past"}

            day_name = dt.strftime("%A")
            if day_name not in doctor["available_days"]:
                return {
                    "success": True, "available": False,
                    "message": f"{doctor['name']} is not available on {day_name}",
                    "available_days": doctor["available_days"],
                }

            time_str = dt.strftime("%H:%M")
            if time_str not in doctor["available_hours"]:
                return {
                    "success": True, "available": False,
                    "message": f"Time {appointment_time} is not available",
                    "available_hours": doctor["available_hours"],
                }

            for appt in _appointments:
                if (
                    appt["doctor_id"] == doctor_id
                    and appt["appointment_date"] == appointment_date
                    and appt["appointment_time"] == appointment_time
                    and appt["status"] != "cancelled"
                ):
                    return {"success": True, "available": False, "message": "Time slot already booked"}

            return {
                "success": True,
                "available": True,
                "doctor": doctor,
                "appointment_datetime": format_datetime_friendly(dt),
            }

        except Exception as exc:
            logger.error("check_doctor_availability error: %s", exc, exc_info=True)
            return {"success": False, "available": False, "error": "Failed to check availability"}

    @staticmethod
    def get_available_slots(doctor_id: str, date: str) -> Dict[str, Any]:
        try:
            doctor = AppointmentService._find_doctor(doctor_id)
            if not doctor:
                return {"success": False, "error": f"Doctor not found: {doctor_id}"}

            dt = parse_date(date)
            if not dt:
                return {"success": False, "error": "Invalid date format"}

            if dt.date() < datetime.now().date():
                return {"success": False, "error": "Cannot check availability for past dates"}

            day_name = dt.strftime("%A")
            if day_name not in doctor["available_days"]:
                return {
                    "success": True, "available_slots": [],
                    "message": f"{doctor['name']} is not available on {day_name}",
                    "available_days": doctor["available_days"],
                }

            booked = {
                a["appointment_time"]
                for a in _appointments
                if a["doctor_id"] == doctor_id
                and a["appointment_date"] == date
                and a["status"] != "cancelled"
            }
            available = [t for t in doctor["available_hours"] if t not in booked]

            return {
                "success": True,
                "doctor": doctor,
                "date": date,
                "day_of_week": day_name,
                "available_slots": available,
                "total_slots": len(available),
            }

        except Exception as exc:
            logger.error("get_available_slots error: %s", exc, exc_info=True)
            return {"success": False, "error": "Failed to get available slots"}

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
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        global _id_counter

        required = [patient_name, doctor_id, appointment_date, appointment_time,
                    patient_email, patient_phone, date_of_birth, insurance_provider]
        if not all(required):
            return {"success": False, "error": "Missing required fields"}

        if not validate_email(patient_email):
            return {"success": False, "error": "Invalid email address"}
        if not validate_phone(patient_phone):
            return {"success": False, "error": "Invalid phone number"}

        availability = AppointmentService.check_doctor_availability(
            doctor_id, appointment_date, appointment_time
        )
        if not availability.get("available"):
            return {
                "success": False,
                "error": availability.get("message", "Time slot not available"),
                "available_slots": AppointmentService.get_available_slots(
                    doctor_id, appointment_date
                ).get("available_slots", []),
            }

        code = generate_confirmation_code()
        appt: Dict[str, Any] = {
            "id": _id_counter,
            "confirmation_code": code,
            "patient_name": sanitize_string(patient_name, 200),
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
            "reason": sanitize_string(reason, 500) if reason else "Not specified",
            "status": "confirmed",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        _appointments.append(appt)
        _id_counter += 1

        logger.info("Appointment booked: id=%d code=%s", appt["id"], code)
        return {
            "success": True,
            "message": "Appointment booked successfully",
            "appointment": appt,
            "confirmation_code": code,
            "appointment_datetime": format_datetime_friendly(
                f"{appointment_date} {appointment_time}"
            ),
        }

    @staticmethod
    def lookup_appointment(
        appointment_id: Optional[int] = None,
        confirmation_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not appointment_id and not confirmation_code:
            return {"success": False, "error": "Provide appointment_id or confirmation_code"}

        appt = None
        if appointment_id:
            appt = next((a for a in _appointments if a["id"] == appointment_id), None)
        elif confirmation_code:
            appt = next((a for a in _appointments if a["confirmation_code"] == confirmation_code), None)

        if not appt:
            return {"success": False, "error": "Appointment not found"}

        return {
            "success": True,
            "appointment": appt,
            "appointment_datetime": format_datetime_friendly(
                f"{appt['appointment_date']} {appt['appointment_time']}"
            ),
        }

    @staticmethod
    def cancel_appointment(
        appointment_id: int, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        appt = next((a for a in _appointments if a["id"] == appointment_id), None)
        if not appt:
            return {"success": False, "error": "Appointment not found"}
        if appt["status"] == "cancelled":
            return {"success": False, "error": "Appointment already cancelled"}

        dt = parse_date(f"{appt['appointment_date']} {appt['appointment_time']}")
        if dt and (dt - datetime.now()).total_seconds() < 86400:
            return {
                "success": False,
                "error": "Cannot cancel within 24 hours of appointment",
                "policy": "Please call our office for late cancellations",
            }

        appt["status"] = "cancelled"
        appt["cancellation_reason"] = reason or "Not specified"
        appt["cancelled_at"] = datetime.now().isoformat()
        appt["updated_at"] = datetime.now().isoformat()
        logger.info("Appointment cancelled: id=%d", appointment_id)
        return {"success": True, "message": "Appointment cancelled successfully", "appointment": appt}

    @staticmethod
    def get_patient_history(
        patient_name: Optional[str] = None,
        patient_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not patient_name and not patient_email:
            return {"success": False, "error": "Provide patient_name or patient_email"}

        matches = []
        for a in _appointments:
            if patient_email and a["patient_email"].lower() == patient_email.lower():
                matches.append(a)
            elif patient_name and a["patient_name"].lower() == patient_name.lower():
                matches.append(a)

        matches.sort(key=lambda x: (x["appointment_date"], x["appointment_time"]), reverse=True)
        return {"success": True, "appointments": matches, "count": len(matches)}
