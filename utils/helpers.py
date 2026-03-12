"""
utils/helpers.py
Shared helper functions: date parsing, validation, formatting.
"""
import logging
import re
import secrets
import string
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_DATE_FORMATS = [
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%B %d, %Y %H:%M",
    "%B %d, %Y",
    "%b %d, %Y %H:%M",
    "%b %d, %Y",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
]


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse a date string using multiple format patterns."""
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    logger.warning("Could not parse date: %s", date_str)
    return None


def generate_confirmation_code(length: int = 8) -> str:
    """Generate a random alphanumeric confirmation code."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def format_datetime_friendly(dt: datetime | str) -> str:
    """Return a human-readable datetime string."""
    if isinstance(dt, str):
        dt = parse_date(dt)
    if not dt:
        return "Unknown"
    return dt.strftime("%A, %B %d, %Y at %I:%M %p")


def validate_email(email: str) -> bool:
    """Basic email format validation."""
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


def validate_phone(phone: str) -> bool:
    """Basic phone number validation (digits only, 10-15 chars)."""
    digits = re.sub(r"\D", "", phone)
    return 10 <= len(digits) <= 15


def sanitize_string(value: str, max_length: int = 500) -> str:
    """Strip whitespace and truncate to max_length."""
    if not value:
        return ""
    return value.strip()[:max_length]
