"""
Utils: Helper functions and utilities
"""
import logging
from datetime import datetime
from typing import Optional, Union
import secrets
import hashlib

logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string into datetime object with multiple format support.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        logger.warning(f"Invalid date_str provided: {date_str}")
        return None
    
    # Strip whitespace
    date_str = date_str.strip()
    
    formats = [
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
        "%Y/%m/%d"
    ]
    
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            logger.debug(f"Successfully parsed '{date_str}' using format '{fmt}'")
            return parsed_date
        except ValueError:
            continue
    
    logger.error(f"Failed to parse date string: '{date_str}'")
    return None


def generate_confirmation_code() -> str:
    """
    Generate unique appointment confirmation code.
    
    Returns:
        Confirmation code in format APT##### (e.g., APT12345)
    """
    try:
        # Use timestamp and random component for uniqueness
        timestamp = int(datetime.now().timestamp() * 1000)
        random_component = secrets.randbelow(100000)
        
        # Combine and hash for additional uniqueness
        combined = f"{timestamp}{random_component}"
        hash_value = int(hashlib.sha256(combined.encode()).hexdigest(), 16)
        
        # Generate 5-digit code
        code_number = (hash_value % 90000) + 10000  # Ensures 5 digits
        
        confirmation_code = f"APT{code_number:05d}"
        logger.debug(f"Generated confirmation code: {confirmation_code}")
        
        return confirmation_code
        
    except Exception as e:
        logger.error(f"Error generating confirmation code: {e}")
        # Fallback to simple timestamp-based code
        timestamp = int(datetime.now().timestamp())
        return f"APT{timestamp % 100000:05d}"


def format_datetime_friendly(dt: Union[datetime, str]) -> str:
    """
    Format datetime in friendly, human-readable format.
    
    Args:
        dt: datetime object or string to format
        
    Returns:
        Formatted string (e.g., "December 25, 2024 at 02:30 PM")
    """
    if not dt:
        return "Unknown date"
    
    try:
        if isinstance(dt, str):
            dt = parse_date(dt)
            if not dt:
                return "Invalid date"
        
        return dt.strftime("%B %d, %Y at %I:%M %p")
        
    except Exception as e:
        logger.error(f"Error formatting datetime: {e}")
        return "Unknown date"


def format_currency(amount: Union[float, int, str]) -> str:
    """
    Format currency amount with proper formatting.
    
    Args:
        amount: Amount to format (float, int, or string)
        
    Returns:
        Formatted currency string (e.g., "$150.00")
    """
    try:
        if isinstance(amount, str):
            # Remove currency symbols and commas
            amount = amount.replace('$', '').replace(',', '').strip()
            amount = float(amount)
        
        return f"${float(amount):.2f}"
        
    except (ValueError, TypeError) as e:
        logger.error(f"Error formatting currency: {e}")
        return "$0.00"


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    import re
    # Simple email validation pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_phone(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not phone or not isinstance(phone, str):
        return False
    
    import re
    # Remove common formatting characters
    cleaned = re.sub(r'[\s\-\(\)\.]+', '', phone)
    # Check if it's a valid phone number 
    return bool(re.match(r'^\+?[0-9]{10,15}$', cleaned))


def sanitize_string(text: str, max_length: int = 500) -> str:
    """
    Sanitize string input by removing dangerous characters and limiting length.
    
    Args:
        text: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")
    
    return text


def calculate_age(date_of_birth: str) -> Optional[int]:
    """
    Calculate age from date of birth.
    
    Args:
        date_of_birth: Date of birth string (YYYY-MM-DD)
        
    Returns:
        Age in years or None if invalid
    """
    try:
        dob = parse_date(date_of_birth)
        if not dob:
            return None
        
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        
        if age < 0 or age > 150:
            logger.warning(f"Calculated age {age} seems invalid")
            return None
        
        return age
        
    except Exception as e:
        logger.error(f"Error calculating age: {e}")
        return None


def format_phone_display(phone: str) -> str:
    """
    Format phone number for display.
    
    Args:
        phone: Raw phone number
        
    Returns:
        Formatted phone number (e.g., "(555) 123-4567")
    """
    if not phone or not isinstance(phone, str):
        return ""
    
    try:
        import re
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        if len(digits) == 10:
            # US format: (555) 123-4567
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            # US format with country code: +1 (555) 123-4567
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            # Return as-is if format unknown
            return phone
            
    except Exception as e:
        logger.error(f"Error formatting phone: {e}")
        return phone


def is_business_hours(dt: datetime, start_hour: int = 8, end_hour: int = 18) -> bool:
    """
    Check if datetime falls within business hours.
    
    Args:
        dt: datetime to check
        start_hour: Business start hour (default: 8 AM)
        end_hour: Business end hour (default: 6 PM)
        
    Returns:
        True if within business hours, False otherwise
    """
    try:
        if not dt:
            return False
        
        # Check if weekend
        if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within business hours
        return start_hour <= dt.hour < end_hour
        
    except Exception as e:
        logger.error(f"Error checking business hours: {e}")
        return False


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix