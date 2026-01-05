"""
Date utility functions for document extraction.

Provides date parsing, formatting, and validation functions
for handling various date formats in medical documents.
"""

import re
from datetime import UTC, date, datetime


class DateParseError(Exception):
    """Exception raised when date parsing fails."""


# Common date format patterns with their strptime formats
DATE_FORMATS: list[tuple[str, str]] = [
    # US formats
    (r"^\d{1,2}/\d{1,2}/\d{4}$", "%m/%d/%Y"),  # MM/DD/YYYY
    (r"^\d{1,2}-\d{1,2}-\d{4}$", "%m-%d-%Y"),  # MM-DD-YYYY
    (r"^\d{1,2}/\d{1,2}/\d{2}$", "%m/%d/%y"),  # MM/DD/YY
    (r"^\d{1,2}-\d{1,2}-\d{2}$", "%m-%d-%y"),  # MM-DD-YY
    # ISO format
    (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),  # YYYY-MM-DD
    (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d"),  # YYYY/MM/DD
    # Text formats
    (r"^\w+ \d{1,2}, \d{4}$", "%B %d, %Y"),  # January 1, 2024
    (r"^\w+ \d{1,2} \d{4}$", "%B %d %Y"),  # January 1 2024
    (r"^\d{1,2} \w+ \d{4}$", "%d %B %Y"),  # 1 January 2024
    (r"^\w{3} \d{1,2}, \d{4}$", "%b %d, %Y"),  # Jan 1, 2024
    (r"^\w{3} \d{1,2} \d{4}$", "%b %d %Y"),  # Jan 1 2024
    # Compact formats
    (r"^\d{8}$", "%m%d%Y"),  # MMDDYYYY
    (r"^\d{6}$", "%m%d%y"),  # MMDDYY
]


def parse_date(
    date_string: str,
    formats: list[tuple[str, str]] | None = None,
    default: date | None = None,
) -> date | None:
    """
    Parse a date string into a date object.

    Attempts multiple common date formats used in medical documents.

    Args:
        date_string: String representation of date.
        formats: Optional list of (pattern, strptime_format) tuples.
        default: Default value if parsing fails.

    Returns:
        Parsed date object or default value.

    Example:
        parse_date("01/15/2024") -> date(2024, 1, 15)
        parse_date("2024-01-15") -> date(2024, 1, 15)
        parse_date("January 15, 2024") -> date(2024, 1, 15)
    """
    if not date_string:
        return default

    date_string = date_string.strip()

    if formats is None:
        formats = DATE_FORMATS

    for pattern, date_format in formats:
        if re.match(pattern, date_string, re.IGNORECASE):
            try:
                return datetime.strptime(date_string, date_format).date()
            except ValueError:
                continue

    # Try fallback patterns
    fallback_patterns = [
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ]

    for date_format in fallback_patterns:
        try:
            return datetime.strptime(date_string, date_format).date()
        except ValueError:
            continue

    return default


def format_date(
    d: date | datetime,
    output_format: str = "%m/%d/%Y",
) -> str:
    """
    Format a date object to string.

    Args:
        d: Date or datetime object to format.
        output_format: strftime format string.

    Returns:
        Formatted date string.

    Example:
        format_date(date(2024, 1, 15)) -> "01/15/2024"
        format_date(date(2024, 1, 15), "%Y-%m-%d") -> "2024-01-15"
    """
    if isinstance(d, datetime):
        d = d.date()

    return d.strftime(output_format)


def parse_date_range(
    date_range_string: str,
    separator: str = "-",
) -> tuple[date | None, date | None]:
    """
    Parse a date range string into start and end dates.

    Args:
        date_range_string: String like "01/01/2024 - 01/31/2024".
        separator: Character separating start and end dates.

    Returns:
        Tuple of (start_date, end_date).

    Example:
        parse_date_range("01/01/2024 - 01/31/2024")
        -> (date(2024, 1, 1), date(2024, 1, 31))
    """
    if not date_range_string:
        return (None, None)

    # Handle various separators
    separators = [" - ", "-", " to ", " through "]

    parts = None
    for sep in separators:
        if sep in date_range_string:
            parts = date_range_string.split(sep, 1)
            break

    if not parts or len(parts) != 2:
        # Try single date
        single_date = parse_date(date_range_string)
        return (single_date, single_date)

    start_date = parse_date(parts[0].strip())
    end_date = parse_date(parts[1].strip())

    return (start_date, end_date)


def is_valid_date(
    date_string: str,
    min_date: date | None = None,
    max_date: date | None = None,
) -> bool:
    """
    Check if a date string represents a valid date.

    Args:
        date_string: String to validate.
        min_date: Optional minimum valid date.
        max_date: Optional maximum valid date.

    Returns:
        True if valid date within range.

    Example:
        is_valid_date("02/30/2024") -> False
        is_valid_date("01/15/2024") -> True
        is_valid_date("01/15/2024", min_date=date(2024, 2, 1)) -> False
    """
    parsed = parse_date(date_string)

    if parsed is None:
        return False

    if min_date and parsed < min_date:
        return False

    if max_date and parsed > max_date:
        return False

    return True


def get_current_timestamp() -> datetime:
    """
    Get current UTC timestamp.

    Returns:
        Current datetime with UTC timezone.
    """
    return datetime.now(UTC)


def get_current_date() -> date:
    """
    Get current date in UTC.

    Returns:
        Current date in UTC timezone.
    """
    return datetime.now(UTC).date()


def calculate_age(
    birth_date: date | str,
    as_of_date: date | None = None,
) -> int | None:
    """
    Calculate age in years from birth date.

    Args:
        birth_date: Date of birth.
        as_of_date: Date to calculate age as of. Defaults to today.

    Returns:
        Age in years, or None if birth_date is invalid.

    Example:
        calculate_age("01/15/1990") -> 34 (as of 2024)
        calculate_age(date(1990, 1, 15)) -> 34
    """
    if isinstance(birth_date, str):
        birth_date_parsed = parse_date(birth_date)
        if birth_date_parsed is None:
            return None
        birth_date = birth_date_parsed

    if as_of_date is None:
        as_of_date = get_current_date()

    age = as_of_date.year - birth_date.year

    # Adjust if birthday hasn't occurred yet this year
    if (as_of_date.month, as_of_date.day) < (birth_date.month, birth_date.day):
        age -= 1

    return max(0, age)


def normalize_date(
    date_string: str,
    output_format: str = "%m/%d/%Y",
) -> str | None:
    """
    Normalize a date string to a consistent format.

    Args:
        date_string: Date in any supported format.
        output_format: Desired output format.

    Returns:
        Normalized date string, or None if parsing fails.

    Example:
        normalize_date("2024-01-15") -> "01/15/2024"
        normalize_date("January 15, 2024") -> "01/15/2024"
    """
    parsed = parse_date(date_string)

    if parsed is None:
        return None

    return format_date(parsed, output_format)


def date_difference_days(
    date1: date | str,
    date2: date | str,
) -> int | None:
    """
    Calculate the difference between two dates in days.

    Args:
        date1: First date.
        date2: Second date.

    Returns:
        Number of days between dates (positive if date2 > date1).
    """
    if isinstance(date1, str):
        date1_parsed = parse_date(date1)
        if date1_parsed is None:
            return None
        date1 = date1_parsed

    if isinstance(date2, str):
        date2_parsed = parse_date(date2)
        if date2_parsed is None:
            return None
        date2 = date2_parsed

    return (date2 - date1).days


def is_future_date(date_string: str) -> bool:
    """
    Check if a date is in the future.

    Args:
        date_string: Date to check.

    Returns:
        True if date is after today.
    """
    parsed = parse_date(date_string)

    if parsed is None:
        return False

    return parsed > get_current_date()


def is_past_date(date_string: str) -> bool:
    """
    Check if a date is in the past.

    Args:
        date_string: Date to check.

    Returns:
        True if date is before today.
    """
    parsed = parse_date(date_string)

    if parsed is None:
        return False

    return parsed < get_current_date()


def get_year_from_date(date_string: str) -> int | None:
    """
    Extract year from a date string.

    Args:
        date_string: Date string to parse.

    Returns:
        Year as integer, or None if parsing fails.
    """
    parsed = parse_date(date_string)

    if parsed is None:
        return None

    return parsed.year


def dates_in_order(
    dates: list[str | date],
    allow_equal: bool = True,
) -> bool:
    """
    Check if dates are in chronological order.

    Args:
        dates: List of dates to check.
        allow_equal: Whether equal consecutive dates are allowed.

    Returns:
        True if dates are in order.

    Example:
        dates_in_order(["01/01/2024", "01/15/2024", "02/01/2024"]) -> True
        dates_in_order(["01/15/2024", "01/01/2024"]) -> False
    """
    parsed_dates: list[date] = []

    for d in dates:
        if isinstance(d, str):
            parsed = parse_date(d)
            if parsed is None:
                return False
            parsed_dates.append(parsed)
        else:
            parsed_dates.append(d)

    for i in range(1, len(parsed_dates)):
        if allow_equal:
            if parsed_dates[i] < parsed_dates[i - 1]:
                return False
        elif parsed_dates[i] <= parsed_dates[i - 1]:
            return False

    return True
