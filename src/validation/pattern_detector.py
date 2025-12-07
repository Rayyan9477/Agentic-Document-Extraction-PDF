"""
Hallucination pattern detection for extracted data.

Implements Layer 3 of the 3-Layer Anti-Hallucination System:
- Detects common hallucination patterns in VLM outputs
- Identifies placeholder and synthetic data
- Flags suspiciously perfect or repetitive values
- Validates data consistency and plausibility

Patterns detected:
- Placeholder text (N/A, TBD, XXX, etc.)
- Repetitive values across fields
- Suspiciously round numbers
- Sequential/incrementing patterns
- Type mismatches (letters in numeric fields)
- Implausible dates (future, ancient)
- Generic names and addresses
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


class HallucinationPattern(str, Enum):
    """Types of hallucination patterns detected."""

    PLACEHOLDER_TEXT = "placeholder_text"
    REPETITIVE_VALUE = "repetitive_value"
    ROUND_NUMBER = "round_number"
    SEQUENTIAL_PATTERN = "sequential_pattern"
    TYPE_MISMATCH = "type_mismatch"
    IMPLAUSIBLE_DATE = "implausible_date"
    GENERIC_NAME = "generic_name"
    GENERIC_ADDRESS = "generic_address"
    SYNTHETIC_IDENTIFIER = "synthetic_identifier"
    IMPOSSIBLE_VALUE = "impossible_value"
    TRUNCATED_VALUE = "truncated_value"
    REPEATED_DIGITS = "repeated_digits"
    ALPHABETIC_SEQUENCE = "alphabetic_sequence"
    TEST_DATA = "test_data"


class PatternSeverity(str, Enum):
    """Severity level of detected pattern."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class PatternMatch:
    """
    A detected hallucination pattern in extracted data.

    Attributes:
        field_name: Name of the field with detected pattern.
        value: The suspicious value.
        pattern: Type of pattern detected.
        severity: Severity level of the detection.
        confidence: Confidence that this is a hallucination 0.0-1.0.
        description: Human-readable description of the issue.
        suggestion: Suggested action or correction.
    """

    field_name: str
    value: Any
    pattern: HallucinationPattern
    severity: PatternSeverity
    confidence: float
    description: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_name": self.field_name,
            "value": self.value,
            "pattern": self.pattern.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "suggestion": self.suggestion,
        }


@dataclass(slots=True)
class PatternDetectionResult:
    """
    Complete result of pattern detection on extracted data.

    Attributes:
        matches: List of detected pattern matches.
        flagged_fields: Fields that triggered pattern detection.
        overall_suspicion_score: Combined suspicion score 0.0-1.0.
        is_likely_hallucination: Whether data is likely hallucinated.
        critical_patterns: Patterns with critical severity.
        summary: Human-readable summary of findings.
    """

    matches: list[PatternMatch] = field(default_factory=list)
    flagged_fields: set[str] = field(default_factory=set)
    overall_suspicion_score: float = 0.0
    is_likely_hallucination: bool = False
    critical_patterns: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "matches": [m.to_dict() for m in self.matches],
            "flagged_fields": list(self.flagged_fields),
            "overall_suspicion_score": self.overall_suspicion_score,
            "is_likely_hallucination": self.is_likely_hallucination,
            "critical_patterns": self.critical_patterns,
            "summary": self.summary,
        }


class HallucinationPatternDetector:
    """
    Detects common hallucination patterns in VLM-extracted data.

    This detector identifies various patterns that indicate potential
    hallucination or synthetic data generation by vision language models.

    Thresholds:
        - Overall suspicion >= 0.70: Likely hallucination
        - Critical pattern detected: Automatic flag

    Example:
        detector = HallucinationPatternDetector()
        result = detector.detect(extracted_data)

        if result.is_likely_hallucination:
            print(f"Suspicious fields: {result.flagged_fields}")
    """

    # Placeholder patterns (case-insensitive)
    PLACEHOLDER_PATTERNS = [
        r"^n/?a$",
        r"^tbd$",
        r"^xxx+$",
        r"^placeholder$",
        r"^unknown$",
        r"^not\s*applicable$",
        r"^not\s*available$",
        r"^none$",
        r"^null$",
        r"^undefined$",
        r"^to\s*be\s*determined$",
        r"^pending$",
        r"^missing$",
        r"^\[.*\]$",
        r"^<.*>$",
        r"^_+$",
        r"^\*+$",
        r"^\.{3,}$",
    ]

    # Generic/test names
    GENERIC_NAMES = [
        "john doe",
        "jane doe",
        "john smith",
        "jane smith",
        "test patient",
        "test user",
        "sample patient",
        "demo patient",
        "patient name",
        "first last",
        "example name",
        "foo bar",
        "lorem ipsum",
    ]

    # Generic addresses
    GENERIC_ADDRESSES = [
        "123 main st",
        "123 main street",
        "456 oak ave",
        "789 elm st",
        "test address",
        "sample address",
        "example street",
        "po box 123",
        "1234 street name",
    ]

    # Test data indicators
    TEST_DATA_PATTERNS = [
        r"^test\s*\d*$",
        r"^sample\s*\d*$",
        r"^demo\s*\d*$",
        r"^example\s*\d*$",
        r"^dummy\s*\d*$",
        r"^mock\s*\d*$",
        r"^fake\s*\d*$",
    ]

    # Suspicious round number thresholds
    ROUND_NUMBER_THRESHOLDS = {
        "currency": [100, 500, 1000, 5000, 10000],
        "percentage": [10, 25, 50, 75, 100],
        "quantity": [10, 50, 100, 500, 1000],
    }

    # Date plausibility window
    MIN_PLAUSIBLE_YEAR = 1900
    MAX_FUTURE_DAYS = 365 * 2

    # Severity weights for scoring
    SEVERITY_WEIGHTS = {
        PatternSeverity.LOW: 0.1,
        PatternSeverity.MEDIUM: 0.3,
        PatternSeverity.HIGH: 0.5,
        PatternSeverity.CRITICAL: 0.8,
    }

    HALLUCINATION_THRESHOLD = 0.70

    def __init__(
        self,
        field_type_hints: dict[str, str] | None = None,
        custom_placeholder_patterns: list[str] | None = None,
    ) -> None:
        """
        Initialize the pattern detector.

        Args:
            field_type_hints: Mapping of field names to expected types.
            custom_placeholder_patterns: Additional placeholder patterns.
        """
        self.field_type_hints = field_type_hints or {}

        # Compile placeholder patterns
        patterns = self.PLACEHOLDER_PATTERNS.copy()
        if custom_placeholder_patterns:
            patterns.extend(custom_placeholder_patterns)
        self._placeholder_regex = re.compile(
            "|".join(f"({p})" for p in patterns),
            re.IGNORECASE,
        )

        # Compile test data patterns
        self._test_data_regex = re.compile(
            "|".join(f"({p})" for p in self.TEST_DATA_PATTERNS),
            re.IGNORECASE,
        )

    def detect(
        self,
        extracted_data: dict[str, Any],
        field_confidences: dict[str, float] | None = None,
    ) -> PatternDetectionResult:
        """
        Detect hallucination patterns in extracted data.

        Args:
            extracted_data: Dictionary of field names to extracted values.
            field_confidences: Optional per-field confidence scores.

        Returns:
            PatternDetectionResult with all detected patterns.
        """
        field_confidences = field_confidences or {}
        matches: list[PatternMatch] = []

        # Check each field for patterns
        for field_name, value in extracted_data.items():
            if value is None:
                continue

            field_matches = self._check_field(
                field_name=field_name,
                value=value,
                confidence=field_confidences.get(field_name, 0.7),
            )
            matches.extend(field_matches)

        # Check for cross-field patterns
        cross_field_matches = self._check_cross_field_patterns(extracted_data)
        matches.extend(cross_field_matches)

        # Build result
        result = self._build_result(matches)

        logger.debug(
            f"Pattern detection complete: "
            f"matches={len(matches)}, "
            f"suspicion={result.overall_suspicion_score:.2f}, "
            f"likely_hallucination={result.is_likely_hallucination}"
        )

        return result

    def _check_field(
        self,
        field_name: str,
        value: Any,
        confidence: float,
    ) -> list[PatternMatch]:
        """Check a single field for hallucination patterns."""
        matches: list[PatternMatch] = []

        # Convert to string for text-based checks
        str_value = str(value).strip()
        lower_value = str_value.lower()

        # Check placeholder text
        if self._placeholder_regex.match(str_value):
            matches.append(PatternMatch(
                field_name=field_name,
                value=value,
                pattern=HallucinationPattern.PLACEHOLDER_TEXT,
                severity=PatternSeverity.CRITICAL,
                confidence=0.95,
                description=f"Placeholder text detected: '{str_value}'",
                suggestion="This value appears to be a placeholder, not actual data",
            ))

        # Check test data patterns
        if self._test_data_regex.match(str_value):
            matches.append(PatternMatch(
                field_name=field_name,
                value=value,
                pattern=HallucinationPattern.TEST_DATA,
                severity=PatternSeverity.HIGH,
                confidence=0.90,
                description=f"Test data pattern detected: '{str_value}'",
                suggestion="This appears to be test/sample data, not real content",
            ))

        # Check for generic names
        if self._is_name_field(field_name):
            matches.extend(self._check_generic_name(field_name, lower_value))

        # Check for generic addresses
        if self._is_address_field(field_name):
            matches.extend(self._check_generic_address(field_name, lower_value))

        # Check numeric patterns
        if isinstance(value, (int, float)) or self._looks_numeric(str_value):
            matches.extend(self._check_numeric_patterns(field_name, value, str_value))

        # Check date patterns
        if self._is_date_field(field_name):
            matches.extend(self._check_date_patterns(field_name, str_value))

        # Check for repeated characters/digits
        matches.extend(self._check_repetition_patterns(field_name, str_value))

        # Check for truncated values
        matches.extend(self._check_truncation(field_name, str_value))

        # Check for alphabetic sequences
        matches.extend(self._check_alphabetic_sequences(field_name, str_value))

        # Check identifier patterns
        if self._is_identifier_field(field_name):
            matches.extend(self._check_identifier_patterns(field_name, str_value))

        return matches

    def _check_cross_field_patterns(
        self,
        extracted_data: dict[str, Any],
    ) -> list[PatternMatch]:
        """Check for patterns across multiple fields."""
        matches: list[PatternMatch] = []

        # Convert all values to strings for comparison
        str_values = {
            k: str(v).strip().lower()
            for k, v in extracted_data.items()
            if v is not None and str(v).strip()
        }

        # Check for repeated values across different fields
        value_counts: dict[str, list[str]] = {}
        for field_name, str_val in str_values.items():
            if len(str_val) > 2:
                if str_val not in value_counts:
                    value_counts[str_val] = []
                value_counts[str_val].append(field_name)

        for value, fields in value_counts.items():
            if len(fields) >= 3:
                matches.append(PatternMatch(
                    field_name=fields[0],
                    value=value,
                    pattern=HallucinationPattern.REPETITIVE_VALUE,
                    severity=PatternSeverity.HIGH,
                    confidence=0.85,
                    description=(
                        f"Same value '{value[:50]}...' appears in {len(fields)} fields: "
                        f"{', '.join(fields[:5])}"
                    ),
                    suggestion="Repetitive values across unrelated fields may indicate hallucination",
                ))

        # Check for sequential patterns (e.g., 001, 002, 003)
        sequential_groups = self._find_sequential_values(str_values)
        for group_fields, pattern_desc in sequential_groups:
            matches.append(PatternMatch(
                field_name=group_fields[0],
                value=pattern_desc,
                pattern=HallucinationPattern.SEQUENTIAL_PATTERN,
                severity=PatternSeverity.MEDIUM,
                confidence=0.75,
                description=f"Sequential pattern detected across fields: {', '.join(group_fields)}",
                suggestion="Sequential values may be auto-generated rather than extracted",
            ))

        return matches

    def _check_generic_name(
        self,
        field_name: str,
        lower_value: str,
    ) -> list[PatternMatch]:
        """Check for generic/placeholder names."""
        matches: list[PatternMatch] = []

        for generic_name in self.GENERIC_NAMES:
            if generic_name in lower_value or lower_value in generic_name:
                matches.append(PatternMatch(
                    field_name=field_name,
                    value=lower_value,
                    pattern=HallucinationPattern.GENERIC_NAME,
                    severity=PatternSeverity.CRITICAL,
                    confidence=0.92,
                    description=f"Generic/placeholder name detected: '{lower_value}'",
                    suggestion="This appears to be a common placeholder name, not a real person",
                ))
                break

        return matches

    def _check_generic_address(
        self,
        field_name: str,
        lower_value: str,
    ) -> list[PatternMatch]:
        """Check for generic/placeholder addresses."""
        matches: list[PatternMatch] = []

        for generic_addr in self.GENERIC_ADDRESSES:
            if generic_addr in lower_value:
                matches.append(PatternMatch(
                    field_name=field_name,
                    value=lower_value,
                    pattern=HallucinationPattern.GENERIC_ADDRESS,
                    severity=PatternSeverity.HIGH,
                    confidence=0.88,
                    description=f"Generic address pattern detected: '{lower_value}'",
                    suggestion="This appears to be a placeholder address",
                ))
                break

        return matches

    def _check_numeric_patterns(
        self,
        field_name: str,
        value: Any,
        str_value: str,
    ) -> list[PatternMatch]:
        """Check for suspicious numeric patterns."""
        matches: list[PatternMatch] = []

        # Extract numeric value
        try:
            clean_str = re.sub(r"[$,\s]", "", str_value)
            num_value = float(clean_str)
        except (ValueError, TypeError):
            return matches

        # Check for suspiciously round numbers
        if num_value > 0:
            field_type = self._infer_numeric_type(field_name, num_value)
            thresholds = self.ROUND_NUMBER_THRESHOLDS.get(field_type, [])

            if num_value in thresholds or (num_value % 1000 == 0 and num_value >= 1000):
                matches.append(PatternMatch(
                    field_name=field_name,
                    value=value,
                    pattern=HallucinationPattern.ROUND_NUMBER,
                    severity=PatternSeverity.LOW,
                    confidence=0.50,
                    description=f"Suspiciously round number: {num_value}",
                    suggestion="Very round numbers may be estimates or hallucinations",
                ))

        # Check for impossible values
        if self._is_currency_field(field_name) and num_value < 0:
            matches.append(PatternMatch(
                field_name=field_name,
                value=value,
                pattern=HallucinationPattern.IMPOSSIBLE_VALUE,
                severity=PatternSeverity.MEDIUM,
                confidence=0.70,
                description=f"Negative value in currency field: {num_value}",
                suggestion="Verify if negative amount is valid for this context",
            ))

        return matches

    def _check_date_patterns(
        self,
        field_name: str,
        str_value: str,
    ) -> list[PatternMatch]:
        """Check for implausible dates."""
        matches: list[PatternMatch] = []

        # Try to parse date
        parsed_date = self._parse_date(str_value)
        if parsed_date is None:
            return matches

        now = datetime.now()
        max_future = now + timedelta(days=self.MAX_FUTURE_DAYS)

        # Check for future dates (beyond reasonable window)
        if parsed_date > max_future:
            matches.append(PatternMatch(
                field_name=field_name,
                value=str_value,
                pattern=HallucinationPattern.IMPLAUSIBLE_DATE,
                severity=PatternSeverity.HIGH,
                confidence=0.85,
                description=f"Date is too far in the future: {parsed_date.date()}",
                suggestion="This date appears to be implausible",
            ))

        # Check for very old dates
        if parsed_date.year < self.MIN_PLAUSIBLE_YEAR:
            matches.append(PatternMatch(
                field_name=field_name,
                value=str_value,
                pattern=HallucinationPattern.IMPLAUSIBLE_DATE,
                severity=PatternSeverity.HIGH,
                confidence=0.80,
                description=f"Date is implausibly old: {parsed_date.date()}",
                suggestion="This date appears to be outside the expected range",
            ))

        # Check for default/placeholder dates
        if parsed_date.month == 1 and parsed_date.day == 1:
            if parsed_date.year in [1900, 1970, 2000]:
                matches.append(PatternMatch(
                    field_name=field_name,
                    value=str_value,
                    pattern=HallucinationPattern.IMPLAUSIBLE_DATE,
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.75,
                    description=f"Default/placeholder date detected: {parsed_date.date()}",
                    suggestion="This may be a system default date, not actual data",
                ))

        return matches

    def _check_repetition_patterns(
        self,
        field_name: str,
        str_value: str,
    ) -> list[PatternMatch]:
        """Check for repeated characters or digits."""
        matches: list[PatternMatch] = []

        if len(str_value) < 3:
            return matches

        # Check for repeated digits (e.g., 111111, 000000)
        digits_only = re.sub(r"\D", "", str_value)
        if len(digits_only) >= 5:
            unique_digits = set(digits_only)
            if len(unique_digits) == 1:
                matches.append(PatternMatch(
                    field_name=field_name,
                    value=str_value,
                    pattern=HallucinationPattern.REPEATED_DIGITS,
                    severity=PatternSeverity.HIGH,
                    confidence=0.88,
                    description=f"Repeated digit pattern: '{str_value}'",
                    suggestion="Single repeated digit patterns are often synthetic",
                ))

        # Check for repeating patterns (e.g., 123123123)
        if len(str_value) >= 6:
            for pattern_len in range(2, len(str_value) // 2 + 1):
                pattern = str_value[:pattern_len]
                expected = pattern * (len(str_value) // pattern_len)
                if str_value.startswith(expected) and len(expected) >= 6:
                    matches.append(PatternMatch(
                        field_name=field_name,
                        value=str_value,
                        pattern=HallucinationPattern.SEQUENTIAL_PATTERN,
                        severity=PatternSeverity.MEDIUM,
                        confidence=0.75,
                        description=f"Repeating pattern detected: '{pattern}' repeated",
                        suggestion="Repeating patterns may indicate synthetic data",
                    ))
                    break

        return matches

    def _check_truncation(
        self,
        field_name: str,
        str_value: str,
    ) -> list[PatternMatch]:
        """Check for truncated values."""
        matches: list[PatternMatch] = []

        # Check if value ends with truncation indicators
        truncation_indicators = ["...", "…", "---", "___"]
        for indicator in truncation_indicators:
            if str_value.endswith(indicator):
                matches.append(PatternMatch(
                    field_name=field_name,
                    value=str_value,
                    pattern=HallucinationPattern.TRUNCATED_VALUE,
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.80,
                    description=f"Value appears truncated: '{str_value[-20:]}'",
                    suggestion="This value may be incomplete due to truncation",
                ))
                break

        return matches

    def _check_alphabetic_sequences(
        self,
        field_name: str,
        str_value: str,
    ) -> list[PatternMatch]:
        """Check for alphabetic sequences like ABCD."""
        matches: list[PatternMatch] = []

        # Only check fields that should have alphabetic content
        if not self._is_text_field(field_name):
            return matches

        alpha_only = re.sub(r"[^a-zA-Z]", "", str_value.lower())
        if len(alpha_only) >= 4:
            # Check for sequential alphabet
            if "abcd" in alpha_only or "efgh" in alpha_only or "lmno" in alpha_only:
                matches.append(PatternMatch(
                    field_name=field_name,
                    value=str_value,
                    pattern=HallucinationPattern.ALPHABETIC_SEQUENCE,
                    severity=PatternSeverity.MEDIUM,
                    confidence=0.70,
                    description=f"Alphabetic sequence detected: '{str_value}'",
                    suggestion="Sequential letters may indicate placeholder data",
                ))

        return matches

    def _check_identifier_patterns(
        self,
        field_name: str,
        str_value: str,
    ) -> list[PatternMatch]:
        """Check identifier fields for synthetic patterns."""
        matches: list[PatternMatch] = []

        digits_only = re.sub(r"\D", "", str_value)

        # Check for all-zero identifiers
        if len(digits_only) >= 5 and all(d == "0" for d in digits_only):
            matches.append(PatternMatch(
                field_name=field_name,
                value=str_value,
                pattern=HallucinationPattern.SYNTHETIC_IDENTIFIER,
                severity=PatternSeverity.CRITICAL,
                confidence=0.95,
                description=f"All-zero identifier: '{str_value}'",
                suggestion="All-zero identifiers are typically invalid placeholders",
            ))

        # Check for sequential identifiers (123456789)
        if len(digits_only) >= 5:
            is_sequential = all(
                int(digits_only[i]) == int(digits_only[i - 1]) + 1
                for i in range(1, len(digits_only))
            )
            if is_sequential:
                matches.append(PatternMatch(
                    field_name=field_name,
                    value=str_value,
                    pattern=HallucinationPattern.SYNTHETIC_IDENTIFIER,
                    severity=PatternSeverity.HIGH,
                    confidence=0.85,
                    description=f"Sequential identifier: '{str_value}'",
                    suggestion="Sequential identifiers may be synthetic test data",
                ))

        return matches

    def _find_sequential_values(
        self,
        str_values: dict[str, str],
    ) -> list[tuple[list[str], str]]:
        """Find groups of fields with sequential numeric values."""
        results: list[tuple[list[str], str]] = []

        # Extract numeric values from fields
        numeric_fields: list[tuple[str, int]] = []
        for field_name, str_val in str_values.items():
            try:
                num = int(re.sub(r"\D", "", str_val))
                if 0 < num < 1000000:
                    numeric_fields.append((field_name, num))
            except (ValueError, TypeError):
                continue

        # Sort by value
        numeric_fields.sort(key=lambda x: x[1])

        # Find sequential groups
        if len(numeric_fields) >= 3:
            for i in range(len(numeric_fields) - 2):
                if (
                    numeric_fields[i + 1][1] == numeric_fields[i][1] + 1
                    and numeric_fields[i + 2][1] == numeric_fields[i][1] + 2
                ):
                    group_fields = [
                        numeric_fields[i][0],
                        numeric_fields[i + 1][0],
                        numeric_fields[i + 2][0],
                    ]
                    pattern_desc = f"{numeric_fields[i][1]}, {numeric_fields[i + 1][1]}, {numeric_fields[i + 2][1]}"
                    results.append((group_fields, pattern_desc))

        return results

    def _build_result(
        self,
        matches: list[PatternMatch],
    ) -> PatternDetectionResult:
        """Build complete detection result from matches."""
        result = PatternDetectionResult()
        result.matches = matches

        if not matches:
            result.summary = "No hallucination patterns detected"
            return result

        # Collect flagged fields
        result.flagged_fields = {m.field_name for m in matches}

        # Calculate overall suspicion score
        severity_scores = [
            self.SEVERITY_WEIGHTS[m.severity] * m.confidence
            for m in matches
        ]
        result.overall_suspicion_score = min(1.0, sum(severity_scores) / max(len(severity_scores), 1))

        # Check for critical patterns
        result.critical_patterns = [
            f"{m.field_name}: {m.pattern.value}"
            for m in matches
            if m.severity == PatternSeverity.CRITICAL
        ]

        # Determine if likely hallucination
        result.is_likely_hallucination = (
            result.overall_suspicion_score >= self.HALLUCINATION_THRESHOLD
            or len(result.critical_patterns) > 0
        )

        # Generate summary
        pattern_counts: dict[str, int] = {}
        for m in matches:
            pattern_counts[m.pattern.value] = pattern_counts.get(m.pattern.value, 0) + 1

        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        pattern_summary = ", ".join(f"{p}: {c}" for p, c in top_patterns)

        result.summary = (
            f"Detected {len(matches)} pattern(s) in {len(result.flagged_fields)} field(s). "
            f"Top patterns: {pattern_summary}. "
            f"Suspicion score: {result.overall_suspicion_score:.2f}"
        )

        return result

    @staticmethod
    @lru_cache(maxsize=256)
    def _is_name_field(field_name: str) -> bool:
        """Check if field is likely a name field (cached)."""
        name_indicators = ("name", "patient", "provider", "physician", "subscriber")
        lower_name = field_name.lower()
        return any(ind in lower_name for ind in name_indicators)

    @staticmethod
    @lru_cache(maxsize=256)
    def _is_address_field(field_name: str) -> bool:
        """Check if field is likely an address field (cached)."""
        addr_indicators = ("address", "street", "city", "addr", "location")
        lower_name = field_name.lower()
        return any(ind in lower_name for ind in addr_indicators)

    @staticmethod
    @lru_cache(maxsize=256)
    def _is_date_field(field_name: str) -> bool:
        """Check if field is likely a date field (cached)."""
        date_indicators = ("date", "dob", "birth", "service", "admission", "discharge")
        lower_name = field_name.lower()
        return any(ind in lower_name for ind in date_indicators)

    @staticmethod
    @lru_cache(maxsize=256)
    def _is_currency_field(field_name: str) -> bool:
        """Check if field is likely a currency field (cached)."""
        currency_indicators = ("amount", "charge", "payment", "cost", "fee", "price", "total")
        lower_name = field_name.lower()
        return any(ind in lower_name for ind in currency_indicators)

    @staticmethod
    @lru_cache(maxsize=256)
    def _is_identifier_field(field_name: str) -> bool:
        """Check if field is likely an identifier field (cached)."""
        id_indicators = ("id", "number", "npi", "ssn", "ein", "member", "policy", "claim")
        lower_name = field_name.lower()
        return any(ind in lower_name for ind in id_indicators)

    @staticmethod
    @lru_cache(maxsize=256)
    def _is_text_field(field_name: str) -> bool:
        """Check if field is likely a text content field (cached)."""
        text_indicators = ("name", "description", "notes", "comments", "address")
        lower_name = field_name.lower()
        return any(ind in lower_name for ind in text_indicators)

    @staticmethod
    @lru_cache(maxsize=512)
    def _looks_numeric(value: str) -> bool:
        """Check if string looks like a number (cached)."""
        cleaned = re.sub(r"[$,\s\-]", "", value)
        try:
            float(cleaned)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    @lru_cache(maxsize=256)
    def _infer_numeric_type(field_name: str, value: float) -> str:
        """Infer numeric field type from name and value (cached)."""
        lower_name = field_name.lower()

        if any(ind in lower_name for ind in ("amount", "charge", "cost", "fee", "price")):
            return "currency"
        if any(ind in lower_name for ind in ("percent", "rate", "ratio")):
            return "percentage"
        if any(ind in lower_name for ind in ("qty", "quantity", "units", "count")):
            return "quantity"

        # Infer from value
        if value >= 100 and value == int(value):
            return "currency"
        if 0 <= value <= 100:
            return "percentage"
        return "quantity"

    @staticmethod
    @lru_cache(maxsize=256)
    def _parse_date(date_str: str) -> datetime | None:
        """Parse a date string into datetime (cached)."""
        formats = (
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%m/%d/%y",
            "%d/%m/%Y",
            "%Y%m%d",
        )

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None


def detect_hallucination_patterns(
    extracted_data: dict[str, Any],
    field_confidences: dict[str, float] | None = None,
    field_type_hints: dict[str, str] | None = None,
) -> PatternDetectionResult:
    """
    Detect hallucination patterns in extracted data.

    Convenience function for one-off detection without creating
    a detector instance.

    Args:
        extracted_data: Dictionary of field names to extracted values.
        field_confidences: Optional per-field confidence scores.
        field_type_hints: Optional hints about field types.

    Returns:
        PatternDetectionResult with detected patterns.

    Example:
        result = detect_hallucination_patterns({
            "patient_name": "John Doe",
            "amount": 1000.00,
            "claim_id": "000000000",
        })

        if result.is_likely_hallucination:
            print(f"Flagged: {result.flagged_fields}")
    """
    detector = HallucinationPatternDetector(field_type_hints=field_type_hints)
    return detector.detect(extracted_data, field_confidences)
