"""
Tests for src/utils/string_utils.py — string manipulation and matching utilities.
"""

from decimal import Decimal

import pytest

from src.utils.string_utils import (
    clean_currency,
    clean_ocr_text,
    extract_between,
    extract_integers,
    extract_numbers,
    fuzzy_match,
    is_empty_or_whitespace,
    levenshtein_distance,
    normalize_name,
    normalize_whitespace,
    pad_string,
    remove_diacritics,
    safe_string,
    similarity_ratio,
    split_on_pattern,
    truncate_text,
)


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:

    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_whitespace("hello\t\tworld\n\nfoo") == "hello world foo"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hi  ") == "hi"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_none_like_empty(self):
        # function guards on falsy input
        assert normalize_whitespace("") == ""

    def test_already_normalized(self):
        assert normalize_whitespace("hello world") == "hello world"


# ---------------------------------------------------------------------------
# normalize_name
# ---------------------------------------------------------------------------


class TestNormalizeName:

    def test_first_last_to_last_first(self):
        assert normalize_name("John Smith") == "SMITH, JOHN"

    def test_already_last_comma_first(self):
        assert normalize_name("Smith, John A") == "SMITH, JOHN A"

    def test_removes_prefix_dr(self):
        assert normalize_name("Dr. John Smith") == "SMITH, JOHN"

    def test_removes_suffix_md(self):
        assert normalize_name("John Smith MD") == "SMITH, JOHN"

    def test_single_name(self):
        assert normalize_name("Smith") == "SMITH"

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_first_middle_last(self):
        result = normalize_name("John A Smith")
        assert result == "SMITH, JOHN A"


# ---------------------------------------------------------------------------
# extract_numbers / extract_integers
# ---------------------------------------------------------------------------


class TestExtractNumbers:

    def test_extracts_integers_and_decimals(self):
        assert extract_numbers("Total $150.00 for 3 items") == ["150.00", "3"]

    def test_negative_numbers(self):
        assert extract_numbers("Balance: -42.50") == ["-42.50"]

    def test_no_numbers(self):
        assert extract_numbers("hello world") == []

    def test_empty_string(self):
        assert extract_numbers("") == []


class TestExtractIntegers:

    def test_extracts_ints(self):
        assert extract_integers("Page 1 of 10") == [1, 10]

    def test_negative_ints(self):
        assert extract_integers("offset -5") == [-5]

    def test_empty(self):
        assert extract_integers("") == []


# ---------------------------------------------------------------------------
# clean_currency
# ---------------------------------------------------------------------------


class TestCleanCurrency:

    def test_standard_dollar(self):
        assert clean_currency("$1,234.56") == Decimal("1234.56")

    def test_parentheses_negative(self):
        assert clean_currency("($500.00)") == Decimal("-500.00")

    def test_cr_negative(self):
        assert clean_currency("100.00CR") == Decimal("-100.00")

    def test_plain_integer(self):
        assert clean_currency("1234") == Decimal("1234")

    def test_empty_returns_none(self):
        assert clean_currency("") is None

    def test_none_returns_none(self):
        assert clean_currency(None) is None

    def test_euro_symbol(self):
        assert clean_currency("€50.00") == Decimal("50.00")

    def test_dash_negative(self):
        assert clean_currency("200.00-") == Decimal("-200.00")


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


class TestTruncateText:

    def test_no_truncation_needed(self):
        assert truncate_text("Hi", 10) == "Hi"

    def test_truncates_with_suffix(self):
        result = truncate_text("Hello World Foo Bar", 10)
        assert result.endswith("...")
        assert len(result) <= 10

    def test_word_boundary_false(self):
        result = truncate_text("Hello World", 8, word_boundary=False)
        assert result.endswith("...")

    def test_empty_string(self):
        assert truncate_text("", 5) == ""

    def test_custom_suffix(self):
        result = truncate_text("Hello World Foo", 10, suffix="..")
        assert result.endswith("..")


# ---------------------------------------------------------------------------
# levenshtein_distance
# ---------------------------------------------------------------------------


class TestLevenshteinDistance:

    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_classic_example(self):
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert levenshtein_distance("abc", "") == 3

    def test_single_char_diff(self):
        assert levenshtein_distance("cat", "hat") == 1


# ---------------------------------------------------------------------------
# fuzzy_match
# ---------------------------------------------------------------------------


class TestFuzzyMatch:

    def test_identical_match(self):
        assert fuzzy_match("Smith", "Smith") is True

    def test_close_match(self):
        assert fuzzy_match("Smith", "Smyth") is True

    def test_no_match(self):
        assert fuzzy_match("John", "Jane", threshold=0.9) is False

    def test_case_insensitive_default(self):
        assert fuzzy_match("SMITH", "smith") is True

    def test_both_empty(self):
        assert fuzzy_match("", "") is True

    def test_one_empty(self):
        assert fuzzy_match("abc", "") is False


# ---------------------------------------------------------------------------
# similarity_ratio
# ---------------------------------------------------------------------------


class TestSimilarityRatio:

    def test_identical(self):
        assert similarity_ratio("hello", "hello") == 1.0

    def test_completely_different(self):
        assert similarity_ratio("abc", "xyz") < 0.5

    def test_both_empty(self):
        assert similarity_ratio("", "") == 1.0

    def test_one_empty(self):
        assert similarity_ratio("abc", "") == 0.0

    def test_close_strings(self):
        ratio = similarity_ratio("Smith", "Smyth")
        assert 0.7 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# remove_diacritics
# ---------------------------------------------------------------------------


class TestRemoveDiacritics:

    def test_accented_chars(self):
        assert remove_diacritics("José García") == "Jose Garcia"

    def test_no_diacritics(self):
        assert remove_diacritics("Hello") == "Hello"

    def test_empty(self):
        assert remove_diacritics("") == ""

    def test_umlaut(self):
        assert remove_diacritics("über") == "uber"


# ---------------------------------------------------------------------------
# clean_ocr_text
# ---------------------------------------------------------------------------


class TestCleanOcrText:

    def test_removes_control_chars(self):
        assert "\x00" not in clean_ocr_text("Hello\x00World")

    def test_normalizes_quotes(self):
        result = clean_ocr_text("\u201cHello\u201d")
        assert result == '"Hello"'

    def test_normalizes_single_quotes(self):
        result = clean_ocr_text("\u2018it\u2019s")
        assert result == "'it's"

    def test_removes_zero_width_chars(self):
        assert "\u200b" not in clean_ocr_text("Hello\u200bWorld")

    def test_empty(self):
        assert clean_ocr_text("") == ""


# ---------------------------------------------------------------------------
# extract_between
# ---------------------------------------------------------------------------


class TestExtractBetween:

    def test_extracts_content(self):
        assert extract_between("Name: John Smith, Age:", "Name: ", ", Age:") == "John Smith"

    def test_inclusive(self):
        result = extract_between("Name: John, Age:", "Name: ", ", Age:", inclusive=True)
        assert result == "Name: John, Age:"

    def test_not_found(self):
        assert extract_between("Hello", "X", "Y") is None

    def test_empty_text(self):
        assert extract_between("", "a", "b") is None


# ---------------------------------------------------------------------------
# pad_string
# ---------------------------------------------------------------------------


class TestPadString:

    def test_left_align(self):
        assert pad_string("Hi", 5) == "Hi   "

    def test_right_align(self):
        assert pad_string("Hi", 5, align="right") == "   Hi"

    def test_center_align(self):
        result = pad_string("Hi", 6, align="center")
        assert len(result) == 6

    def test_already_long_enough(self):
        assert pad_string("Hello", 3) == "Hel"

    def test_custom_pad_char(self):
        assert pad_string("Hi", 5, pad_char="0", align="right") == "000Hi"


# ---------------------------------------------------------------------------
# split_on_pattern
# ---------------------------------------------------------------------------


class TestSplitOnPattern:

    def test_split_on_semicolon(self):
        assert split_on_pattern("a;b;c", ";") == ["a", "b", "c"]

    def test_keep_delimiter(self):
        parts = split_on_pattern("Name: John Age: 30", r"[A-Z]\w+:", keep_delimiter=True)
        assert len(parts) >= 2

    def test_empty_string(self):
        assert split_on_pattern("", ";") == []


# ---------------------------------------------------------------------------
# is_empty_or_whitespace / safe_string
# ---------------------------------------------------------------------------


class TestIsEmptyOrWhitespace:

    def test_none(self):
        assert is_empty_or_whitespace(None) is True

    def test_empty(self):
        assert is_empty_or_whitespace("") is True

    def test_whitespace_only(self):
        assert is_empty_or_whitespace("   \t\n") is True

    def test_non_empty(self):
        assert is_empty_or_whitespace("hello") is False


class TestSafeString:

    def test_none_returns_default(self):
        assert safe_string(None) == ""

    def test_int_conversion(self):
        assert safe_string(42) == "42"

    def test_custom_default(self):
        assert safe_string(None, "N/A") == "N/A"

    def test_string_passthrough(self):
        assert safe_string("hello") == "hello"
