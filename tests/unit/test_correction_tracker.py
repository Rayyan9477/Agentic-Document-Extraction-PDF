"""
Tests for src/memory/correction_tracker.py — correction tracking and learning.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.memory.correction_tracker import Correction, CorrectionTracker


# ---------------------------------------------------------------------------
# Correction dataclass
# ---------------------------------------------------------------------------


class TestCorrection:

    def test_create_with_defaults(self):
        c = Correction(
            id="c1",
            field_name="patient_name",
            original_value="Jon",
            corrected_value="John",
            document_type="cms1500",
        )
        assert c.id == "c1"
        assert c.confidence_before == 0.0
        assert c.correction_type == "value"
        assert c.user_id == "default"
        assert c.metadata == {}

    def test_create_with_all_fields(self):
        c = Correction(
            id="c2",
            field_name="npi",
            original_value="123",
            corrected_value="456",
            document_type="eob",
            confidence_before=0.75,
            correction_type="format",
            user_id="user1",
            metadata={"source": "manual"},
        )
        assert c.correction_type == "format"
        assert c.user_id == "user1"

    def test_to_dict(self):
        c = Correction(
            id="c3",
            field_name="dob",
            original_value="01-01-1990",
            corrected_value="1990-01-01",
            document_type="cms1500",
        )
        d = c.to_dict()
        assert d["id"] == "c3"
        assert d["field_name"] == "dob"
        assert "created_at" in d

    def test_from_dict(self):
        data = {
            "id": "c4",
            "field_name": "amount",
            "original_value": "100",
            "corrected_value": "100.00",
            "document_type": "eob",
            "confidence_before": 0.80,
            "correction_type": "format",
        }
        c = Correction.from_dict(data)
        assert c.id == "c4"
        assert c.confidence_before == 0.80

    def test_from_dict_missing_optional(self):
        data = {"id": "c5", "field_name": "x"}
        c = Correction.from_dict(data)
        assert c.document_type == ""
        assert c.correction_type == "value"

    def test_round_trip(self):
        c = Correction(
            id="rt",
            field_name="name",
            original_value="A",
            corrected_value="B",
            document_type="cms1500",
            confidence_before=0.5,
        )
        d = c.to_dict()
        c2 = Correction.from_dict(d)
        assert c2.id == c.id
        assert c2.field_name == c.field_name
        assert c2.confidence_before == c.confidence_before


# ---------------------------------------------------------------------------
# CorrectionTracker — init and persistence
# ---------------------------------------------------------------------------


class TestCorrectionTrackerInit:

    @patch("src.memory.correction_tracker.Mem0Client")
    @patch("src.memory.correction_tracker.get_settings")
    def test_init_creates_tracker(self, mock_settings, mock_mem0):
        mock_settings.return_value.mem0.data_dir = "/tmp/test_corrections"
        tracker = CorrectionTracker(
            mem0_client=MagicMock(),
            data_dir="/tmp/test_corrections",
        )
        assert isinstance(tracker._corrections, dict)

    @patch("src.memory.correction_tracker.Mem0Client")
    @patch("src.memory.correction_tracker.get_settings")
    def test_loads_existing_corrections(self, mock_settings, mock_mem0, tmp_path):
        mock_settings.return_value.mem0.data_dir = str(tmp_path)
        # Pre-populate corrections file
        corrections_file = tmp_path / "corrections.json"
        corrections_file.write_text(
            json.dumps(
                {
                    "c1": {
                        "id": "c1",
                        "field_name": "name",
                        "original_value": "A",
                        "corrected_value": "B",
                        "document_type": "cms1500",
                    }
                }
            ),
            encoding="utf-8",
        )
        tracker = CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)
        assert len(tracker._corrections) == 1

    @patch("src.memory.correction_tracker.Mem0Client")
    @patch("src.memory.correction_tracker.get_settings")
    def test_handles_corrupt_file(self, mock_settings, mock_mem0, tmp_path):
        mock_settings.return_value.mem0.data_dir = str(tmp_path)
        corrections_file = tmp_path / "corrections.json"
        corrections_file.write_text("{bad json", encoding="utf-8")
        tracker = CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)
        assert len(tracker._corrections) == 0


# ---------------------------------------------------------------------------
# CorrectionTracker — record_correction
# ---------------------------------------------------------------------------


class TestRecordCorrection:

    @pytest.fixture()
    def tracker(self, tmp_path):
        with patch("src.memory.correction_tracker.Mem0Client"), \
             patch("src.memory.correction_tracker.get_settings") as mock_settings:
            mock_settings.return_value.mem0.data_dir = str(tmp_path)
            t = CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)
            return t

    def test_record_returns_correction(self, tracker):
        c = tracker.record_correction(
            field_name="patient_name",
            original_value="Jon",
            corrected_value="John",
            document_type="cms1500",
        )
        assert isinstance(c, Correction)
        assert c.field_name == "patient_name"
        assert c.id != ""

    def test_correction_persisted(self, tracker, tmp_path):
        tracker.record_correction(
            field_name="dob",
            original_value="01/01/90",
            corrected_value="1990-01-01",
            document_type="eob",
        )
        corrections_file = tmp_path / "corrections.json"
        assert corrections_file.exists()
        data = json.loads(corrections_file.read_text(encoding="utf-8"))
        assert len(data) == 1

    def test_stores_in_memory(self, tracker):
        tracker.record_correction(
            field_name="npi",
            original_value="000",
            corrected_value="123",
            document_type="cms1500",
        )
        tracker._client.add.assert_called_once()


# ---------------------------------------------------------------------------
# CorrectionTracker — get_corrections_for_field
# ---------------------------------------------------------------------------


class TestGetCorrectionsForField:

    @pytest.fixture()
    def tracker_with_data(self, tmp_path):
        with patch("src.memory.correction_tracker.Mem0Client"), \
             patch("src.memory.correction_tracker.get_settings") as mock_settings:
            mock_settings.return_value.mem0.data_dir = str(tmp_path)
            t = CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)
            # Use distinct document_types to ensure unique generated IDs
            for i in range(5):
                t.record_correction(
                    field_name="name",
                    original_value=f"v{i}",
                    corrected_value=f"cv{i}",
                    document_type=f"type_{i}",
                    confidence_before=0.5,
                )
            t.record_correction(
                field_name="dob",
                original_value="bad",
                corrected_value="good",
                document_type="cms1500",
            )
            return t

    def test_returns_field_corrections(self, tracker_with_data):
        corrections = tracker_with_data.get_corrections_for_field("name")
        assert len(corrections) == 5
        assert all(c.field_name == "name" for c in corrections)

    def test_limit_parameter(self, tracker_with_data):
        corrections = tracker_with_data.get_corrections_for_field("name", limit=2)
        assert len(corrections) == 2

    def test_empty_for_unknown_field(self, tracker_with_data):
        corrections = tracker_with_data.get_corrections_for_field("unknown")
        assert corrections == []


# ---------------------------------------------------------------------------
# CorrectionTracker — get_field_hints
# ---------------------------------------------------------------------------


class TestGetFieldHints:

    @pytest.fixture()
    def tracker_with_data(self, tmp_path):
        with patch("src.memory.correction_tracker.Mem0Client"), \
             patch("src.memory.correction_tracker.get_settings") as mock_settings:
            mock_settings.return_value.mem0.data_dir = str(tmp_path)
            t = CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)
            for i in range(5):
                t.record_correction(
                    field_name="amount",
                    original_value=f"{100 + i}",
                    corrected_value=f"{200 + i}",
                    document_type=f"eob_{i}",
                    confidence_before=0.6,
                )
            return t

    def test_hints_for_tracked_field(self, tracker_with_data):
        hints = tracker_with_data.get_field_hints("amount")
        assert hints["total_corrections"] == 5
        assert hints["confidence_boost"] < 0  # penalty for corrections

    def test_hints_for_unknown_field(self, tracker_with_data):
        hints = tracker_with_data.get_field_hints("unknown")
        assert hints == {}

    def test_confidence_boost_tiers(self, tmp_path):
        with patch("src.memory.correction_tracker.Mem0Client"), \
             patch("src.memory.correction_tracker.get_settings") as mock_settings:
            mock_settings.return_value.mem0.data_dir = str(tmp_path)
            t = CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)

            # 1 correction → small penalty
            t.record_correction(
                field_name="f1", original_value="a", corrected_value="b",
                document_type="x", confidence_before=0.5,
            )
            hints1 = t.get_field_hints("f1")
            assert hints1["confidence_boost"] == -0.05


# ---------------------------------------------------------------------------
# CorrectionTracker — get_statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:

    @pytest.fixture()
    def tracker(self, tmp_path):
        with patch("src.memory.correction_tracker.Mem0Client"), \
             patch("src.memory.correction_tracker.get_settings") as mock_settings:
            mock_settings.return_value.mem0.data_dir = str(tmp_path)
            return CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)

    def test_empty_stats(self, tracker):
        stats = tracker.get_statistics()
        assert stats["total_corrections"] == 0
        assert stats["fields_with_corrections"] == 0

    def test_stats_after_corrections(self, tracker):
        tracker.record_correction(
            field_name="name", original_value="A", corrected_value="B",
            document_type="cms1500",
        )
        tracker.record_correction(
            field_name="dob", original_value="X", corrected_value="Y",
            document_type="cms1500",
        )
        stats = tracker.get_statistics()
        assert stats["total_corrections"] == 2
        assert stats["fields_with_corrections"] == 2
        assert len(stats["top_corrected_fields"]) == 2


# ---------------------------------------------------------------------------
# CorrectionTracker — apply_learned_patterns
# ---------------------------------------------------------------------------


class TestApplyLearnedPatterns:

    @pytest.fixture()
    def tracker_with_patterns(self, tmp_path):
        with patch("src.memory.correction_tracker.Mem0Client"), \
             patch("src.memory.correction_tracker.get_settings") as mock_settings:
            mock_settings.return_value.mem0.data_dir = str(tmp_path)
            t = CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)
            t.record_correction(
                field_name="date",
                original_value="01/01/90",
                corrected_value="1990-01-01",
                document_type="cms1500",
            )
            return t

    def test_string_match_passthrough(self, tracker_with_patterns):
        # String value matches error but apply_learned_patterns only flags dict values
        extraction = {"date": "01/01/90"}
        enhanced = tracker_with_patterns.apply_learned_patterns(extraction, "cms1500")
        # String values can't have needs_review added; extraction is unchanged
        assert enhanced["date"] == "01/01/90"

    def test_no_match_passes_through(self, tracker_with_patterns):
        extraction = {"date": "2024-03-15"}
        enhanced = tracker_with_patterns.apply_learned_patterns(extraction, "cms1500")
        assert enhanced["date"] == "2024-03-15"

    def test_unknown_field_passes_through(self, tracker_with_patterns):
        extraction = {"unknown_field": "value"}
        enhanced = tracker_with_patterns.apply_learned_patterns(extraction, "cms1500")
        assert enhanced["unknown_field"] == "value"


# ---------------------------------------------------------------------------
# CorrectionTracker — clear
# ---------------------------------------------------------------------------


class TestClear:

    @pytest.fixture()
    def tracker(self, tmp_path):
        with patch("src.memory.correction_tracker.Mem0Client"), \
             patch("src.memory.correction_tracker.get_settings") as mock_settings:
            mock_settings.return_value.mem0.data_dir = str(tmp_path)
            t = CorrectionTracker(mem0_client=MagicMock(), data_dir=tmp_path)
            t.record_correction(
                field_name="x", original_value="a", corrected_value="b",
                document_type="test",
            )
            return t

    def test_clear_returns_count(self, tracker):
        count = tracker.clear()
        assert count == 1

    def test_clear_empties_corrections(self, tracker):
        tracker.clear()
        assert len(tracker._corrections) == 0

    def test_clear_resets_stats(self, tracker):
        tracker.clear()
        stats = tracker.get_statistics()
        assert stats["total_corrections"] == 0
