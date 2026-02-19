"""
Tests for src/storage/result_store.py — file-based extraction result storage.
"""

import json
import time

import pytest

from src.storage.result_store import ResultStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_result(processing_id="proc_abc", doc_type="cms1500"):
    return {
        "processing_id": processing_id,
        "document_type": doc_type,
        "status": "completed",
        "overall_confidence": 0.92,
        "fields": {"patient_name": "John Smith"},
    }


# ---------------------------------------------------------------------------
# ResultStore init
# ---------------------------------------------------------------------------


class TestResultStoreInit:

    def test_creates_storage_dir(self, tmp_path):
        store_dir = tmp_path / "results"
        store = ResultStore(storage_dir=store_dir)
        assert store_dir.exists()

    def test_default_max_age(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        assert store._max_age_hours == 24 * 7

    def test_custom_max_age(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path, max_age_hours=48)
        assert store._max_age_hours == 48


# ---------------------------------------------------------------------------
# save + get round-trip
# ---------------------------------------------------------------------------


class TestSaveAndGet:

    def test_save_returns_path(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        path = store.save("proc_001", _sample_result())
        assert path.exists()
        assert path.suffix == ".json"

    def test_get_returns_saved_data(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        store.save("proc_002", _sample_result("proc_002"))
        result = store.get("proc_002")
        assert result is not None
        assert result["processing_id"] == "proc_002"
        assert result["document_type"] == "cms1500"

    def test_get_includes_storage_metadata(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        store.save("proc_003", _sample_result("proc_003"))
        result = store.get("proc_003")
        assert "_storage_metadata" in result
        assert result["_storage_metadata"]["processing_id"] == "proc_003"

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        assert store.get("nonexistent") is None

    def test_save_creates_subdirectory(self, tmp_path):
        """First 2 chars of processing_id become a subdirectory."""
        store = ResultStore(storage_dir=tmp_path)
        path = store.save("ab_test", _sample_result())
        assert "ab" in str(path.parent.name)

    def test_short_id_uses_fallback_subdir(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        path = store.save("x", _sample_result())
        assert path.exists()


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExists:

    def test_exists_true(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        store.save("proc_e1", _sample_result())
        assert store.exists("proc_e1") is True

    def test_exists_false(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        assert store.exists("no_such_id") is False


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:

    def test_delete_existing(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        store.save("proc_del", _sample_result())
        assert store.delete("proc_del") is True
        assert store.get("proc_del") is None

    def test_delete_nonexistent(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        assert store.delete("nope") is False


# ---------------------------------------------------------------------------
# list_results
# ---------------------------------------------------------------------------


class TestListResults:

    def test_list_empty(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        assert store.list_results() == []

    def test_list_returns_entries(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        store.save("proc_l1", _sample_result("proc_l1"))
        store.save("proc_l2", _sample_result("proc_l2", "eob"))
        results = store.list_results()
        assert len(results) == 2

    def test_list_pagination(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        for i in range(5):
            store.save(f"proc_p{i}", _sample_result(f"proc_p{i}"))
            time.sleep(0.01)  # ensure distinct mtime
        page = store.list_results(limit=2, offset=0)
        assert len(page) == 2

    def test_list_entry_has_expected_keys(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        store.save("proc_k1", _sample_result("proc_k1"))
        entries = store.list_results()
        entry = entries[0]
        assert "processing_id" in entry
        assert "document_type" in entry
        assert "status" in entry


# ---------------------------------------------------------------------------
# cleanup_old_results
# ---------------------------------------------------------------------------


class TestCleanup:

    def test_cleanup_removes_old(self, tmp_path):
        import os
        store = ResultStore(storage_dir=tmp_path, max_age_hours=1)
        path = store.save("proc_old", _sample_result())
        # Backdate file mtime by 2 hours so it's older than max_age
        old_time = time.time() - 7200
        os.utime(path, (old_time, old_time))
        deleted = store.cleanup_old_results()
        assert deleted >= 1
        assert store.get("proc_old") is None

    def test_cleanup_no_old_results(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path, max_age_hours=9999)
        store.save("proc_new", _sample_result())
        deleted = store.cleanup_old_results()
        assert deleted == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:

    def test_get_corrupted_json(self, tmp_path):
        store = ResultStore(storage_dir=tmp_path)
        path = store.save("proc_bad", _sample_result())
        # Corrupt the file
        path.write_text("{invalid json", encoding="utf-8")
        assert store.get("proc_bad") is None
