"""
Result storage for document extraction results.

Provides file-based storage for extraction results that can be
retrieved later for preview and export operations.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


class ResultStore:
    """
    File-based storage for extraction results.

    Stores results as JSON files indexed by processing_id,
    enabling later retrieval for preview and export operations.
    """

    def __init__(
        self,
        storage_dir: str | Path = "./data/results",
        max_age_hours: int = 24 * 7,  # 1 week default retention
    ) -> None:
        """
        Initialize the result store.

        Args:
            storage_dir: Directory to store result files.
            max_age_hours: Maximum age for results before cleanup.
        """
        self._storage_dir = Path(storage_dir)
        self._max_age_hours = max_age_hours
        self._lock = threading.Lock()
        self._logger = logger

        # Ensure storage directory exists
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_result_path(self, processing_id: str) -> Path:
        """Get the file path for a processing ID."""
        # Use first 2 chars as subdirectory for better file distribution
        subdir = processing_id[:2] if len(processing_id) >= 2 else "00"
        return self._storage_dir / subdir / f"{processing_id}.json"

    def save(
        self,
        processing_id: str,
        result: dict[str, Any],
    ) -> Path:
        """
        Save extraction result.

        Args:
            processing_id: Unique processing identifier.
            result: Extraction result dictionary.

        Returns:
            Path to saved result file.
        """
        result_path = self._get_result_path(processing_id)

        # Ensure parent directory exists
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Add storage metadata
        result_with_meta = {
            **result,
            "_storage_metadata": {
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "processing_id": processing_id,
            },
        }

        with self._lock:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_with_meta, f, indent=2, default=str)

        self._logger.info(
            "result_stored",
            processing_id=processing_id,
            path=str(result_path),
        )

        return result_path

    def get(self, processing_id: str) -> dict[str, Any] | None:
        """
        Retrieve extraction result.

        Args:
            processing_id: Unique processing identifier.

        Returns:
            Result dictionary or None if not found.
        """
        result_path = self._get_result_path(processing_id)

        if not result_path.exists():
            self._logger.debug(
                "result_not_found",
                processing_id=processing_id,
            )
            return None

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            self._logger.debug(
                "result_retrieved",
                processing_id=processing_id,
            )

            return result

        except json.JSONDecodeError as e:
            self._logger.error(
                "result_parse_error",
                processing_id=processing_id,
                error=str(e),
            )
            return None

        except Exception as e:
            self._logger.error(
                "result_read_error",
                processing_id=processing_id,
                error=str(e),
            )
            return None

    def exists(self, processing_id: str) -> bool:
        """Check if a result exists for the processing ID."""
        return self._get_result_path(processing_id).exists()

    def delete(self, processing_id: str) -> bool:
        """
        Delete a stored result.

        Args:
            processing_id: Unique processing identifier.

        Returns:
            True if deleted, False if not found.
        """
        result_path = self._get_result_path(processing_id)

        if not result_path.exists():
            return False

        try:
            with self._lock:
                result_path.unlink()

            self._logger.info(
                "result_deleted",
                processing_id=processing_id,
            )

            return True

        except Exception as e:
            self._logger.error(
                "result_delete_error",
                processing_id=processing_id,
                error=str(e),
            )
            return False

    def list_results(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List stored results with pagination.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of result metadata dictionaries.
        """
        results = []

        try:
            # Gather all result files
            all_files = list(self._storage_dir.glob("**/*.json"))

            # Sort by modification time (newest first)
            all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Apply pagination
            for file_path in all_files[offset:offset + limit]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    results.append({
                        "processing_id": data.get("processing_id", file_path.stem),
                        "document_type": data.get("document_type", "unknown"),
                        "status": data.get("status", "unknown"),
                        "stored_at": data.get("_storage_metadata", {}).get("stored_at"),
                        "overall_confidence": data.get("overall_confidence", 0.0),
                    })

                except Exception:
                    # Skip invalid files
                    continue

            return results

        except Exception as e:
            self._logger.error(
                "list_results_error",
                error=str(e),
            )
            return []

    def cleanup_old_results(self) -> int:
        """
        Clean up results older than max_age_hours.

        Returns:
            Number of results deleted.
        """
        from datetime import timedelta

        deleted_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._max_age_hours)

        try:
            for file_path in self._storage_dir.glob("**/*.json"):
                try:
                    # Check file modification time
                    mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime,
                        tz=timezone.utc,
                    )

                    if mtime < cutoff:
                        file_path.unlink()
                        deleted_count += 1

                except Exception as e:
                    self._logger.warning(
                        "cleanup_file_error",
                        path=str(file_path),
                        error=str(e),
                    )

            if deleted_count > 0:
                self._logger.info(
                    "results_cleanup_complete",
                    deleted_count=deleted_count,
                    max_age_hours=self._max_age_hours,
                )

            return deleted_count

        except Exception as e:
            self._logger.error(
                "cleanup_error",
                error=str(e),
            )
            return 0


# Module-level singleton
_result_store: ResultStore | None = None
_store_lock = threading.Lock()


def get_result_store(
    storage_dir: str | Path | None = None,
) -> ResultStore:
    """
    Get or create the result store singleton.

    Args:
        storage_dir: Optional storage directory override.

    Returns:
        ResultStore instance.
    """
    global _result_store

    with _store_lock:
        if _result_store is None:
            _result_store = ResultStore(
                storage_dir=storage_dir or "./data/results",
            )

    return _result_store


def save_result(processing_id: str, result: dict[str, Any]) -> Path:
    """
    Convenience function to save a result.

    Args:
        processing_id: Unique processing identifier.
        result: Extraction result dictionary.

    Returns:
        Path to saved result file.
    """
    return get_result_store().save(processing_id, result)


def get_result(processing_id: str) -> dict[str, Any] | None:
    """
    Convenience function to retrieve a result.

    Args:
        processing_id: Unique processing identifier.

    Returns:
        Result dictionary or None if not found.
    """
    return get_result_store().get(processing_id)
