"""
Batch processing manager for memory-efficient document processing.

Provides intelligent batching, streaming, and memory management
for processing large PDF documents without exhausting system memory.
"""

import gc
import os
import sys
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from src.config import get_logger, get_settings
from src.preprocessing.image_enhancer import EnhancementResult, ImageEnhancer
from src.preprocessing.pdf_processor import (
    PageImage,
    PDFMetadata,
    PDFProcessor,
)


logger = get_logger(__name__)

T = TypeVar("T")


class BatchStatus(str, Enum):
    """Batch processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class BatchProgress:
    """
    Progress information for batch processing.

    Attributes:
        total_pages: Total pages to process.
        processed_pages: Pages processed so far.
        current_batch: Current batch number.
        total_batches: Total number of batches.
        start_time: Processing start time.
        estimated_completion: Estimated completion time.
        memory_usage_mb: Current memory usage in MB.
        errors: List of error messages.
    """

    total_pages: int
    processed_pages: int = 0
    current_batch: int = 0
    total_batches: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    estimated_completion: datetime | None = None
    memory_usage_mb: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.processed_pages / self.total_pages) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now(UTC) - self.start_time).total_seconds()

    @property
    def pages_per_second(self) -> float:
        """Get processing rate."""
        if self.elapsed_seconds == 0:
            return 0.0
        return self.processed_pages / self.elapsed_seconds

    def estimate_remaining_time(self) -> float | None:
        """Estimate remaining time in seconds."""
        if self.pages_per_second == 0:
            return None
        remaining_pages = self.total_pages - self.processed_pages
        return remaining_pages / self.pages_per_second

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "pages_per_second": self.pages_per_second,
            "estimated_remaining_seconds": self.estimate_remaining_time(),
            "memory_usage_mb": self.memory_usage_mb,
            "errors_count": len(self.errors),
        }


@dataclass(slots=True)
class BatchResult:
    """
    Result of batch processing operation.

    Attributes:
        metadata: PDF document metadata.
        pages: List of processed and enhanced page images.
        enhancement_results: Enhancement results for each page.
        progress: Final progress information.
        status: Final batch status.
        processing_time_ms: Total processing time in milliseconds.
        peak_memory_mb: Peak memory usage during processing.
    """

    metadata: PDFMetadata
    pages: list[PageImage] = field(default_factory=list)
    enhancement_results: list[EnhancementResult] = field(default_factory=list)
    progress: BatchProgress | None = None
    status: BatchStatus = BatchStatus.COMPLETED
    processing_time_ms: int = 0
    peak_memory_mb: float = 0.0

    @property
    def page_count(self) -> int:
        """Get number of processed pages."""
        return len(self.pages)

    def get_enhanced_pages(self) -> list[PageImage]:
        """Get list of enhanced page images."""
        if self.enhancement_results:
            return [result.page_image for result in self.enhancement_results]
        return self.pages

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "page_count": self.page_count,
            "status": self.status.value,
            "processing_time_ms": self.processing_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "progress": self.progress.to_dict() if self.progress else None,
        }


class MemoryMonitor:
    """
    Monitor and manage memory usage during processing.

    Provides memory tracking and garbage collection triggers
    to prevent OOM conditions during large document processing.
    """

    def __init__(self, threshold_mb: float = 1000.0) -> None:
        """
        Initialize memory monitor.

        Args:
            threshold_mb: Memory threshold in MB to trigger cleanup.
        """
        self._threshold_bytes = threshold_mb * 1024 * 1024
        self._peak_memory = 0.0
        self._lock = threading.Lock()

    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except ImportError:
            # Fallback to sys.getsizeof if psutil not available
            return sys.getsizeof(0) / (1024 * 1024)

    def update_peak(self) -> float:
        """Update and return peak memory usage."""
        current = self.get_current_memory_mb()
        with self._lock:
            self._peak_memory = max(self._peak_memory, current)
        return self._peak_memory

    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self._peak_memory

    def check_and_cleanup(self) -> bool:
        """
        Check memory usage and trigger cleanup if needed.

        Returns:
            True if cleanup was triggered, False otherwise.
        """
        current_bytes = self.get_current_memory_mb() * 1024 * 1024
        if current_bytes > self._threshold_bytes:
            logger.warning(
                "memory_threshold_exceeded",
                current_mb=current_bytes / (1024 * 1024),
                threshold_mb=self._threshold_bytes / (1024 * 1024),
            )
            gc.collect()
            return True
        return False

    def force_cleanup(self) -> None:
        """Force garbage collection."""
        gc.collect()


class BatchManager:
    """
    Memory-efficient batch processing manager for PDF documents.

    Provides intelligent batching, parallel processing, and memory
    management for processing large documents efficiently.

    Example:
        manager = BatchManager(batch_size=5)
        result = manager.process_document(Path("large_document.pdf"))

        # Or use streaming for very large documents
        for batch in manager.stream_batches(Path("huge_document.pdf")):
            for page in batch:
                # Process each page
                pass
    """

    def __init__(
        self,
        batch_size: int | None = None,
        max_workers: int | None = None,
        enable_enhancement: bool | None = None,
        memory_threshold_mb: float = 1000.0,
    ) -> None:
        """
        Initialize the batch manager.

        Args:
            batch_size: Number of pages per batch. Defaults to settings.
            max_workers: Maximum parallel workers. Defaults to CPU count / 2.
            enable_enhancement: Enable image enhancement. Defaults to settings.
            memory_threshold_mb: Memory threshold for cleanup in MB.
        """
        settings = get_settings()

        self._batch_size = batch_size or settings.extraction.batch_size
        self._max_workers = max_workers or max(1, (os.cpu_count() or 4) // 2)
        self._enable_enhancement = (
            enable_enhancement
            if enable_enhancement is not None
            else settings.pdf.enable_enhancement
        )

        self._pdf_processor = PDFProcessor()
        self._image_enhancer = ImageEnhancer() if self._enable_enhancement else None
        self._memory_monitor = MemoryMonitor(threshold_mb=memory_threshold_mb)

        self._cancel_requested = False
        self._lock = threading.Lock()

        logger.debug(
            "batch_manager_initialized",
            batch_size=self._batch_size,
            max_workers=self._max_workers,
            enable_enhancement=self._enable_enhancement,
        )

    def process_document(
        self,
        file_path: Path,
        page_range: tuple[int, int] | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ) -> BatchResult:
        """
        Process a complete PDF document with batching and memory management.

        Args:
            file_path: Path to the PDF file.
            page_range: Optional (start, end) page range (1-indexed).
            progress_callback: Optional callback for progress updates.

        Returns:
            BatchResult containing all processed pages.

        Raises:
            PDFProcessingError: If processing fails.
        """
        import time

        start_time = time.perf_counter()
        self._cancel_requested = False

        # Initial validation and metadata extraction
        self._pdf_processor.validate(file_path)
        metadata = self._pdf_processor.extract_metadata(file_path)

        # Determine page range
        if page_range:
            start_page = max(1, page_range[0])
            end_page = min(metadata.page_count, page_range[1])
        else:
            start_page = 1
            end_page = metadata.page_count

        total_pages = end_page - start_page + 1
        total_batches = (total_pages + self._batch_size - 1) // self._batch_size

        # Initialize progress tracking
        progress = BatchProgress(
            total_pages=total_pages,
            total_batches=total_batches,
        )

        all_pages: list[PageImage] = []
        all_enhancements: list[EnhancementResult] = []

        try:
            # Process in batches
            for batch_idx in range(total_batches):
                if self._cancel_requested:
                    logger.info("batch_processing_cancelled")
                    return BatchResult(
                        metadata=metadata,
                        pages=all_pages,
                        enhancement_results=all_enhancements,
                        progress=progress,
                        status=BatchStatus.CANCELLED,
                        processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                        peak_memory_mb=self._memory_monitor.peak_memory_mb,
                    )

                # Calculate batch page range
                batch_start = start_page + (batch_idx * self._batch_size)
                batch_end = min(batch_start + self._batch_size - 1, end_page)

                progress.current_batch = batch_idx + 1
                progress.memory_usage_mb = self._memory_monitor.get_current_memory_mb()

                logger.debug(
                    "processing_batch",
                    batch=batch_idx + 1,
                    total_batches=total_batches,
                    pages=f"{batch_start}-{batch_end}",
                )

                # Process batch
                batch_result = self._pdf_processor.process(
                    file_path,
                    page_range=(batch_start, batch_end),
                )

                # Apply enhancements if enabled
                if self._enable_enhancement and self._image_enhancer:
                    enhanced_results = self._image_enhancer.enhance_batch(batch_result.pages)
                    all_enhancements.extend(enhanced_results)
                    all_pages.extend([r.page_image for r in enhanced_results])
                else:
                    all_pages.extend(batch_result.pages)

                # Update progress
                progress.processed_pages += len(batch_result.pages)
                self._memory_monitor.update_peak()

                # Check memory and cleanup if needed
                self._memory_monitor.check_and_cleanup()

                # Invoke callback
                if progress_callback:
                    progress_callback(progress)

                # Clean up batch temp directory
                self._pdf_processor.cleanup(batch_result)

            # Calculate final processing time
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            result = BatchResult(
                metadata=metadata,
                pages=all_pages,
                enhancement_results=all_enhancements,
                progress=progress,
                status=BatchStatus.COMPLETED,
                processing_time_ms=processing_time_ms,
                peak_memory_mb=self._memory_monitor.peak_memory_mb,
            )

            logger.info(
                "batch_processing_complete",
                total_pages=total_pages,
                processing_time_ms=processing_time_ms,
                peak_memory_mb=self._memory_monitor.peak_memory_mb,
            )

            return result

        except Exception as e:
            progress.errors.append(str(e))
            logger.error(
                "batch_processing_failed",
                error=str(e),
                processed_pages=progress.processed_pages,
            )

            return BatchResult(
                metadata=metadata,
                pages=all_pages,
                enhancement_results=all_enhancements,
                progress=progress,
                status=BatchStatus.FAILED,
                processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                peak_memory_mb=self._memory_monitor.peak_memory_mb,
            )

    def stream_batches(
        self,
        file_path: Path,
        page_range: tuple[int, int] | None = None,
    ) -> Generator[list[PageImage], None, None]:
        """
        Stream document pages in batches for memory efficiency.

        Yields batches of pages one at a time, allowing processing
        without loading the entire document into memory.

        Args:
            file_path: Path to the PDF file.
            page_range: Optional (start, end) page range.

        Yields:
            Lists of PageImage objects, one batch at a time.
        """
        self._cancel_requested = False

        # Validate and get metadata
        self._pdf_processor.validate(file_path)
        metadata = self._pdf_processor.extract_metadata(file_path)

        # Determine page range
        if page_range:
            start_page = max(1, page_range[0])
            end_page = min(metadata.page_count, page_range[1])
        else:
            start_page = 1
            end_page = metadata.page_count

        total_pages = end_page - start_page + 1
        total_batches = (total_pages + self._batch_size - 1) // self._batch_size

        for batch_idx in range(total_batches):
            if self._cancel_requested:
                logger.info("batch_streaming_cancelled")
                return

            # Calculate batch page range
            batch_start = start_page + (batch_idx * self._batch_size)
            batch_end = min(batch_start + self._batch_size - 1, end_page)

            # Process batch
            batch_result = self._pdf_processor.process(
                file_path,
                page_range=(batch_start, batch_end),
            )

            # Apply enhancements if enabled
            if self._enable_enhancement and self._image_enhancer:
                enhanced_results = self._image_enhancer.enhance_batch(batch_result.pages)
                batch_pages = [r.page_image for r in enhanced_results]
            else:
                batch_pages = batch_result.pages

            # Yield batch
            yield batch_pages

            # Cleanup
            self._pdf_processor.cleanup(batch_result)
            self._memory_monitor.check_and_cleanup()

    def stream_pages(
        self,
        file_path: Path,
        page_range: tuple[int, int] | None = None,
    ) -> Generator[tuple[PDFMetadata, PageImage], None, None]:
        """
        Stream individual pages for maximum memory efficiency.

        Yields pages one at a time, suitable for very large documents
        or memory-constrained environments.

        Args:
            file_path: Path to the PDF file.
            page_range: Optional (start, end) page range.

        Yields:
            Tuples of (metadata, page_image) for each page.
        """
        for metadata, page in self._pdf_processor.process_streaming(file_path, page_range):
            if self._cancel_requested:
                return

            # Enhance if enabled
            if self._enable_enhancement and self._image_enhancer:
                result = self._image_enhancer.enhance(page)
                yield metadata, result.page_image
            else:
                yield metadata, page

            # Check memory
            self._memory_monitor.check_and_cleanup()

    def process_parallel(
        self,
        file_paths: list[Path],
        progress_callback: Callable[[str, BatchProgress], None] | None = None,
    ) -> dict[str, BatchResult]:
        """
        Process multiple documents in parallel.

        Args:
            file_paths: List of PDF file paths.
            progress_callback: Optional callback receiving (file_name, progress).

        Returns:
            Dictionary mapping file names to BatchResults.
        """
        results: dict[str, BatchResult] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all documents
            future_to_path = {
                executor.submit(self._process_single, path): path for path in file_paths
            }

            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[path.name] = result

                    if progress_callback:
                        progress_callback(path.name, result.progress or BatchProgress(0))

                except Exception as e:
                    logger.error(
                        "parallel_document_failed",
                        file_path=str(path),
                        error=str(e),
                    )
                    # Create failed result
                    metadata = self._pdf_processor.extract_metadata(path)
                    results[path.name] = BatchResult(
                        metadata=metadata,
                        status=BatchStatus.FAILED,
                        progress=BatchProgress(total_pages=0, errors=[str(e)]),
                    )

        return results

    def _process_single(self, file_path: Path) -> BatchResult:
        """Process a single document (for parallel processing)."""
        return self.process_document(file_path)

    def cancel(self) -> None:
        """Request cancellation of current processing."""
        with self._lock:
            self._cancel_requested = True
        logger.info("batch_cancellation_requested")

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        return {
            "current_mb": self._memory_monitor.get_current_memory_mb(),
            "peak_mb": self._memory_monitor.peak_memory_mb,
        }

    def estimate_memory_requirement(self, file_path: Path) -> dict[str, Any]:
        """
        Estimate memory requirements for processing a document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary with memory estimates.
        """
        self._pdf_processor.validate(file_path)
        metadata = self._pdf_processor.extract_metadata(file_path)

        # Rough estimates based on typical document characteristics
        # At 300 DPI, a letter-size page is roughly 3300x2550 pixels
        # RGB = 3 bytes per pixel = ~25 MB per page raw
        # With compression, typically 1-5 MB per page
        estimated_per_page_mb = 5.0

        # Enhancement adds temporary memory usage
        enhancement_overhead = 2.0 if self._enable_enhancement else 1.0

        # Batch size affects peak memory
        peak_estimate = (
            estimated_per_page_mb
            * min(self._batch_size, metadata.page_count)
            * enhancement_overhead
        )

        return {
            "page_count": metadata.page_count,
            "file_size_mb": metadata.file_size_bytes / (1024 * 1024),
            "estimated_per_page_mb": estimated_per_page_mb,
            "batch_size": self._batch_size,
            "estimated_peak_mb": peak_estimate,
            "recommended_batch_size": max(
                1, int(1000 / (estimated_per_page_mb * enhancement_overhead))
            ),
        }
