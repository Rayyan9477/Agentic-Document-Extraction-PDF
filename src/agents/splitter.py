"""
Document Splitter Agent for multi-document PDF detection and splitting.

Automatically detects document boundaries in multi-document PDFs
(e.g., 200-page batch of mixed EOBs, claims, intake forms) and
splits them into sub-documents for individual extraction.

Inspired by: LandingAI Parse→Split→Extract pipeline, DocsRay hierarchical TOC.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from src.agents.base import BaseAgent
from src.client.lm_client import LMStudioClient
from src.config import get_logger
from src.pipeline.state import ExtractionState, update_state


logger = get_logger(__name__)

# Batch size for page classification (balance between context and accuracy)
CLASSIFICATION_BATCH_SIZE = 5


@dataclass(slots=True)
class DocumentSegment:
    """A detected document segment within a multi-document PDF."""

    start_page: int
    end_page: int
    document_type: str
    confidence: float
    page_count: int = 0
    title: str = ""

    def __post_init__(self) -> None:
        self.page_count = self.end_page - self.start_page + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_page": self.start_page,
            "end_page": self.end_page,
            "document_type": self.document_type,
            "confidence": self.confidence,
            "page_count": self.page_count,
            "title": self.title,
        }


BOUNDARY_DETECTION_SYSTEM_PROMPT = """You are a document boundary detection specialist.

Given a sequence of document page images, determine if each page is:
1. A NEW document (first page of a different document)
2. A CONTINUATION of the previous document

Look for these boundary indicators:
- Page headers/logos that change between documents
- "Page 1 of N" indicators resetting
- Different form types (e.g., CMS-1500 vs UB-04 vs EOB)
- Fax cover sheets or separator pages
- Blank pages between documents
- Different patient names/dates at the top

Respond in JSON:
{
  "pages": [
    {
      "page_number": 1,
      "is_new_document": true,
      "document_type": "CMS-1500",
      "confidence": 0.95,
      "reason": "First page, shows CMS-1500 form header"
    },
    ...
  ]
}
"""


class SplitterAgent(BaseAgent):
    """
    VLM-powered document boundary detection agent.

    Strategy:
    1. Batch pages in groups of 3-5
    2. Ask VLM to classify each page as new document or continuation
    3. Group consecutive continuation pages into DocumentSegments
    4. Handle edge cases: fax cover sheets, blank pages, appendices
    """

    def __init__(self, client: LMStudioClient | None = None) -> None:
        super().__init__(name="splitter", client=client)

    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Detect document boundaries and split multi-document PDFs.

        Adds to state:
            document_segments: list of segment dicts
            is_multi_document: bool
            active_segment_index: int (0 = first segment)
        """
        page_images = state.get("page_images", [])

        if not page_images:
            return update_state(state, {
                "document_segments": [],
                "is_multi_document": False,
                "active_segment_index": 0,
            })

        # Single-page documents are never multi-document
        if len(page_images) <= 1:
            segment = DocumentSegment(
                start_page=1,
                end_page=1,
                document_type="unknown",
                confidence=1.0,
            )
            return update_state(state, {
                "document_segments": [segment.to_dict()],
                "is_multi_document": False,
                "active_segment_index": 0,
            })

        self._logger.info(
            "splitting_start",
            total_pages=len(page_images),
        )

        # Classify boundaries in batches
        page_classifications = self._classify_all_pages(page_images)

        # Build segments from classifications
        segments = self._build_segments(page_classifications)

        is_multi = len(segments) > 1

        self._logger.info(
            "splitting_complete",
            total_pages=len(page_images),
            segments=len(segments),
            is_multi_document=is_multi,
        )

        return update_state(state, {
            "document_segments": [s.to_dict() for s in segments],
            "is_multi_document": is_multi,
            "active_segment_index": 0,
        })

    def _classify_all_pages(
        self, page_images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Classify all pages as new document or continuation."""
        all_classifications: list[dict[str, Any]] = []

        # Process in batches
        for batch_start in range(0, len(page_images), CLASSIFICATION_BATCH_SIZE):
            batch = page_images[batch_start : batch_start + CLASSIFICATION_BATCH_SIZE]
            batch_results = self._classify_batch(batch, batch_start)
            all_classifications.extend(batch_results)

        return all_classifications

    def _classify_batch(
        self,
        batch: list[dict[str, Any]],
        offset: int,
    ) -> list[dict[str, Any]]:
        """
        Classify a batch of pages using VLM.

        Falls back to heuristics if VLM fails.
        """
        if not batch:
            return []

        # Use the first page's image for the VLM request
        # For multi-page batches, we describe each page's position
        prompt_parts = [
            f"Analyze these {len(batch)} consecutive pages from a document batch.",
            f"Pages {offset + 1} through {offset + len(batch)}.",
            "For each page, determine if it starts a NEW document or CONTINUES the previous one.",
        ]

        try:
            # Send first page of batch (VLM sees one image at a time)
            first_page = batch[0]
            image_data = first_page.get("data_uri", first_page.get("base64_encoded", ""))

            response = self.send_vision_request_with_json(
                image_data=image_data,
                prompt="\n".join(prompt_parts),
                system_prompt=BOUNDARY_DETECTION_SYSTEM_PROMPT,
                max_tokens=2048,
                temperature=0.1,
            )

            pages_data = response.get("pages", [])

            # Map response to our format
            results: list[dict[str, Any]] = []
            for i, page in enumerate(batch):
                page_num = offset + i + 1

                if i < len(pages_data):
                    classification = pages_data[i]
                else:
                    # Default: continuation
                    classification = {
                        "is_new_document": False,
                        "document_type": "unknown",
                        "confidence": 0.5,
                    }

                results.append({
                    "page_number": page_num,
                    "is_new_document": classification.get("is_new_document", i == 0 and offset == 0),
                    "document_type": classification.get("document_type", "unknown"),
                    "confidence": classification.get("confidence", 0.5),
                    "reason": classification.get("reason", ""),
                })

            return results

        except Exception as e:
            self._logger.warning(
                "batch_classification_failed",
                offset=offset,
                batch_size=len(batch),
                error=str(e),
            )
            # Fallback: first page of entire document is new, rest are continuation
            return [
                {
                    "page_number": offset + i + 1,
                    "is_new_document": (offset == 0 and i == 0),
                    "document_type": "unknown",
                    "confidence": 0.3,
                    "reason": "VLM classification failed, using fallback",
                }
                for i in range(len(batch))
            ]

    def _build_segments(
        self, classifications: list[dict[str, Any]],
    ) -> list[DocumentSegment]:
        """Build document segments from page classifications."""
        if not classifications:
            return []

        segments: list[DocumentSegment] = []
        current_start = classifications[0]["page_number"]
        current_type = classifications[0].get("document_type", "unknown")
        current_confidence_sum = classifications[0].get("confidence", 0.5)
        current_count = 1

        for i in range(1, len(classifications)):
            page = classifications[i]

            if page.get("is_new_document", False):
                # Close current segment
                segments.append(DocumentSegment(
                    start_page=current_start,
                    end_page=classifications[i - 1]["page_number"],
                    document_type=current_type,
                    confidence=current_confidence_sum / current_count,
                ))

                # Start new segment
                current_start = page["page_number"]
                current_type = page.get("document_type", "unknown")
                current_confidence_sum = page.get("confidence", 0.5)
                current_count = 1
            else:
                current_confidence_sum += page.get("confidence", 0.5)
                current_count += 1

        # Close final segment
        segments.append(DocumentSegment(
            start_page=current_start,
            end_page=classifications[-1]["page_number"],
            document_type=current_type,
            confidence=current_confidence_sum / current_count,
        ))

        return segments

    def get_segment_pages(
        self,
        page_images: list[dict[str, Any]],
        segment: DocumentSegment | dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Get page images for a specific document segment.

        Args:
            page_images: All page images.
            segment: DocumentSegment or dict with start_page/end_page.

        Returns:
            List of page images within the segment.
        """
        if isinstance(segment, dict):
            start = segment["start_page"]
            end = segment["end_page"]
        else:
            start = segment.start_page
            end = segment.end_page

        return [
            p for p in page_images
            if start <= p.get("page_number", 0) <= end
        ]
