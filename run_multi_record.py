"""
Run multi-record extraction on superbill1.pdf.

Extracts individual patient records from each page, detects duplicates,
and exports to Excel + Markdown + JSON.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Bypass corporate proxy for localhost LM Studio
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Set logging to INFO and silence noisy loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for noisy in ("httpcore", "httpx", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from src.client.lm_client import LMStudioClient
from src.pipeline.runner import PipelineRunner
from src.export.consolidated_export import export_excel, export_markdown, export_json


def main():
    pdf_path = "superbill1.pdf"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("MULTI-RECORD EXTRACTION")
    print("=" * 70)
    print(f"PDF: {pdf_path}")
    print()

    start = time.time()

    # Use 127.0.0.1 instead of localhost to bypass corporate proxy
    # Explicitly set VLM model (LM Studio uses full name)
    client = LMStudioClient(
        base_url="http://127.0.0.1:1234/v1",
        model="qwen/qwen3-vl-8b",
    )

    # Create pipeline runner (reuses existing PDF loading infrastructure)
    runner = PipelineRunner(
        client=client,
        enable_checkpointing=False,
        dpi=200,
    )

    # Run multi-record extraction on ALL pages
    result = runner.extract_multi_record(
        pdf_path=pdf_path,
    )

    elapsed = time.time() - start

    # Print summary
    print()
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Document Type : {result.document_type}")
    print(f"Entity Type   : {result.entity_type}")
    print(f"Total Pages   : {result.total_pages}")
    print(f"Total Records : {result.total_records}")
    print(f"VLM Calls     : {result.total_vlm_calls}")
    print(f"Time          : {elapsed:.1f}s")
    print(f"Avg/Record    : {elapsed / max(result.total_records, 1):.1f}s")
    print()

    # Print per-record summary
    print("RECORDS FOUND:")
    print("-" * 70)
    for rec in result.records:
        print(
            f"  [{rec.record_id:3d}] Page {rec.page_number:2d} | "
            f"{rec.primary_identifier:30s} | "
            f"{rec.confidence:.0%} confidence | "
            f"{len(rec.fields)} fields"
        )
    print()

    # Export
    excel_path = output_dir / "superbill1_multi_record.xlsx"
    md_path = output_dir / "superbill1_multi_record.md"
    json_path = output_dir / "superbill1_multi_record.json"

    print("EXPORTING...")
    export_excel(result, excel_path)
    print(f"  Excel    : {excel_path}")

    export_markdown(result, md_path)
    print(f"  Markdown : {md_path}")

    export_json(result, json_path)
    print(f"  JSON     : {json_path}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
