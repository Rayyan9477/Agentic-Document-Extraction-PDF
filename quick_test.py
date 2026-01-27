"""
Quick test of VLM-first agents without full pipeline initialization.
"""

import json
import sys
from pathlib import Path

# Set environment variable to avoid config issues
import os
if "AGENT" in os.environ:
    del os.environ["AGENT"]  # Remove conflicting AGENT env var
os.environ.setdefault("API_ENV", "development")
os.environ.setdefault("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
os.environ.setdefault("LM_STUDIO_MODEL", "qwen/qwen3-vl-8b")

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.pdf_processor import PDFProcessor
from src.config import get_logger

def main():
    """Quick test of PDF processing only."""
    
    logger = get_logger("quick_test")
    
    print("\n" + "="*80)
    print("QUICK TEST - PDF Processing Only")
    print("="*80 + "\n")
    
    # Load PDF
    pdf_path = Path("superbill1.pdf")
    if not pdf_path.exists():
        print(f"[ERROR]: {pdf_path} not found!")
        return
    
    print(f"[*] Loading PDF: {pdf_path}")
    processor = PDFProcessor()
    
    try:
        result = processor.process(pdf_path)
        print(f"[OK] PDF loaded successfully!")
        print(f"  Pages: {result.page_count}")
        print(f"  File size: {result.metadata.file_size_bytes / 1024:.1f} KB")
        print(f"  Processing time: {result.processing_time_ms}ms")
        print(f"  Hash: {result.metadata.file_hash[:16]}...")
        
        # Show page details
        for page in result.pages:
            print(f"\n  Page {page.page_number}:")
            print(f"    Size: {page.width}x{page.height} pixels @ {page.dpi} DPI")
            print(f"    Orientation: {page.orientation.value}")
            print(f"    Image size: {page.size_kb:.1f} KB")
            print(f"    Has text: {page.has_text}")
            print(f"    Has images: {page.has_images}")
            print(f"    Rotation: {page.rotation}°")
        
        print(f"\n[OK] PDF processing test completed!")
        print(f"\nTo test VLM agents, ensure LM Studio is running at:")
        print(f"  http://localhost:1234/v1")
        print(f"  Model loaded: qwen/qwen3-vl-8b (or compatible VLM)")
        
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
