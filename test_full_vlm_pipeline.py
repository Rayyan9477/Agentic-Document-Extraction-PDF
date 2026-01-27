"""
Test complete VLM-first pipeline: Layout → Components → Schema → Extraction.

Tests all 4 VLM stages on first page of superbill1.pdf for speed.
"""

import json
import os
import sys
from pathlib import Path

# Fix: Remove AGENT environment variable if set
if "AGENT" in os.environ:
    del os.environ["AGENT"]

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.layout_agent import LayoutAgent
from src.agents.component_detector import ComponentDetectorAgent
from src.agents.schema_generator import SchemaGeneratorAgent
from src.agents.extractor import ExtractorAgent
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import create_initial_state
from src.preprocessing.pdf_processor import PDFProcessor


def main():
    """Test complete VLM-first pipeline on first page only."""
    
    logger = get_logger("test_full_vlm")
    settings = get_settings()
    
    print("\n" + "="*80)
    print("FULL VLM-FIRST PIPELINE TEST - superbill1.pdf (Page 1 Only)")
    print("="*80 + "\n")
    
    # Step 1: Load PDF (first page only)
    pdf_path = Path("superbill1.pdf")
    if not pdf_path.exists():
        print(f"[ERROR] {pdf_path} not found!")
        return
    
    print(f"[PDF] Loading first page of: {pdf_path}")
    processor = PDFProcessor()
    
    try:
        # Process full PDF but we'll only use first page
        result = processor.process(pdf_path)
        print(f"[OK] PDF loaded: {result.page_count} pages (using page 1 only)")
        print(f"  Processing time: {result.processing_time_ms}ms\n")
    except Exception as e:
        print(f"[ERROR] Failed to load PDF: {e}")
        return
    
    # Step 2: Create extraction state with ONLY first page
    print("[STATE] Creating extraction state (page 1 only)...")
    
    page_images = []
    first_page = result.pages[0]  # Only first page
    page_images.append({
        "page_number": first_page.page_number,
        "data_uri": first_page.data_uri,
        "width": first_page.width,
        "height": first_page.height,
        "dpi": first_page.dpi,
    })
    
    state = create_initial_state(
        pdf_path=pdf_path,
        pdf_hash=result.metadata.file_hash,
        page_images=page_images,
    )
    state["use_adaptive_extraction"] = True  # Force VLM-first mode
    print(f"[OK] State created with 1 page image\n")
    
    # Create VLM client
    client = LMStudioClient()
    
    # Check LM Studio health
    print("Checking LM Studio connection...")
    if not client.is_healthy():
        print("[ERROR] LM Studio server not responding!")
        return
    print(f"[OK] LM Studio connected: {settings.lm_studio.base_url}\n")
    
    # Step 3: Run LayoutAgent
    print("="*80)
    print("STAGE 1: VLM Layout Analysis")
    print("="*80 + "\n")
    
    layout_agent = LayoutAgent(client=client)
    
    try:
        print("[ANALYZE] Analyzing layout and visual marks...")
        state = layout_agent.process(state)
        print(f"[OK] Layout analysis completed!\n")
        
        # Display results
        layout = state.get("layout_analyses", [])[0]
        print(f"Layout Type: {layout.get('layout_type', 'unknown')}")
        print(f"Confidence: {layout.get('layout_confidence', 0.0):.2f}")
        print(f"Estimated Fields: {layout.get('estimated_field_count', 0)}")
        print(f"Regions: {len(layout.get('regions', []))}")
        print(f"Visual Marks: {len(layout.get('visual_marks', []))}")
        print()
    
    except Exception as e:
        print(f"[ERROR] Layout analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Run ComponentDetectorAgent
    print("="*80)
    print("STAGE 2: VLM Component Detection")
    print("="*80 + "\n")
    
    component_agent = ComponentDetectorAgent(client=client)
    
    try:
        print("[DETECT] Detecting components...")
        state = component_agent.process(state)
        print(f"[OK] Component detection completed!\n")
        
        # Display results
        comp_map = state.get("component_maps", [])[0]
        print(f"Tables: {len(comp_map.get('tables', []))}")
        print(f"Form Fields: {len(comp_map.get('forms', []))}")
        print(f"Key-Value Pairs: {len(comp_map.get('key_value_pairs', []))}")
        print(f"Special Elements: {len(comp_map.get('special_elements', []))}")
        print()
    
    except Exception as e:
        print(f"[ERROR] Component detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Run SchemaGeneratorAgent
    print("="*80)
    print("STAGE 3: VLM Schema Generation")
    print("="*80 + "\n")
    
    schema_agent = SchemaGeneratorAgent(client=client)
    
    try:
        print("[SCHEMA] Generating adaptive schema...")
        state = schema_agent.process(state)
        print(f"[OK] Schema generation completed!\n")
        
        # Display results
        schema = state.get("adaptive_schema")
        if schema:
            field_groups = schema.get("field_groups", [])
            fields = schema.get("fields", [])
            print(f"Schema Generated:")
            print(f"  Document Type: {schema.get('document_type', 'unknown')}")
            print(f"  Total Field Groups: {len(field_groups)}")
            print(f"  Total Fields: {len(fields)}")
            print(f"  Schema Confidence: {schema.get('schema_confidence', 0.0)}")
            print(f"  Recommended Confidence Threshold: {schema.get('recommended_confidence_threshold', 0.0)}")
            print(f"\nField Groups:")
            for group in field_groups:
                group_name = group.get("group_name", "unknown")
                field_names = group.get("field_names", [])
                print(f"  - {group_name}: {len(field_names)} fields")
                print(f"    Strategy: {group.get('extraction_strategy', 'unknown')}")
            print(f"\nSample Fields:")
            for field_def in fields[:5]:
                field_name = field_def.get("field_name", "unknown")
                print(f"  - {field_name}:")
                print(f"    Type: {field_def.get('field_type', 'unknown')}")
                print(f"    Required: {field_def.get('required', False)}")
                if field_def.get('location_hint'):
                    print(f"    Location: {field_def['location_hint'][:60]}...")
            print()
        else:
            print("[WARNING] No schema generated!")
            print()
    
    except Exception as e:
        print(f"[ERROR] Schema generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Run ExtractorAgent (adaptive mode)
    print("="*80)
    print("STAGE 4: VLM Adaptive Extraction")
    print("="*80 + "\n")
    
    extractor_agent = ExtractorAgent(client=client)
    
    try:
        print("[EXTRACT] Extracting data using adaptive schema...")
        state = extractor_agent.process(state)
        print(f"[OK] Extraction completed!\n")
        
        # Display results
        extracted = state.get("merged_extraction", {})
        if not extracted:
            extracted = state.get("extracted_data", {})
        
        print(f"Extracted Data Summary:")
        print(f"  Total top-level keys: {len(extracted)}")
        for key, value in list(extracted.items())[:10]:  # Show first 10
            if isinstance(value, dict):
                print(f"  - {key}: {len(value)} sub-items")
            elif isinstance(value, list):
                print(f"  - {key}: {len(value)} items")
            else:
                val_str = str(value)[:50]
                print(f"  - {key}: {val_str}")
        
        if len(extracted) > 10:
            print(f"  ... and {len(extracted) - 10} more fields")
        print()
    
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save complete results
    print("="*80)
    print("RESULTS SAVED")
    print("="*80 + "\n")
    
    output_path = Path("test_full_vlm_results.json")
    with open(output_path, "w") as f:
        # Create output with all stages
        output = {
            "pdf_path": str(pdf_path),
            "page_count": 1,
            "total_vlm_calls": state.get("total_vlm_calls", 0),
            "total_processing_time_ms": state.get("total_processing_time_ms", 0),
            "layout_analysis": state.get("layout_analyses", [])[0] if state.get("layout_analyses") else {},
            "component_map": state.get("component_maps", [])[0] if state.get("component_maps") else {},
            "adaptive_schema": state.get("adaptive_schema", {}),
            "merged_extraction": state.get("merged_extraction", {}),
            "field_metadata": state.get("field_metadata", {}),
            "status": state.get("status", "unknown"),
        }
        json.dump(output, f, indent=2)
    
    print(f"[OK] Results saved to: {output_path}")
    print(f"\nPipeline Summary:")
    print(f"  Total VLM calls: {state.get('total_vlm_calls', 0)}")
    print(f"  Total time: {state.get('total_processing_time_ms', 0)/1000:.1f}s")
    print(f"  Status: {state.get('status', 'unknown')}")
    
    print("\n" + "="*80)
    print("[SUCCESS] FULL VLM-FIRST PIPELINE TEST COMPLETED!")
    print("="*80 + "\n")
    
    # Display extracted data sample
    print("\n" + "="*80)
    print("EXTRACTED DATA SAMPLE")
    print("="*80 + "\n")
    
    extracted = state.get("merged_extraction", {})
    if not extracted:
        extracted = state.get("extracted_data", {})
    print(json.dumps(extracted, indent=2)[:2000])
    print("\n... (see test_full_vlm_results.json for complete output)")


if __name__ == "__main__":
    main()
