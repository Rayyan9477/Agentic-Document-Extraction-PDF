"""
Test VLM-first pipeline on superbill1.pdf sample.

Tests new LayoutAgent and ComponentDetectorAgent on real document.
"""

import json
import os
import sys
from pathlib import Path

# Fix: Remove AGENT environment variable if set (conflicts with pydantic Settings)
if "AGENT" in os.environ:
    del os.environ["AGENT"]

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.layout_agent import LayoutAgent
from src.agents.component_detector import ComponentDetectorAgent
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import create_initial_state
from src.preprocessing.pdf_processor import PDFProcessor


def main():
    """Test VLM-first pipeline on superbill1.pdf."""
    
    logger = get_logger("test_vlm_first")
    settings = get_settings()
    
    print("\n" + "="*80)
    print("VLM-FIRST PIPELINE TEST - superbill1.pdf")
    print("="*80 + "\n")
    
    # Step 1: Load PDF
    pdf_path = Path("superbill1.pdf")
    if not pdf_path.exists():
        print(f"[ERROR] {pdf_path} not found!")
        return
    
    print(f"[PDF] Loading PDF: {pdf_path}")
    processor = PDFProcessor()
    
    try:
        result = processor.process(pdf_path)
        print(f"[OK] PDF loaded: {result.page_count} pages")
        print(f"  File size: {result.metadata.file_size_bytes / 1024:.1f} KB")
        print(f"  Processing time: {result.processing_time_ms}ms\n")
    except Exception as e:
        print(f"[ERROR] Failed to load PDF: {e}")
        return
    
    # Step 2: Create extraction state
    print("[STATE] Creating extraction state...")
    
    # Serialize page images to state format
    page_images = []
    for page in result.pages:
        page_images.append({
            "page_number": page.page_number,
            "data_uri": page.data_uri,
            "width": page.width,
            "height": page.height,
            "dpi": page.dpi,
        })
    
    state = create_initial_state(
        pdf_path=pdf_path,
        pdf_hash=result.metadata.file_hash,
        page_images=page_images,
    )
    print(f"[OK] State created with {len(page_images)} page images\n")
    
    # Step 3: Run LayoutAgent
    print("="*80)
    print("STAGE 1: VLM Layout Analysis")
    print("="*80 + "\n")
    
    client = LMStudioClient()
    
    # Check LM Studio health
    print("Checking LM Studio connection...")
    if not client.is_healthy():
        print("[ERROR] LM Studio server not responding!")
        print(f"   Make sure LM Studio is running at: {settings.lm_studio.base_url}")
        print(f"   Model loaded: {settings.lm_studio.model}")
        return
    print(f"[OK] LM Studio connected: {settings.lm_studio.base_url}")
    print(f"  Model: {settings.lm_studio.model}\n")
    
    layout_agent = LayoutAgent(client=client)
    
    try:
        print("[ANALYZE] Analyzing layout and visual marks...")
        state = layout_agent.process(state)
        
        print(f"\n[OK] Layout analysis completed!")

        print(f"  VLM calls: {state.get('total_vlm_calls', 0)}")
        print(f"  Processing time: {state.get('total_processing_time_ms', 0)}ms\n")
        
        # Display layout analysis results
        for layout in state.get("layout_analyses", []):
            page_num = layout.get("page_number", 0)
            print(f"--- Page {page_num} Layout ---")
            print(f"  Type: {layout.get('layout_type', 'unknown')}")
            print(f"  Confidence: {layout.get('layout_confidence', 0.0):.2f}")
            print(f"  Columns: {layout.get('column_count', 1)}")
            print(f"  Reading Order: {layout.get('reading_order', 'unknown')}")
            print(f"  Density: {layout.get('density_estimate', 'unknown')}")
            print(f"  Estimated Fields: {layout.get('estimated_field_count', 0)}")
            print(f"  Pre-printed Form: {layout.get('has_pre_printed_structure', False)}")
            print(f"  Handwriting: {layout.get('has_handwritten_content', False)}")
            
            # Visual marks
            visual_marks = layout.get("visual_marks", [])
            print(f"\n  Visual Marks Detected: {len(visual_marks)}")
            if visual_marks:
                mark_types = {}
                for mark in visual_marks:
                    mark_type = mark.get("mark_type", "unknown")
                    mark_types[mark_type] = mark_types.get(mark_type, 0) + 1
                
                for mark_type, count in sorted(mark_types.items()):
                    print(f"    - {mark_type}: {count}")
            
            # Regions
            regions = layout.get("regions", [])
            print(f"\n  Regions: {len(regions)}")
            for region in regions[:5]:  # Show first 5
                print(f"    - {region.get('region_type', 'unknown')}: {region.get('description', 'no description')}")
            
            # VLM observations
            observations = layout.get("vlm_observations", "")
            if observations:
                print(f"\n  VLM Observations:")
                print(f"    {observations[:200]}...")
            
            print(f"\n  Extraction Difficulty: {layout.get('extraction_difficulty', 'unknown')}")
            print(f"  Recommended Strategy: {layout.get('recommended_strategy', 'unknown')}")
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
        print("[DETECT] Detecting components (tables, forms, checkboxes, key-value pairs)...")
        state = component_agent.process(state)
        
        print(f"\n[OK] Component detection completed!")
        print(f"  Total VLM calls: {state.get('total_vlm_calls', 0)}")
        print(f"  Total time: {state.get('total_processing_time_ms', 0)}ms\n")
        
        # Display component detection results
        for comp_map in state.get("component_maps", []):
            page_num = comp_map.get("page_number", 0)
            print(f"--- Page {page_num} Components ---")
            
            # Tables
            tables = comp_map.get("tables", [])
            print(f"  Tables: {len(tables)}")
            for table in tables:
                print(f"    - {table.get('description', 'table')}: "
                      f"{table.get('row_count', 0)} rows × {table.get('column_count', 0)} cols "
                      f"(confidence: {table.get('confidence', 0.0):.2f})")
            
            # Forms
            forms = comp_map.get("forms", [])
            print(f"\n  Form Fields: {len(forms)}")
            field_types = {}
            for form in forms:
                field_type = form.get("field_type", "unknown")
                field_types[field_type] = field_types.get(field_type, 0) + 1
            
            for field_type, count in sorted(field_types.items()):
                print(f"    - {field_type}: {count}")
            
            # Checkboxes specifically
            checkboxes = [f for f in forms if "checkbox" in f.get("field_type", "")]
            if checkboxes:
                print(f"\n  Checkbox States:")
                for cb in checkboxes[:10]:  # Show first 10
                    label = cb.get("label_text", "unknown")
                    is_filled = cb.get("is_filled", False)
                    state_indicator = "[X]" if is_filled else "[ ]"
                    print(f"    {state_indicator} {label}")
            
            # Key-value pairs
            kv_pairs = comp_map.get("key_value_pairs", [])
            print(f"\n  Key-Value Pairs: {len(kv_pairs)}")
            for kv in kv_pairs[:5]:  # Show first 5
                key = kv.get("key_text", "unknown")
                value_type = kv.get("value_type_hint", "text")
                print(f"    - {key} → ({value_type})")
            
            # Special elements
            special = comp_map.get("special_elements", [])
            if special:
                print(f"\n  Special Elements: {len(special)}")
                for elem in special:
                    print(f"    - {elem.get('element_type', 'unknown')}: {elem.get('description', 'no description')}")
            
            # Content summary
            print(f"\n  Content Summary:")
            print(f"    Tabular data: {comp_map.get('has_tabular_data', False)}")
            print(f"    Form fields: {comp_map.get('has_form_fields', False)}")
            print(f"    Checkboxes: {comp_map.get('has_checkboxes', False)}")
            print(f"    Signatures: {comp_map.get('has_signatures', False)}")
            print(f"    Handwriting: {comp_map.get('has_handwriting', False)}")
            
            # Extraction strategies
            strategies = comp_map.get("suggested_extraction_strategies", {})
            if strategies:
                print(f"\n  Suggested Extraction Strategies:")
                for comp_type, strategy in strategies.items():
                    print(f"    - {comp_type}: {strategy}")
            
            # Extraction order
            order = comp_map.get("extraction_order", [])
            if order:
                print(f"\n  Recommended Extraction Order: {' → '.join(order)}")
            
            print()
    
    except Exception as e:
        print(f"[ERROR] Component detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    print("="*80)
    print("RESULTS SAVED")
    print("="*80 + "\n")
    
    output_path = Path("test_vlm_first_results.json")
    with open(output_path, "w") as f:
        # Create simplified output for JSON serialization
        output = {
            "pdf_path": str(pdf_path),
            "page_count": len(page_images),
            "total_vlm_calls": state.get("total_vlm_calls", 0),
            "total_processing_time_ms": state.get("total_processing_time_ms", 0),
            "layout_analyses": state.get("layout_analyses", []),
            "component_maps": state.get("component_maps", []),
        }
        json.dump(output, f, indent=2)
    
    print(f"[OK] Results saved to: {output_path}")
    print(f"\n  Layout analyses: {len(state.get('layout_analyses', []))} pages")
    print(f"  Component maps: {len(state.get('component_maps', []))} pages")
    print(f"  Total VLM calls: {state.get('total_vlm_calls', 0)}")
    print(f"  Total time: {state.get('total_processing_time_ms', 0)}ms")
    
    print("\n" + "="*80)
    print("[SUCCESS] VLM-FIRST PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
