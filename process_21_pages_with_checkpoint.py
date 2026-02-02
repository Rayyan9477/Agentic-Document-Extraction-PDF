"""
Batch processor for 21 pages with incremental saving and resume capability.
"""

import os
if "AGENT" in os.environ:
    del os.environ["AGENT"]

import json
from pathlib import Path
import time

from universal_multi_record_extraction import (
    convert_from_path, detect_document_type_and_entity, generate_adaptive_schema,
    process_single_page, DocumentExtractionResult, save_results
)
from export_consolidated_multipage import export_consolidated_excel, export_consolidated_markdown


def process_pages_batch(pdf_path, start_page, end_page, checkpoint_file="checkpoint.json"):
    """
    Process pages in batches with checkpointing.
    """
    # Check for existing checkpoint
    checkpoint = {}
    if Path(checkpoint_file).exists():
        print(f"[Found checkpoint file: {checkpoint_file}]")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"[Resuming from page {checkpoint.get('last_completed_page', 0) + 1}]")
    
    # Initialize or resume
    if not checkpoint:
        # First run - detect document type and schema
        print("\n[Initializing - detecting document type and generating schema]")
        pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
        first_page = pages[0]
        
        doc_metadata = detect_document_type_and_entity(first_page)
        schema = generate_adaptive_schema(
            first_page,
            doc_metadata['document_type'],
            doc_metadata['entity_type']
        )
        
        checkpoint = {
            'pdf_path': pdf_path,
            'start_page': start_page,
            'end_page': end_page,
            'document_metadata': doc_metadata,
            'schema': schema,
            'records': [],
            'last_completed_page': 0,
            'total_vlm_calls': 2,  # 1 for doc type + 1 for schema
            'start_time': time.time()
        }
    
    doc_metadata = checkpoint['document_metadata']
    schema = checkpoint['schema']
    all_records = checkpoint['records']
    vlm_calls = checkpoint['total_vlm_calls']
    resume_page = checkpoint['last_completed_page'] + 1
    
    # Process remaining pages
    for page_num in range(max(resume_page, start_page), end_page + 1):
        print(f"\n{'='*70}")
        print(f"PROCESSING PAGE {page_num} ({page_num - start_page + 1}/{end_page - start_page + 1})")
        print(f"{'='*70}")
        
        try:
            page_records = process_single_page(pdf_path, page_num, doc_metadata, schema)
            
            # Convert to dict format
            for record in page_records:
                all_records.append({
                    'record_id': record.record_id,
                    'page_number': record.page_number,
                    'primary_identifier': record.primary_identifier,
                    'entity_type': record.entity_type,
                    'fields': record.fields,
                    'confidence': record.confidence,
                    'extraction_time_ms': record.extraction_time_ms
                })
            
            vlm_calls += 1 + len(page_records)  # 1 boundary + N extractions
            
            # Update checkpoint
            checkpoint['records'] = all_records
            checkpoint['last_completed_page'] = page_num
            checkpoint['total_vlm_calls'] = vlm_calls
            
            # Save checkpoint after each page
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            print(f"[Checkpoint saved: {len(all_records)} records so far]")
            
        except Exception as e:
            print(f"[ERROR on page {page_num}: {str(e)}]")
            print("[Checkpoint saved - you can resume from this point]")
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            raise
    
    # All pages completed
    total_time = time.time() - checkpoint['start_time']
    
    result_data = {
        'pdf_path': checkpoint['pdf_path'],
        'total_pages': end_page - start_page + 1,
        'total_records': len(all_records),
        'document_type': doc_metadata['document_type'],
        'entity_type': doc_metadata['entity_type'],
        'schema': schema,
        'records': all_records,
        'total_processing_time_ms': int(total_time * 1000),
        'total_vlm_calls': vlm_calls
    }
    
    return result_data


def main():
    print("="*70)
    print("BATCH PROCESSING: 21 Pages with Checkpointing")
    print("="*70)
    
    pdf_path = "superbill1.pdf"
    checkpoint_file = "extraction_checkpoint.json"
    
    try:
        # Process all pages
        result_data = process_pages_batch(
            pdf_path=pdf_path,
            start_page=1,
            end_page=21,
            checkpoint_file=checkpoint_file
        )
        
        # Save final results
        output_file = "universal_extraction_all_21_pages.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print(f"Total Pages: {result_data['total_pages']}")
        print(f"Total Records: {result_data['total_records']}")
        print(f"Processing Time: {result_data['total_processing_time_ms'] / 1000:.2f}s")
        print(f"VLM Calls: {result_data['total_vlm_calls']}")
        print(f"Results saved to: {output_file}")
        
        # Create exports
        print("\n" + "="*70)
        print("CREATING EXPORTS")
        print("="*70)
        
        excel_file = "consolidated_all_21_pages.xlsx"
        export_consolidated_excel(result_data, excel_file)
        
        markdown_file = "consolidated_all_21_pages.md"
        export_consolidated_markdown(result_data, markdown_file)
        
        # Clean up checkpoint
        Path(checkpoint_file).unlink()
        print(f"\n[Checkpoint file deleted]")
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progress saved in checkpoint file")
        print(f"[Run again to resume from last completed page]")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print(f"[Progress saved - run again to resume]")


if __name__ == "__main__":
    main()
