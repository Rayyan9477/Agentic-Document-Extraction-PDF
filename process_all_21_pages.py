"""
Process ALL 21 pages of superbill1.pdf with universal multi-record extraction.
"""

import os
if "AGENT" in os.environ:
    del os.environ["AGENT"]

from universal_multi_record_extraction import process_multi_page_document, save_results
from export_consolidated_multipage import export_consolidated_excel, export_consolidated_markdown, load_extraction_results

def main():
    print("="*70)
    print("FULL 21-PAGE EXTRACTION")
    print("="*70)
    
    pdf_path = "superbill1.pdf"
    
    print("\n[WARNING] This will process ALL 21 pages")
    print("[ESTIMATED TIME] ~15-20 minutes")
    print("[ESTIMATED VLM CALLS] ~60-80 calls")
    print()
    
    # Process all 21 pages
    result = process_multi_page_document(
        pdf_path=pdf_path,
        start_page=1,
        end_page=21,  # All pages
    )
    
    # Save raw results
    results_file = "universal_extraction_all_21_pages.json"
    save_results(result, results_file)
    
    # Load and export
    print("\n" + "="*70)
    print("CREATING CONSOLIDATED EXPORTS")
    print("="*70)
    
    data = load_extraction_results(results_file)
    
    # Export to Excel
    excel_output = "consolidated_all_21_pages.xlsx"
    export_consolidated_excel(data, excel_output)
    
    # Export to Markdown
    markdown_output = "consolidated_all_21_pages.md"
    export_consolidated_markdown(data, markdown_output)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  - {results_file} (raw JSON)")
    print(f"  - {excel_output} (consolidated Excel)")
    print(f"  - {markdown_output} (consolidated Markdown)")
    print()
    print("Summary:")
    print(f"  - Pages processed: {data['total_pages']}")
    print(f"  - Records extracted: {data['total_records']}")
    print(f"  - Processing time: {data['total_processing_time_ms'] / 1000:.2f}s")
    print(f"  - VLM calls: {data['total_vlm_calls']}")


if __name__ == "__main__":
    main()
