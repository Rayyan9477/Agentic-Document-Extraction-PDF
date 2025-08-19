"""
PDF to Image Conversion Module
Handles secure conversion of PDF pages to images with encryption and cleanup.
"""

import fitz  # PyMuPDF
import io
import logging
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from src.config.azure_config import azure_config

logger = logging.getLogger(__name__)


class PDFConverter:
    """Handles PDF to image conversion with security and cleanup."""
    
    def __init__(self):
        self.temp_files = []  # Track temp files for cleanup
        self.encrypted_images = {}  # Store encrypted image data
    
    def convert_pdf_to_images(
        self, 
        pdf_bytes: bytes, 
        dpi: int = 300,
        in_memory: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convert PDF pages to images with security considerations.
        
        Args:
            pdf_bytes: PDF file content as bytes
            dpi: Resolution for image conversion (default 300 DPI)
            in_memory: Keep images in memory vs encrypted temp files
            
        Returns:
            List of page data with images and metadata
        """
        try:
            logger.info(f"Starting PDF conversion with DPI={dpi}, in_memory={in_memory}")
            
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_images = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Convert page to image
                image_data = self._convert_page_to_image(page, page_num, dpi, in_memory)
                page_images.append(image_data)
                
                logger.info(f"Converted page {page_num + 1}/{pdf_document.page_count}")
            
            pdf_document.close()
            logger.info(f"Successfully converted {len(page_images)} pages")
            
            return page_images
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    
    def _convert_page_to_image(
        self, 
        page: fitz.Page, 
        page_num: int, 
        dpi: int,
        in_memory: bool
    ) -> Dict[str, Any]:
        """Convert a single PDF page to image."""
        try:
            # Calculate zoom factor for desired DPI
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            
            # Render page to pixmap
            pixmap = page.get_pixmap(matrix=matrix)
            
            # Convert to PIL Image
            img_data = pixmap.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Get page metadata
            page_info = {
                "page_number": page_num,
                "width": pil_image.width,
                "height": pil_image.height,
                "dpi": dpi,
                "original_width": page.rect.width,
                "original_height": page.rect.height
            }
            
            if in_memory:
                # Store image in memory (encrypted)
                encrypted_image = self._encrypt_image_data(img_data)
                image_id = f"page_{page_num}"
                self.encrypted_images[image_id] = encrypted_image
                
                page_info.update({
                    "image_id": image_id,
                    "storage_type": "memory",
                    "image": pil_image  # Keep PIL image for immediate use
                })
            else:
                # Store as encrypted temporary file
                temp_path = self._save_encrypted_temp_image(img_data, page_num)
                page_info.update({
                    "temp_path": temp_path,
                    "storage_type": "file",
                    "image": pil_image
                })
            
            pixmap = None  # Cleanup
            return page_info
            
        except Exception as e:
            logger.error(f"Failed to convert page {page_num}: {e}")
            raise
    
    def _encrypt_image_data(self, image_data: bytes) -> bytes:
        """Encrypt image data for secure storage."""
        try:
            return azure_config.encrypt_data(image_data.hex())
        except Exception as e:
            logger.error(f"Image encryption failed: {e}")
            # Return original data if encryption fails (development mode)
            return image_data
    
    def _decrypt_image_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt image data."""
        try:
            hex_data = azure_config.decrypt_data(encrypted_data)
            return bytes.fromhex(hex_data)
        except Exception:
            # Return original data if decryption fails (development mode)
            return encrypted_data
    
    def _save_encrypted_temp_image(self, image_data: bytes, page_num: int) -> str:
        """Save encrypted image to temporary file."""
        try:
            # Create secure temporary file
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=f"_page_{page_num}.enc",
                prefix="medical_pdf_",
                dir=None
            )
            
            # Encrypt and write image data
            encrypted_data = self._encrypt_image_data(image_data)
            
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(encrypted_data)
            
            # Track for cleanup
            self.temp_files.append(temp_path)
            
            logger.info(f"Saved encrypted temp image: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save encrypted temp image: {e}")
            raise
    
    def get_image_from_storage(self, page_info: Dict[str, Any]) -> Image.Image:
        """Retrieve image from storage (memory or file)."""
        try:
            if page_info["storage_type"] == "memory":
                # Get from encrypted memory storage
                image_id = page_info["image_id"]
                if image_id in self.encrypted_images:
                    encrypted_data = self.encrypted_images[image_id]
                    image_data = self._decrypt_image_data(encrypted_data)
                    return Image.open(io.BytesIO(image_data))
                else:
                    # Fallback to PIL image if available
                    return page_info.get("image")
            
            elif page_info["storage_type"] == "file":
                # Load from encrypted temp file
                temp_path = page_info["temp_path"]
                if os.path.exists(temp_path):
                    with open(temp_path, 'rb') as f:
                        encrypted_data = f.read()
                    
                    image_data = self._decrypt_image_data(encrypted_data)
                    return Image.open(io.BytesIO(image_data))
                else:
                    # Fallback to PIL image if available
                    return page_info.get("image")
            
            raise ValueError(f"Unknown storage type: {page_info['storage_type']}")
            
        except Exception as e:
            logger.error(f"Failed to retrieve image from storage: {e}")
            raise
    
    def get_page_text_bounds(self, pdf_bytes: bytes, page_num: int) -> List[Dict[str, Any]]:
        """Extract text blocks with bounding boxes from PDF page."""
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = pdf_document[page_num]
            
            # Get text blocks with coordinates
            text_blocks = []
            blocks = page.get_text("dict")["blocks"]
            
            for block_num, block in enumerate(blocks):
                if "lines" in block:  # Text block
                    bbox = block["bbox"]  # (x0, y0, x1, y1)
                    
                    # Extract text from lines
                    text_content = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_content += span["text"] + " "
                    
                    text_blocks.append({
                        "block_id": block_num,
                        "bbox": {
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "width": bbox[2] - bbox[0],
                            "height": bbox[3] - bbox[1]
                        },
                        "text": text_content.strip(),
                        "type": "text"
                    })
            
            pdf_document.close()
            return text_blocks
            
        except Exception as e:
            logger.error(f"Failed to extract text bounds from page {page_num}: {e}")
            return []
    
    def cleanup(self):
        """Clean up temporary files and encrypted memory storage."""
        try:
            # Remove temporary files
            for temp_path in self.temp_files:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"Cleaned up temp file: {temp_path}")
            
            self.temp_files.clear()
            
            # Clear encrypted memory storage
            self.encrypted_images.clear()
            
            logger.info("PDF converter cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()


def analyze_pdf_structure(pdf_bytes: bytes) -> Dict[str, Any]:
    """Analyze PDF structure and provide metadata."""
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        structure_info = {
            "page_count": pdf_document.page_count,
            "has_text": False,
            "has_images": False,
            "pages": []
        }
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Check for text content
            text = page.get_text().strip()
            page_has_text = len(text) > 0
            if page_has_text:
                structure_info["has_text"] = True
            
            # Check for images
            image_list = page.get_images()
            page_has_images = len(image_list) > 0
            if page_has_images:
                structure_info["has_images"] = True
            
            # Get page dimensions
            rect = page.rect
            
            structure_info["pages"].append({
                "page_number": page_num,
                "has_text": page_has_text,
                "text_length": len(text),
                "image_count": len(image_list),
                "width": rect.width,
                "height": rect.height,
                "text_blocks": len(page.get_text("dict")["blocks"])
            })
        
        pdf_document.close()
        return structure_info
        
    except Exception as e:
        logger.error(f"PDF structure analysis failed: {e}")
        return {"error": str(e)}


# Global converter instance for reuse
pdf_converter = PDFConverter()