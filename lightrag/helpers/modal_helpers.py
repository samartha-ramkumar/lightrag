import base64
import io
import uuid
from typing import Any, Dict, List, Union
from loguru import logger
import pymupdf4llm
import fitz  # PyMuPDF
from PIL import Image
import json
from bs4 import BeautifulSoup
import markdown2
from loguru import logger
import html2text
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from lightrag.llm.provider import LLMService

# Singleton LLM service instance
_llm_service = None

# Define threshold for switching to full page rendering
MAX_IMAGES_PER_PAGE = 2  # If page has more images than this, render entire page



def get_llm_service():
    """Get or create LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service



@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def process_pdf(file_path_or_bytes: Union[str, bytes], extract_images: bool = True) -> Dict[str, Any]:
    """
    Process a PDF file and convert it to markdown with extracted images if requested.
    
    Args:
        file_path_or_bytes: Either a file path or bytes of the PDF
        extract_images: Whether to extract and process images
        
    Returns:
        Dict with markdown content and extracted images
    """
    logger.info(f"Processing PDF: {file_path_or_bytes if isinstance(file_path_or_bytes, str) else 'bytes'}")
    result = {
        "markdown": "",
        "images": [],
        "pages": []
    }
    
    try:
        # Open the document using PyMuPDF
        if isinstance(file_path_or_bytes, str):
            doc = fitz.open(file_path_or_bytes)
        else:
            doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")
        
        # Use pymupdf4llm for better markdown conversion
        try:
            result["markdown"] = pymupdf4llm.to_markdown(doc)
            logger.info(f"Successfully converted PDF to markdown using pymupdf4llm")
        except (ImportError, Exception) as e:
            logger.warning(f"Error using pymupdf4llm: {str(e)}. Falling back to standard PyMuPDF.")
            
            # Fallback to original implementation
            for page_num, page in enumerate(doc):
                text = page.get_text("markdown")
                result["markdown"] += text + "\n\n"
                result["pages"].append({"page_num": page_num, "content": text})
            
        # Extract images if requested
        if extract_images:
            for page_num, page in enumerate(doc):
                page_images = []
                image_list = page.get_images(full=True)
                
                # Check if we should process individual images or the entire page
                if len(image_list) > MAX_IMAGES_PER_PAGE:
                    logger.info(f"Page {page_num+1} has {len(image_list)} images, rendering entire page as image")
                    try:
                        # Render the entire page as an image
                        page_image = render_page_as_image(page)
                        
                        # Convert to base64 for markdown embedding
                        buffered = io.BytesIO()
                        page_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        image_bytes = buffered.getvalue()
                        
                        # Process with Vision API if available
                        vision_description = ""
                        try:
                            vision_description = await extract_image_description(image_bytes)
                        except Exception as vision_err:
                            logger.error(f"Vision API error for page {page_num+1}: {vision_err}")
                    
                        # image_md = f"![Full page {page_num+1}](data:image/png;base64,{img_base64})"
                        
                        logger.debug(f"Vision Description for page {page_num+1}: {vision_description}")

                        # if vision_description:
                        #     image_md += f"\n\n*Page description: {vision_description}*"
                        
                        page_images.append({
                            "image_index": f"{page_num+1}-full",
                            # "image_markdown": image_md,
                            "markdown": vision_description,
                            "image_bytes": image_bytes
                        })
                        result["markdown"] += f"Detailed description of page {page_num+1}: \n{vision_description}\n\n"
                        
                    except Exception as page_img_err:
                        logger.error(f"Error rendering page {page_num+1} as image: {str(page_img_err)}")
                else:
                    # Process individual images as before
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Convert to PIL Image
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # Convert to base64 for markdown embedding
                            buffered = io.BytesIO()
                            image.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Process with Vision API if available
                            vision_description = ""
                            try:
                                vision_description = await extract_image_description(image_bytes)
                            except Exception as vision_err:
                                logger.error(f"Vision API error for image {img_index+1} on page {page_num+1}: {vision_err}")
                        
                            # image_md = f"![Image from page {page_num+1}, image {img_index+1}](data:image/png;base64,{img_base64})"
                            
                            logger.debug(f"Vision Description for image {img_index+1} on page {page_num+1}: {vision_description}")

                            # if vision_description:
                            #     image_md += f"\n\n*Image description: {vision_description}*"
                            
                            page_images.append({
                                "image_index": f"{page_num+1}-{img_index+1}",
                                # "image_markdown": image_md,
                                "markdown": vision_description,
                                "image_bytes": image_bytes
                            })
                            result["markdown"] += f"Detailed description of images in this document:  ## ![Image from page {page_num+1}, image {img_index+1}] \n{vision_description}\n"

                        except Exception as img_err:
                            logger.error(f"Error processing image {img_index+1} on page {page_num+1}: {str(img_err)}")
                            # Continue processing other images even if one fails
                    
                result["images"].extend(page_images)
                
        doc.close()
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise
        
    return result

def render_page_as_image(page, zoom=1.0):
    """
    Render a PDF page as a PIL Image.
    
    Args:
        page: The PyMuPDF page object
        zoom: Zoom factor for higher resolution
    
    Returns:
        PIL.Image: The rendered page as an image
    """
    try:
        # Get the page dimensions
        rect = page.rect
        
        # Create matrix for rendering at higher resolution
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        return img
    except Exception as e:
        logger.error(f"Error rendering page as image: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def extract_image_description(image_bytes: bytes) -> str:
    """Extract a description of an image using configured vision model"""
    try:
        # Check if we have valid image data
        if not image_bytes or len(image_bytes) < 100:
            logger.warning(f"Image data too small or empty: {len(image_bytes) if image_bytes else 0} bytes")
            return "Unable to process image - insufficient data"
            
        # Check if this is an SVG file
        is_svg = False
        if image_bytes[:5].lower() == b'<?xml' or image_bytes[:4].lower() == b'<svg':
            is_svg = True
        
        # Process the image to ensure it's in a supported format
        try:
            if is_svg:
                # Convert SVG to a format that can be processed by vision API
                try:
                    import cairosvg
                    # Convert SVG to PNG
                    png_bytes = cairosvg.svg2png(bytestring=image_bytes)
                    image = Image.open(io.BytesIO(png_bytes))
                except ImportError:
                    try:
                        from svglib.svglib import svg2rlg
                        from reportlab.graphics import renderPM
                        from io import BytesIO
                        
                        # Convert SVG to PNG using svglib
                        drawing = svg2rlg(BytesIO(image_bytes))
                        png_bytes = BytesIO()
                        renderPM.drawToFile(drawing, png_bytes, fmt="PNG")
                        png_bytes.seek(0)
                        image = Image.open(png_bytes)
                    except ImportError:
                        logger.warning("SVG processing libraries not available - vision API may not work with raw SVG")
                        return "SVG Image (description unavailable - conversion tools missing)"
            else:
                # Standard image processing for non-SVG formats
                image = Image.open(io.BytesIO(image_bytes))
                
            # Check if the image format is supported
            if hasattr(image, 'format') and image.format:
                logger.debug(f"Image format detected: {image.format}")
            else:
                logger.warning("Image format not detected, attempting conversion")

            # Convert to RGB if needed (handles RGBA, CMYK, etc.)
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Save with higher quality (90) for better text recognition
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", quality=90)
            processed_image_bytes = buffered.getvalue()
            base64_image = base64.b64encode(processed_image_bytes).decode('ascii')
        except Exception as img_err:
            logger.error(f"Error processing image format: {str(img_err)}")
            return f"Image processing error: {str(img_err)}"
        
        # Use the LLM service to analyze the image
        llm_service = get_llm_service()
        result = await llm_service.analyze_image(image_data=base64_image, prompt_type="general")
        
        if not result or result.startswith("Error"):
            logger.error(f"LLM service error in image analysis: {result}")
            return "Image analysis service unavailable"
            
        return result
    except Exception as e:
        logger.error(f"Error extracting image description: {str(e)}")
        return f"Failed to extract image description: {str(e)}"



async def process_image(file_path_or_bytes: Union[str, bytes]) -> Dict[str, Any]:
    """Process an image file and convert to markdown with description"""
    try:
        # Handle different input types
        if isinstance(file_path_or_bytes, str):
            # It's a file path
            try:
                with open(file_path_or_bytes, "rb") as img_file:
                    image_bytes = img_file.read()
            except Exception as e:
                logger.error(f"Failed to read image file: {str(e)}")
                return {
                    "markdown": "*Image processing failed: Could not read file*",
                    "description": "Image processing error",
                    "media_id": f"media_{uuid.uuid4().hex[:12]}",
                    "content_type": "text/plain",
                }
        else:
            # It's already bytes
            image_bytes = file_path_or_bytes

        # Try to open and process the image
        try:
            # Check if this is an SVG file first
            is_svg = False
            if image_bytes[:5].lower() == b'<?xml' or image_bytes[:4].lower() == b'<svg':
                is_svg = True
            
            if is_svg:
                # Handle SVG files with cairosvg or svglib if available
                try:
                    import cairosvg
                    # Convert SVG to PNG
                    png_bytes = cairosvg.svg2png(bytestring=image_bytes)
                    image = Image.open(io.BytesIO(png_bytes))
                    logger.info("Processed SVG using cairosvg")
                except ImportError:
                    try:
                        from svglib.svglib import svg2rlg
                        from reportlab.graphics import renderPM
                        from io import BytesIO
                        
                        # Convert SVG to PNG using svglib
                        drawing = svg2rlg(BytesIO(image_bytes))
                        png_bytes = BytesIO()
                        renderPM.drawToFile(drawing, png_bytes, fmt="PNG")
                        png_bytes.seek(0)
                        image = Image.open(png_bytes)
                        logger.info("Processed SVG using svglib")
                    except ImportError:
                        # If neither library is available, just store SVG as is
                        logger.warning("SVG processing libraries not available, using raw SVG")
                        svg_base64 = base64.b64encode(image_bytes).decode()
                        media_id = f"media_{uuid.uuid4().hex[:12]}"
                        
                        llm_service = get_llm_service()
                        description = await llm_service.analyze_image(image_data=svg_base64, prompt_type="general")
                        return {
                            # "image_markdown": f"![SVG Image](data:image/svg+xml;base64,{svg_base64})",
                            "markdown": description,
                            "media_id": media_id,
                            "content_type": "image/svg+xml",
                        }
            else:
                # For non-SVG, try normal PIL processing
                image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to base64 for markdown embedding
            buffered = io.BytesIO()
            
            # Convert to RGB if needed (handles RGBA, CMYK, etc.)
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
                
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Get image description if vision API is enabled
            description = ""
            try:
                llm_service = get_llm_service()
                description = await llm_service.analyze_image(image_data=img_base64, prompt_type="general")
                # If it returns an error message, use a generic description
                if description.startswith("Error"):
                    logger.warning(f"Vision API error: {description}")
                    description = "Image (description unavailable)"
            except Exception as vision_err:
                logger.error(f"Error in vision processing: {str(vision_err)}")
                description = "Image (description unavailable)"
            
            # Create markdown representation with embedded image
            # image_md = f"![Image](data:image/png;base64,{img_base64})"
            # if description and not description.startswith("Error"):
            #     image_md += f"\n\n*Description: {description}*"
            
            # Generate a unique media ID for tracking
            media_id = f"media_{uuid.uuid4().hex[:12]}"
            
            return {
                # "image_markdown": image_md,
                "markdown": description,
                "media_id": media_id,
                "content_type": "image/png",
            }
        except Exception as img_err:
            logger.error(f"Failed to process image: {str(img_err)}")
            return {
                # "image_markdown": "*Image processing failed*",
                "markdown": f"Error: {str(img_err)}",
                "media_id": f"media_{uuid.uuid4().hex[:12]}",
                "content_type": "text/plain",
            }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Return a fallback result instead of raising exception
        return {
            # "image_markdown": "*Image processing failed*",
            "markdown": "Error occurred during image processing",
            "media_id": f"media_{uuid.uuid4().hex[:12]}",
            "content_type": "text/plain",
        }

# Fix the process_text function to support rich text content explicitly
def process_text(text_content: str) -> Dict[str, Any]:
    """Process plain text or rich text content into markdown format"""
    try:
        # Check if this is a JSON string (rich text)
        import json
        
        # Try to parse as JSON first to handle rich text format
        try:
            content_json = json.loads(text_content)
            
            # Handle Draft.js content structure
            if isinstance(content_json, dict) and "blocks" in content_json:
                blocks = content_json.get("blocks", [])
                paragraphs = []
                
                for block in blocks:
                    if "text" in block and block["text"].strip():
                        paragraphs.append(block["text"])
                
                markdown = "\n\n".join(paragraphs)
                return {"markdown": markdown}
        except (json.JSONDecodeError, TypeError):
            # Not JSON or couldn't be parsed as rich text, treat as plain text
            pass
            
        # Process as plain text
        paragraphs = text_content.split('\n\n')
        markdown = '\n\n'.join(paragraphs)
        return {"markdown": markdown}
    except Exception as e:
        logger.error(f"Error processing text content: {str(e)}")
        # Provide a basic fallback to prevent complete failure
        return {"markdown": text_content}



class MarkdownConverter:
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0  # No wrapping

    def html_to_markdown(self, html_content: str) -> str:
        """Convert HTML to Markdown"""
        try:
            return self.html_converter.handle(html_content)
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {str(e)}")
            # Fallback to BeautifulSoup + markdown2
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                # Clean up the HTML
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text()
                # Break into lines and remove leading/trailing space
                lines = (line.strip() for line in text.splitlines())
                # Break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # Join lines
                text = '\n'.join(chunk for chunk in chunks if chunk)
                return text
            except Exception as e2:
                logger.error(f"Fallback HTML conversion failed: {str(e2)}")
                return html_content  # Return original as last resort

    def url_to_markdown(self, url: str) -> str:
        """Fetch URL and convert its content to Markdown"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type', ''):
                return self.html_to_markdown(response.text)
            else:
                return response.text
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return f"Error fetching content from {url}: {str(e)}"

    def json_to_markdown(self, json_content: Union[str, Dict, List]) -> str:
        """Convert JSON to Markdown representation"""
        try:
            # If the input is a string, parse it as JSON
            if isinstance(json_content, str):
                data = json.loads(json_content)
            else:
                data = json_content

            # Handle JSON arrays and objects
            if isinstance(data, list):
                markdown = ""
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        markdown += f"## Item {i+1}\n\n"
                        markdown += self._dict_to_markdown(item)
                    else:
                        markdown += f"- {item}\n"
                return markdown
            elif isinstance(data, dict):
                return self._dict_to_markdown(data)
            else:
                return str(data)
        except Exception as e:
            logger.error(f"Error converting JSON to Markdown: {str(e)}")
            if isinstance(json_content, str):
                return json_content
            return str(json_content)

    def _dict_to_markdown(self, data: Dict[str, Any], level: int = 0) -> str:
        """Helper to convert a dictionary to Markdown with proper formatting"""
        markdown = ""
        for key, value in data.items():
            heading_level = "#" * (level + 2) if level < 4 else "######"
            if isinstance(value, dict):
                markdown += f"{heading_level} {key}\n\n"
                markdown += self._dict_to_markdown(value, level + 1)
            elif isinstance(value, list):
                markdown += f"{heading_level} {key}\n\n"
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        markdown += f"### {key} {i+1}\n\n"
                        markdown += self._dict_to_markdown(item, level + 2)
                    else:
                        markdown += f"- {item}\n"
                markdown += "\n"
            else:
                markdown += f"**{key}**: {value}\n\n"
        return markdown
    



async def process_url(url: str) -> Dict[str, Any]:
    """Fetch and process content from a URL"""
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        
        if 'text/html' in content_type:
            converter = MarkdownConverter()
            markdown = converter.html_to_markdown(response.text)
            return {"markdown": markdown, "url": url}
        elif 'application/pdf' in content_type:
            return await process_pdf(response.content)  # Make sure await is used
        elif 'image/' in content_type:
            return await process_image(response.content)  # Make sure await is used
        else:
            # Default to text processing
            return process_text(response.text)
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise


def process_structured_data(data: Union[List[Dict], Dict, str]) -> Dict[str, Any]:
    """
    Process structured data (JSON, CSV, etc.) to Markdown format
    
    Args:
        data: The structured data to process
        
    Returns:
        Dictionary with processed markdown
    """
    import pandas as pd
    
    logger.info(f"Processing structured data: {type(data)}")
    
    converter = MarkdownConverter()
    
    # If data is a string, try to parse it as JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            # If it's not valid JSON, try to parse as CSV
            try:
                df = pd.read_csv(data)
                return {"markdown": df.to_markdown(index=False)}
            except Exception as e:
                logger.error(f"Failed to parse data as CSV: {str(e)}")
                return {"markdown": f"```\n{data}\n```"}
    
    # Convert to markdown
    if isinstance(data, (list, dict)):
        markdown = converter.json_to_markdown(data)
        return {"markdown": markdown}
    else:
        return {"markdown": f"```\n{data}\n```"}
    

async def cleanup_llm_service():
    """Clean up LLM service resources"""
    global _llm_service
    if _llm_service:
        await _llm_service.cleanup()
        _llm_service = None

def cleanup_ollama_session():
    """Clean up Ollama session when shutting down"""
    global _ollama_session
    if _ollama_session and not _ollama_session.closed:
        import asyncio
        try:
            asyncio.run(_ollama_session.close())
        except Exception as e:
            logger.error(f"Error closing Ollama session: {e}")
        _ollama_session = None

def strip_base64_for_storage(content: str) -> str:
    """Strip base64 image data from content to save space while preserving image references"""
    import re
    
    # Replace base64 data with placeholder but keep the alt text
    base64_pattern = r'(!\[.*?\])\(data:image\/[^;]+;base64,[a-zA-Z0-9+/=]+\)'
    return re.sub(base64_pattern, r'\1([IMAGE DATA REMOVED])', content)

def extract_base64_images(content: str) -> List[Dict[str, Any]]:
    """
    Extract base64 images from markdown content
    
    Args:
        content: Markdown content with embedded base64 images
        
    Returns:
        List of dicts containing image info (alt_text, mime_type, data)
    """
    import re
    import uuid
    
    # Pattern to match markdown images with base64 data
    pattern = r'!\[(.*?)\]\(data:image\/([^;]+);base64,([a-zA-Z0-9+/=]+)\)'
    
    images = []
    for match in re.finditer(pattern, content):
        alt_text = match.group(1)
        mime_type = match.group(2)
        base64_data = match.group(3)
        
        try:
            # Decode base64 to binary
            binary_data = base64.b64decode(base64_data)
            
            images.append({
                "alt_text": alt_text,
                "mime_type": mime_type,
                "content_type": f"image/{mime_type}",
                "data": binary_data,
                "media_id": f"media_{uuid.uuid4().hex[:12]}",
                "base64_match": match.group(0)  # Store the full match for replacement
            })
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
    
    return images