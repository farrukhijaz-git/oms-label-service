import re
import pdfplumber
import io
import logging
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def split_pdf_pages(pdf_bytes: bytes) -> list:
    """Split a multi-page PDF into a list of single-page PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        writer = PdfWriter()
        writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        pages.append(buf.getvalue())
    return pages if pages else [pdf_bytes]


def _ocr_pdf_page(pdf_bytes: bytes) -> str:
    """OCR a single-page PDF using tesseract. Returns extracted text or empty string."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        images = convert_from_bytes(pdf_bytes, dpi=200)
        if not images:
            return ""
        # Single-page PDF — just process the first image
        text = pytesseract.image_to_string(images[0])
        logger.info(f"OCR extracted {len(text)} chars")
        return text
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""

STREET_SUFFIXES = {"st", "street", "ave", "avenue", "blvd", "boulevard", "dr", "drive",
                   "rd", "road", "ln", "lane", "ct", "court", "way", "pl", "place"}
COMPANY_KEYWORDS = {"llc", "inc", "corp", "ltd", "co", "company", "services", "group"}

def extract_label_data(pdf_bytes: bytes) -> dict:
    """Extract customer name and address from PDF bytes.
    Tries pdfplumber text extraction first; falls back to OCR for image-based PDFs."""
    result = {
        "customer_name": None,
        "address": None,
        "is_image_pdf": False,
        "raw_text": "",
    }

    all_text = ""

    # --- Step 1: pdfplumber text extraction ---
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
    except Exception as e:
        logger.warning(f"pdfplumber extraction error: {e}")

    # --- Step 2: OCR fallback for image-based PDFs ---
    if not all_text.strip():
        result["is_image_pdf"] = True
        logger.info("No text from pdfplumber — attempting OCR")
        all_text = _ocr_pdf_page(pdf_bytes)
        if not all_text.strip():
            logger.warning("OCR also returned no text")
            return result
        # OCR succeeded — clear the image flag since we now have text
        result["is_image_pdf"] = False

    result["raw_text"] = all_text
    lines = [line.strip() for line in all_text.split("\n") if line.strip()]
    logger.debug(f"Extracted lines: {lines[:20]}")

    result["customer_name"] = _extract_name(lines)
    result["address"] = _extract_address(lines)

    return result


def _extract_name(lines: list) -> str | None:
    """Find customer name from text lines."""
    name_pattern = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$")

    for line in lines:
        # Skip lines with numbers (likely addresses)
        if re.search(r"\d", line):
            continue

        words = line.split()
        if not 2 <= len(words) <= 4:
            continue

        # Check it looks like a name (title case words)
        if not all(w[0].isupper() for w in words if w):
            continue

        # Skip if contains street/company keywords
        lower_words = {w.lower().rstrip(".,") for w in words}
        if lower_words & STREET_SUFFIXES or lower_words & COMPANY_KEYWORDS:
            continue

        return line

    return None


def _extract_address(lines: list) -> str | None:
    """Find address from text lines."""
    # Pattern: starts with number, has street type
    street_pattern = re.compile(r"^\d+\s+\w+", re.IGNORECASE)
    # Pattern for city, state, zip line
    city_state_zip = re.compile(r"[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}", re.IGNORECASE)

    street_line = None
    city_line = None

    for i, line in enumerate(lines):
        if street_pattern.match(line) and not street_line:
            street_line = line
        if city_state_zip.search(line) and not city_line:
            city_line = line

    if street_line and city_line:
        return f"{street_line}, {city_line}".upper()
    elif street_line:
        return street_line.upper()

    return None


def normalize_text(text: str | None) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Uppercase, strip extra whitespace, expand common abbreviations
    text = text.upper().strip()
    text = re.sub(r"\s+", " ", text)
    abbrevs = {"ST ": "STREET ", "AVE ": "AVENUE ", "BLVD ": "BOULEVARD ",
               "DR ": "DRIVE ", "RD ": "ROAD ", "LN ": "LANE "}
    for abbr, full in abbrevs.items():
        text = text.replace(abbr, full)
    return text
