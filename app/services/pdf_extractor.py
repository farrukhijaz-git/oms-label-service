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
        text = pytesseract.image_to_string(images[0])
        logger.info(f"OCR extracted {len(text)} chars")
        return text
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


# ── Keyword sets for name extraction fallback ──────────────────────────────────

STREET_SUFFIXES = {
    "st", "street", "ave", "avenue", "blvd", "boulevard", "dr", "drive",
    "rd", "road", "ln", "lane", "ct", "court", "way", "pl", "place",
    "ter", "terrace", "cir", "circle", "pkwy", "parkway", "hwy", "highway",
}
COMPANY_KEYWORDS = {"llc", "inc", "corp", "ltd", "co", "company", "services", "group"}

# Lines containing any of these are definitely NOT a recipient name
CARRIER_KEYWORDS = {
    "usps", "ups", "fedex", "dhl", "stamps", "postage", "tracking",
    "priority", "ground", "express", "advantage", "delivery", "shipping",
    "trademark", "mailed", "barcode", "e-postage", "first", "class",
    "media", "parcel", "retail", "commercial", "presorted", "certified",
}

# ── "SHIP TO" anchored extraction ─────────────────────────────────────────────

# Matches: "SHIP TO: Name", "SHIP TO:", "SHIP Name" (USPS two-column format),
#          "DELIVER TO:", "TO:" (when it appears after SHIP on prior line)
_SHIP_TO_RE   = re.compile(r"^SHIP\s+TO:?\s*(.*)", re.IGNORECASE)
_SHIP_NAME_RE = re.compile(r"^SHIP\s+(?!TO)(.+)", re.IGNORECASE)
_TO_RE        = re.compile(r"^TO:\s*(.*)", re.IGNORECASE)
_DELIVER_RE   = re.compile(r"^DELIVER\s+TO:?\s*(.*)", re.IGNORECASE)
# City/state/zip — comma optional (CAPE CORAL FL 33909 or CAPE CORAL, FL 33909)
_CITY_STATE_ZIP_RE = re.compile(r"[A-Za-z][A-Za-z\s]+,?\s+[A-Z]{2}\s+\d{5}", re.IGNORECASE)

# Tracking number patterns for common carriers
_USPS_TRACKING_RE = re.compile(r"\b(9[0-9]{21,34}|[A-Z]{2}[0-9]{9}US|420[0-9]{5}[A-Z]{2}[0-9]{24})\b")
_UPS_TRACKING_RE = re.compile(r"\b(1Z[A-Z0-9]{16})\b")
_FEDEX_TRACKING_RE = re.compile(r"\b([0-9]{12}|[0-9]{15}|[0-9]{20})\b")


def _extract_ship_to(lines: list) -> tuple:
    """
    Anchor on SHIP TO / DELIVER TO keywords to extract recipient name + address.

    Handles three common label formats:
      A) "SHIP TO: Firstname Lastname" then address lines (single combined line)
      B) "SHIP TO:"  (empty) then name on next line then address (multi-line)
      C) "SHIP  Firstname Lastname" / "TO:  123 Main St" (USPS two-column format)

    Returns (name, address) or (None, None) if no shipping section found.
    """
    for i, line in enumerate(lines):

        # ── Format A & B: "SHIP TO:" ──────────────────────────────────────────
        m = _SHIP_TO_RE.match(line)
        if m:
            remainder = m.group(1).strip()
            if remainder and not _looks_like_address(remainder):
                # Format A: name on same line
                name = remainder
                addr_start = i + 1
            else:
                # Format B: name on next line
                if i + 1 >= len(lines):
                    return None, None
                name = lines[i + 1].strip()
                addr_start = i + 2

            address = _collect_address(lines, addr_start)
            return _clean_name(name), address

        # ── Format C: "SHIP  Antonia Pantoja Vidal" (USPS two-column) ────────
        m = _SHIP_NAME_RE.match(line)
        if m:
            name = m.group(1).strip()
            # Next line should be "TO: <address>" or just the address
            if i + 1 < len(lines):
                to_m = _TO_RE.match(lines[i + 1])
                if to_m:
                    addr_line1 = to_m.group(1).strip()
                    addr_line2 = lines[i + 2].strip() if i + 2 < len(lines) else ""
                else:
                    addr_line1 = lines[i + 1].strip()
                    addr_line2 = lines[i + 2].strip() if i + 2 < len(lines) else ""

                address = _build_address(addr_line1, addr_line2)
                return _clean_name(name), address

        # ── DELIVER TO ────────────────────────────────────────────────────────
        m = _DELIVER_RE.match(line)
        if m:
            remainder = m.group(1).strip()
            if remainder and not _looks_like_address(remainder):
                name = remainder
                addr_start = i + 1
            else:
                if i + 1 >= len(lines):
                    return None, None
                name = lines[i + 1].strip()
                addr_start = i + 2

            address = _collect_address(lines, addr_start)
            return _clean_name(name), address

    return None, None


def _looks_like_address(text: str) -> bool:
    """Return True if text starts with a house number."""
    return bool(re.match(r"^\d+\s", text))


def _collect_address(lines: list, start: int) -> str | None:
    """Pull address_line1 + city/state/zip starting at lines[start]."""
    if start >= len(lines):
        return None
    addr_line1 = lines[start].strip()
    addr_line2 = lines[start + 1].strip() if start + 1 < len(lines) else ""
    return _build_address(addr_line1, addr_line2)


def _build_address(line1: str, line2: str) -> str | None:
    """Combine street line + city/state/zip into a single address string."""
    parts = []
    if line1 and re.match(r"^\d+", line1):
        parts.append(line1)
    # line2 might be city/state/zip
    if line2 and (_CITY_STATE_ZIP_RE.search(line2) or re.match(r"^\d+", line2)):
        parts.append(line2)
    return ", ".join(parts).upper() if parts else (line1.upper() if line1 else None)


def _clean_name(name: str) -> str | None:
    """Strip any leading 'TO:' artifact and return None if clearly not a name."""
    if not name:
        return None
    name = re.sub(r"^TO:\s*", "", name, flags=re.IGNORECASE).strip()
    # Reject if it looks like a tracking number, barcode, or service line
    if re.match(r"^[\d\s\-]+$", name):
        return None
    lower_words = {w.lower().rstrip(".,") for w in name.split()}
    if lower_words & CARRIER_KEYWORDS:
        return None
    return name or None


# ── Generic fallback extraction ───────────────────────────────────────────────

def _extract_name(lines: list) -> str | None:
    """
    Fallback name extraction used only when no SHIP TO section is found.
    Skips carrier/service keyword lines and address-like lines.
    """
    for line in lines:
        if re.search(r"\d", line):
            continue

        words = line.split()
        if not 2 <= len(words) <= 5:
            continue

        if not all(w[0].isupper() for w in words if w):
            continue

        lower_words = {w.lower().rstrip(".,") for w in words}
        if lower_words & STREET_SUFFIXES:
            continue
        if lower_words & COMPANY_KEYWORDS:
            continue
        if lower_words & CARRIER_KEYWORDS:
            continue
        # Skip all-caps single-word repeated patterns (barcodes, service names)
        if all(w.isupper() for w in words):
            continue

        return line

    return None


def _extract_address(lines: list) -> str | None:
    """Fallback address extraction."""
    street_pattern = re.compile(r"^\d+\s+\w+", re.IGNORECASE)
    # Comma optional: "CAPE CORAL FL 33909" or "Cape Coral, FL 33909"
    city_state_zip = re.compile(r"[A-Za-z][A-Za-z\s]+,?\s+[A-Z]{2}\s+\d{5}", re.IGNORECASE)

    street_line = None
    city_line = None

    for line in lines:
        # Skip lines that mention carrier/service names
        lower_words = {w.lower().rstrip(".,") for w in line.split()}
        if lower_words & CARRIER_KEYWORDS:
            continue
        if street_pattern.match(line) and not street_line:
            street_line = line
        if city_state_zip.search(line) and not city_line:
            city_line = line

    if street_line and city_line:
        return f"{street_line}, {city_line}".upper()
    elif street_line:
        return street_line.upper()

    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_label_data(pdf_bytes: bytes) -> dict:
    """
    Extract customer name, address, and tracking number from a single-page PDF.

    Strategy:
      1. pdfplumber text extraction
      2. OCR fallback for image-based PDFs
      3. SHIP TO / DELIVER TO anchored extraction (covers USPS, UPS, FedEx)
      4. Generic heuristic fallback if no shipping section found
      5. Tracking number extraction
    """
    result = {
        "customer_name": None,
        "address": None,
        "tracking_number": None,
        "is_image_pdf": False,
        "raw_text": "",
    }

    all_text = ""

    # Step 1: pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
    except Exception as e:
        logger.warning(f"pdfplumber extraction error: {e}")

    # Step 2: OCR fallback
    if not all_text.strip():
        result["is_image_pdf"] = True
        logger.info("No text from pdfplumber — attempting OCR")
        all_text = _ocr_pdf_page(pdf_bytes)
        if not all_text.strip():
            logger.warning("OCR also returned no text")
            return result
        result["is_image_pdf"] = False

    result["raw_text"] = all_text
    lines = [line.strip() for line in all_text.split("\n") if line.strip()]
    logger.info(f"Extracted {len(lines)} lines from PDF")
    logger.debug(f"Lines: {lines[:30]}")

    # Step 3: anchored SHIP TO extraction
    name, address = _extract_ship_to(lines)

    if name or address:
        logger.info(f"Anchored extraction → name={name!r}, address={address!r}")
        result["customer_name"] = name
        result["address"] = address
    else:
        # Step 4: generic heuristics
        logger.info("No SHIP TO section found — using generic heuristics")
        result["customer_name"] = _extract_name(lines)
        result["address"] = _extract_address(lines)

    # Step 5: extract tracking number
    result["tracking_number"] = _extract_tracking_number(lines)
    if result["tracking_number"]:
        logger.info(f"Extracted tracking number: {result['tracking_number']}")

    return result


def normalize_text(text: str | None) -> str:
    """Normalize text for fuzzy comparison."""
    if not text:
        return ""
    text = text.upper().strip()
    text = re.sub(r"\s+", " ", text)
    abbrevs = {
        "ST ": "STREET ", "AVE ": "AVENUE ", "BLVD ": "BOULEVARD ",
        "DR ": "DRIVE ", "RD ": "ROAD ", "LN ": "LANE ",
        "TER ": "TERRACE ", "CIR ": "CIRCLE ", "CT ": "COURT ",
    }
    for abbr, full in abbrevs.items():
        text = text.replace(abbr, full)
    return text
