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
    """
    Pull street + city/state/zip starting at lines[start].
    Scans up to 5 lines so apt/suite lines don't break extraction.
    Stops early if a new section header (FROM, PRIORITY, etc.) is found.
    """
    if start >= len(lines):
        return None

    _SECTION_STOP_RE = re.compile(
        r"^(FROM|RETURN|SHIP\s+FROM|PRIORITY|EXPRESS|GROUND|UPS|USPS|FEDEX)\s*:?\s*$",
        re.IGNORECASE,
    )

    street = None
    city_zip = None

    for i in range(start, min(start + 5, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        if _SECTION_STOP_RE.match(line):
            break
        if re.match(r"^\d+\s", line) and street is None:
            street = line
        if _CITY_STATE_ZIP_RE.search(line) and city_zip is None:
            city_zip = line
        if street and city_zip:
            break

    if street and city_zip:
        return f"{street}, {city_zip}".upper()
    if street:
        return street.upper()
    # Fall back to first non-empty line at start position
    return lines[start].strip().upper() if start < len(lines) else None


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


def _extract_tracking_number(lines: list) -> str | None:
    """
    Extract the first recognisable carrier tracking number from the label text.

    Handles both compact numbers ("9334610990150168420911") and the common
    formatted/spaced layout used on printed USPS labels
    ("9334 6109 9015 0168 4209 11") by collapsing whitespace on lines that
    consist entirely of digits and spaces before running the regexes.
    """
    for line in lines:
        # Build a list of candidates: original line, plus a space-collapsed
        # version when the line is made up exclusively of digits and spaces
        # (formatted tracking number like "9334 6109 9015 0168 4209 11").
        candidates = [line]
        collapsed = re.sub(r"\s+", "", line)
        if collapsed != line and re.match(r"^\d{15,}$", collapsed):
            candidates.append(collapsed)

        for candidate in candidates:
            # USPS — most specific first to avoid false-positives
            m = _USPS_TRACKING_RE.search(candidate)
            if m:
                return m.group(1)
            m = _UPS_TRACKING_RE.search(candidate)
            if m:
                return m.group(1)
            # FedEx last — generic digit patterns overlap with USPS barcodes
            m = _FEDEX_TRACKING_RE.search(candidate)
            if m:
                return m.group(1)
    return None


def _extract_address(lines: list) -> str | None:
    """
    Fallback address extraction used when no SHIP TO anchor is found.

    Collects ALL (street, city/state/zip) pairs in the document, then returns
    the one most likely to be the recipient:
      1. First address found AFTER a SHIP TO / DELIVER TO marker
      2. Otherwise the address NOT associated with a FROM / RETURN block
      3. Otherwise the LAST address in the document (recipient follows sender)
    """
    street_pattern  = re.compile(r"^\d+\s+\w+", re.IGNORECASE)
    city_state_zip  = re.compile(r"[A-Za-z][A-Za-z\s]+,?\s+[A-Z]{2}\s+\d{5}", re.IGNORECASE)
    from_pattern    = re.compile(r"^(FROM|RETURN\s*ADDRESS|SHIP\s*FROM|RETURN)\s*:?", re.IGNORECASE)
    ship_to_pattern = re.compile(r"^(SHIP\s*TO|DELIVER\s*TO)\s*:?", re.IGNORECASE)

    from_idx    = next((i for i, l in enumerate(lines) if from_pattern.match(l)),    None)
    ship_to_idx = next((i for i, l in enumerate(lines) if ship_to_pattern.match(l)), None)

    # Build list of (line_index, full_address_string) candidates
    candidates: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        lower_words = {w.lower().rstrip(".,") for w in line.split()}
        if lower_words & CARRIER_KEYWORDS:
            continue
        if not street_pattern.match(line):
            continue
        # Look ahead up to 4 lines for a city/state/zip
        city_line = None
        for j in range(i + 1, min(i + 5, len(lines))):
            if city_state_zip.search(lines[j]):
                city_line = lines[j]
                break
        addr = f"{line}, {city_line}".upper() if city_line else line.upper()
        candidates.append((i, addr))

    if not candidates:
        return None

    # Preference 1: address after SHIP TO marker
    if ship_to_idx is not None:
        after_ship = [(idx, a) for idx, a in candidates if idx > ship_to_idx]
        if after_ship:
            return after_ship[0][1]

    # Preference 2: address outside the FROM block (±4 lines around from_idx)
    if from_idx is not None:
        not_from = [(idx, a) for idx, a in candidates
                    if idx < from_idx - 1 or idx > from_idx + 4]
        if not_from:
            return not_from[-1][1]  # last non-FROM address = recipient

    # Preference 3: last address in document (recipient typically follows sender)
    return candidates[-1][1] if len(candidates) > 1 else candidates[0][1]


# ── Right-half crop extraction ────────────────────────────────────────────────

def _extract_text_right_half(pdf_bytes: bytes) -> str:
    """
    Extract text from the right ~58% of the page using pdfplumber crop.

    Many thermal label formats (UPS, FedEx, USPS commercial) are two-column:
    FROM on the left, SHIP TO on the right.  pdfplumber's default full-page
    extract_text() interleaves both columns by Y-position, which can surface
    the sender's address before the recipient's.  Cropping to the right half
    isolates the recipient block so the anchored SHIP TO regex has a clean
    input to work with.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # 42% from left keeps a small margin — catches labels where the
                # SHIP TO block starts slightly left of centre.
                mid_x = page.width * 0.42
                right = page.crop((mid_x, 0, page.width, page.height))
                text = right.extract_text()
                if text and text.strip():
                    logger.info(f"Right-half crop extracted {len(text)} chars")
                    return text
    except Exception as e:
        logger.warning(f"Right-half pdfplumber extraction failed: {e}")
    return ""


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

    # Step 3: anchored SHIP TO extraction on full-page text
    name, address = _extract_ship_to(lines)

    # Step 3b: if no SHIP TO found, try right-half crop (two-column label formats).
    # UPS / FedEx / USPS commercial labels often have FROM on the left and SHIP TO
    # on the right; full-page extraction interleaves them by Y-position.
    if not (name or address) and not result["is_image_pdf"]:
        right_text = _extract_text_right_half(pdf_bytes)
        if right_text.strip():
            right_lines = [l.strip() for l in right_text.split("\n") if l.strip()]
            name, address = _extract_ship_to(right_lines)
            if name or address:
                logger.info(f"Right-half SHIP TO → name={name!r}, address={address!r}")
                # Keep full lines for tracking extraction (tracking may be on left half)
            else:
                logger.info("Right-half extraction: no SHIP TO found, will use heuristics")

    if name or address:
        logger.info(f"Anchored extraction → name={name!r}, address={address!r}")
        result["customer_name"] = name
        result["address"] = address
    else:
        # Step 4: generic heuristics — now FROM-aware and prefers last/recipient address
        logger.info("No SHIP TO section found — using generic heuristics")
        result["customer_name"] = _extract_name(lines)
        result["address"] = _extract_address(lines)

    # Step 5: extract tracking number (always use full-page lines for best coverage)
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
