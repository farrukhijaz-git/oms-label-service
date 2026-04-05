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
    """
    OCR a single-page PDF using tesseract. Returns extracted text or empty string.

    Uses 300 DPI for better character recognition on shipping labels (the default
    200 DPI causes garbled output on bold/large fonts like USPS Ground Advantage).
    PSM 3 (fully auto) is kept so tesseract can handle the mixed layout (logo,
    FROM block, SHIP TO block, barcode, product line).
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        images = convert_from_bytes(pdf_bytes, dpi=300)
        if not images:
            return ""
        text = pytesseract.image_to_string(images[0], config='--oem 3 --psm 3')
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
# Standalone "TO:" or "TO" on its own line — USPS Priority Mail labels from
# some generators (Pirateship, Stamps.com) omit "SHIP" and just say "TO:"
_TO_ANCHOR_RE = re.compile(r"^TO:?\s*$", re.IGNORECASE)
# City/state/zip — comma optional (CAPE CORAL FL 33909 or CAPE CORAL, FL 33909)
_CITY_STATE_ZIP_RE = re.compile(r"[A-Za-z][A-Za-z\s]+,?\s+[A-Z]{2}\s+\d{5}", re.IGNORECASE)

# Tracking number patterns for common carriers
_USPS_TRACKING_RE = re.compile(r"\b(9[0-9]{21,34}|[A-Z]{2}[0-9]{9}US|420[0-9]{5}[A-Z]{2}[0-9]{24})\b")
_UPS_TRACKING_RE  = re.compile(r"\b(1Z[A-Z0-9]{16})\b")
_FEDEX_TRACKING_RE = re.compile(r"\b([0-9]{12}|[0-9]{15}|[0-9]{20})\b")

# Tracking section delimiter (marks end of recipient area on the label)
_TRACKING_SECTION_RE = re.compile(r'(?:USPS\s+)?TRACKING\s*#', re.IGNORECASE)


def _is_tracking_number_line(line: str) -> bool:
    """
    Return True if the line looks like a tracking/barcode number.

    More robust than a single regex — handles OCR artifacts (trailing
    punctuation, partial characters) and various spacing formats.
    """
    cleaned = line.strip()
    if not cleaned:
        return False
    # Exact match: starts with digit, rest is digits+spaces, 15+ chars
    if re.match(r'^\d[\d\s]{14,}$', cleaned):
        return True
    # Strip leading/trailing non-alphanumeric (OCR artifacts like .|})
    stripped = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', cleaned)
    if stripped and re.match(r'^\d[\d\s]{14,}$', stripped):
        return True
    # High digit ratio: 15+ digits and >80% of non-space chars are digits
    digits = re.sub(r'\D', '', cleaned)
    non_space = cleaned.replace(' ', '')
    if len(digits) >= 15 and non_space and len(digits) / len(non_space) > 0.8:
        return True
    return False


# ── Carrier detection ─────────────────────────────────────────────────────────

def detect_carrier(text: str) -> str | None:
    """
    Detect carrier from tracking number patterns in raw PDF text.
    Returns 'usps', 'ups', 'fedex', or None.

    Collapses space-formatted tracking numbers (e.g. "9400 1234 5678 ...")
    before matching so both compact and formatted variants are detected.
    USPS is checked first because its patterns are most specific; FedEx
    is checked last because its digit-only patterns overlap with other numbers.
    """
    # Collapse spaces between digit groups (formatted tracking numbers)
    collapsed = re.sub(r'(?<=\d) +(?=\d)', '', text)

    if _USPS_TRACKING_RE.search(collapsed):
        return 'usps'
    if _UPS_TRACKING_RE.search(text):
        return 'ups'
    if _FEDEX_TRACKING_RE.search(collapsed):
        return 'fedex'
    return None


def _extract_ship_to(lines: list) -> tuple:
    """
    Anchor on SHIP TO / DELIVER TO keywords to extract recipient name + address.

    Handles four common label formats:
      A) "SHIP TO: Firstname Lastname" then address lines (single combined line)
      B) "SHIP TO:"  (empty) then name on next line then address (multi-line)
         Skips carrier service lines (e.g. "PRIORITY MAIL®") between anchor and name.
      C) "SHIP  Firstname Lastname" / "TO:  123 Main St" (USPS two-column format)
      D) Standalone "TO:" or "TO" on its own line (USPS Priority Mail / Pirateship)

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
                # Format B: name on next line.  Some USPS Priority Mail labels
                # insert a service line (e.g. "PRIORITY MAIL®") between the
                # "SHIP TO:" header and the actual recipient name.  Scan up to
                # 3 lines ahead to skip any carrier/service keyword lines.
                if i + 1 >= len(lines):
                    return None, None
                name = None
                addr_start = i + 2
                for offset in range(1, 4):
                    if i + offset >= len(lines):
                        break
                    candidate = lines[i + offset].strip()
                    if _clean_name(candidate) is not None:
                        name = candidate
                        addr_start = i + offset + 1
                        break

            address = _collect_address(lines, addr_start)
            return _clean_name(name) if name else None, address

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

        # ── Format D: standalone "TO:" / "TO" (USPS Priority Mail) ───────────
        # Some label generators (Pirateship, Stamps.com Priority Mail) omit the
        # "SHIP" prefix entirely and just print "TO:" on its own line.
        m = _TO_ANCHOR_RE.match(line)
        if m:
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
    Stops early if a new section header (FROM, PRIORITY MAIL, etc.) is found.
    """
    if start >= len(lines):
        return None

    # Stop collecting when we hit a new section header.
    # Uses \b so "FROM: Sender Name 123..." (inline) also stops the scan —
    # the previous pattern required the keyword to be alone on the line.
    _SECTION_STOP_RE = re.compile(
        r"^(FROM|RETURN\s*ADDRESS|RETURN|SHIP\s+FROM|PRIORITY\s+MAIL|EXPRESS\s+MAIL)\b",
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
        # Skip tracking/barcode number lines
        if _is_tracking_number_line(line):
            continue
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
    # Fall back to first non-empty, non-tracking-number line at start position
    for j in range(start, min(start + 5, len(lines))):
        fb = lines[j].strip()
        if fb and not _is_tracking_number_line(fb) and not _SECTION_STOP_RE.match(fb):
            return fb.upper()
    return None


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


# ── Anchor-less layout extraction ─────────────────────────────────────────────

def _extract_from_layout(lines: list) -> tuple:
    """
    Anchor-less layout extraction for labels without SHIP TO / TO: markers.

    Identifies the sender block by company keywords (LLC, Inc, etc.), then
    finds the recipient as the next person-name + address block after the
    sender, stopping before any TRACKING section.

    Designed for Pitney Bowes / USPS Priority Mail labels that lack explicit
    SHIP TO anchors.
    """
    # Find the tracking section boundary
    content_end = len(lines)
    for i, line in enumerate(lines):
        if _TRACKING_SECTION_RE.search(line):
            content_end = i
            break

    # Find sender block (line containing company keywords like LLC, Inc)
    sender_idx = None
    for i in range(content_end):
        lower_words = {w.lower().rstrip(".,") for w in lines[i].split()}
        if lower_words & COMPANY_KEYWORDS:
            sender_idx = i
            break

    if sender_idx is None:
        return None, None

    # Find end of sender address: scan for city/state/zip after sender name
    sender_end = sender_idx + 1
    for j in range(sender_idx + 1, min(sender_idx + 4, content_end)):
        if _CITY_STATE_ZIP_RE.search(lines[j]):
            sender_end = j + 1
            break

    # After sender block, find recipient name + address
    name = None
    addr_start = None
    for i in range(sender_end, content_end):
        line = lines[i].strip()
        if not line:
            continue
        # Skip tracking/barcode number lines
        if _is_tracking_number_line(line):
            continue
        # Skip short alphanumeric codes (0003, R056, C000)
        if re.match(r'^[A-Z0-9]{2,6}$', line):
            continue
        # Skip purely numeric or mostly-numeric lines
        if re.match(r'^[\d\s\-]+$', line):
            continue
        # Skip carrier keywords
        lower_words = {w.lower().rstrip(".,") for w in line.split()}
        if lower_words & CARRIER_KEYWORDS:
            continue
        # Skip other company lines
        if lower_words & COMPANY_KEYWORDS:
            continue
        # Must look like a name: 2+ words, no digits
        candidate = _clean_name(line)
        if candidate and not re.search(r'\d', candidate):
            words = candidate.split()
            if len(words) >= 2:
                name = candidate
                addr_start = i + 1
                break

    if not name or addr_start is None:
        return None, None

    # Collect address between recipient name and tracking section
    address = _collect_address(lines[:content_end], addr_start)
    return name, address


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
        # Reject single-word all-caps (service labels like "CERTIFIED", "EXPRESS").
        # Multi-word all-caps is the norm for names on shipping labels ("JOHN DOE")
        # and must NOT be rejected here — carrier keywords above already filter
        # service lines like "PRIORITY MAIL".
        if len(words) == 1 and words[0].isupper():
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
      1. First address found AFTER a SHIP TO / DELIVER TO / TO: marker
      2. Otherwise the address NOT associated with a FROM / RETURN block
      3. Otherwise the LAST address in the document (recipient follows sender)
    """
    street_pattern  = re.compile(r"^\d+\s+\w+", re.IGNORECASE)
    city_state_zip  = re.compile(r"[A-Za-z][A-Za-z\s]+,?\s+[A-Z]{2}\s+\d{5}", re.IGNORECASE)
    from_pattern    = re.compile(r"^(FROM|RETURN\s*ADDRESS|SHIP\s*FROM|RETURN)\s*:?", re.IGNORECASE)
    ship_to_pattern = re.compile(r"^(SHIP\s*TO|DELIVER\s*TO|TO:?)\s*$", re.IGNORECASE)

    from_idx    = next((i for i, l in enumerate(lines) if from_pattern.match(l)),    None)
    ship_to_idx = next((i for i, l in enumerate(lines) if ship_to_pattern.match(l)), None)

    # Build list of (line_index, full_address_string) candidates
    candidates: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        lower_words = {w.lower().rstrip(".,") for w in line.split()}
        if lower_words & CARRIER_KEYWORDS:
            continue
        # Skip tracking/barcode number lines
        if _is_tracking_number_line(line):
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

    Extraction strategy is selected based on carrier detected from tracking number:

      USPS  — single-column labels: full-page anchored → OCR supplement for
              non-standard fonts (Ground Advantage, some Priority Mail variants)
      UPS / FedEx — two-column thermal labels: right-half crop first (isolates
              SHIP TO from interleaved FROM column) → full-page anchored fallback
      Unknown — full waterfall: full-page → right-half crop → OCR supplement

    In all cases, generic heuristics are used as a last resort if anchored
    extraction finds nothing.
    """
    result = {
        "customer_name": None,
        "address": None,
        "tracking_number": None,
        "carrier": None,
        "is_image_pdf": False,
        "raw_text": "",
    }

    all_text = ""

    # Step 1: pdfplumber full-page text extraction
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
    except Exception as e:
        logger.warning(f"pdfplumber extraction error: {e}")

    # Step 2: OCR fallback for fully image-based PDFs (pdfplumber found nothing)
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
    logger.info(f"RAW LINES: {lines[:40]}")

    # Step 3: carrier detection from tracking number in raw text
    carrier = detect_carrier(all_text)
    result["carrier"] = carrier
    logger.info(f"Carrier detected: {carrier or 'unknown'}")

    name, address = None, None

    if carrier in ('ups', 'fedex'):
        # ── UPS / FedEx ───────────────────────────────────────────────────────
        # Thermal labels are almost always two-column (FROM left, SHIP TO right).
        # Right-half crop is more reliable than full-page for these carriers.
        right_text = _extract_text_right_half(pdf_bytes)
        if right_text.strip():
            right_lines = [l.strip() for l in right_text.split("\n") if l.strip()]
            name, address = _extract_ship_to(right_lines)
            if name or address:
                logger.info(f"UPS/FedEx right-half → name={name!r}, address={address!r}")
        if not (name or address):
            name, address = _extract_ship_to(lines)
            if name or address:
                logger.info(f"UPS/FedEx full-page → name={name!r}, address={address!r}")

    elif carrier == 'usps':
        # ── USPS ──────────────────────────────────────────────────────────────
        # Labels are typically single-column; anchored extraction handles all
        # USPS format variants (SHIP TO:, TO:, DELIVER TO:, two-column Ground
        # Advantage).  Layout extraction handles anchor-less labels (Pitney
        # Bowes, CommPrice USPS Priority Mail).  OCR supplement fires when
        # pdfplumber finds partial text but no recipient.
        name, address = _extract_ship_to(lines)
        if name or address:
            logger.info(f"USPS full-page → name={name!r}, address={address!r}")

        # Anchor-less layout extraction (Pitney Bowes / Priority Mail without
        # SHIP TO markers)
        if not (name and address):
            layout_name, layout_address = _extract_from_layout(lines)
            if layout_name or layout_address:
                name = name or layout_name
                address = address or layout_address
                logger.info(f"USPS layout → name={name!r}, address={address!r}")

        if not name and not result["is_image_pdf"] and all_text.strip():
            logger.info("USPS: no recipient found — running OCR supplement")
            ocr_text = _ocr_pdf_page(pdf_bytes)
            if ocr_text.strip():
                ocr_lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
                ocr_name, ocr_address = _extract_ship_to(ocr_lines)
                if ocr_name or ocr_address:
                    name, address = ocr_name, ocr_address
                    lines = ocr_lines
                    logger.info(f"USPS OCR supplement → name={name!r}, address={address!r}")
                else:
                    # Try layout extraction on OCR text
                    ocr_name, ocr_address = _extract_from_layout(ocr_lines)
                    if ocr_name or ocr_address:
                        name = name or ocr_name
                        address = address or ocr_address
                        lines = ocr_lines
                        logger.info(f"USPS OCR layout → name={name!r}, address={address!r}")
                    else:
                        lines = ocr_lines
                        logger.info("USPS OCR: no match, using OCR lines for heuristics")

    else:
        # ── Unknown carrier: full waterfall ───────────────────────────────────
        name, address = _extract_ship_to(lines)
        if not (name and address):
            layout_name, layout_address = _extract_from_layout(lines)
            if layout_name or layout_address:
                name = name or layout_name
                address = address or layout_address
                logger.info(f"Unknown layout → name={name!r}, address={address!r}")
        if not (name or address) and not result["is_image_pdf"]:
            right_text = _extract_text_right_half(pdf_bytes)
            if right_text.strip():
                right_lines = [l.strip() for l in right_text.split("\n") if l.strip()]
                name, address = _extract_ship_to(right_lines)
                if name or address:
                    logger.info(f"Unknown right-half → name={name!r}, address={address!r}")

        if not name and not result["is_image_pdf"] and all_text.strip():
            logger.info("Unknown carrier: no recipient — running OCR supplement")
            ocr_text = _ocr_pdf_page(pdf_bytes)
            if ocr_text.strip():
                ocr_lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
                ocr_name, ocr_address = _extract_ship_to(ocr_lines)
                if ocr_name or ocr_address:
                    name, address = ocr_name, ocr_address
                    lines = ocr_lines
                    logger.info(f"Unknown OCR supplement → name={name!r}, address={address!r}")
                else:
                    ocr_name, ocr_address = _extract_from_layout(ocr_lines)
                    if ocr_name or ocr_address:
                        name = name or ocr_name
                        address = address or ocr_address
                        lines = ocr_lines
                        logger.info(f"Unknown OCR layout → name={name!r}, address={address!r}")
                    else:
                        lines = ocr_lines

    if name or address:
        logger.info(f"Anchored extraction result → name={name!r}, address={address!r}")
        result["customer_name"] = name
        result["address"] = address
    else:
        # Last resort: generic heuristics (FROM-aware address preference)
        logger.info("No anchored result — using generic heuristics")
        result["customer_name"] = _extract_name(lines)
        result["address"] = _extract_address(lines)

    # Validate address is not actually a tracking number
    if result["address"] and _is_tracking_number_line(result["address"].replace(', ', ' ')):
        logger.info(f"Rejecting tracking-number-like address: {result['address']!r}")
        result["address"] = None
        # Try to recover address via layout extraction
        _, layout_addr = _extract_from_layout(lines)
        if layout_addr:
            result["address"] = layout_addr
            logger.info(f"Recovered address via layout: {layout_addr!r}")

    # Step 5: tracking number extraction (always use richest line set available)
    result["tracking_number"] = _extract_tracking_number(lines)
    if result["tracking_number"]:
        logger.info(f"Tracking number: {result['tracking_number']}")

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
