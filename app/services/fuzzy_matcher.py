from rapidfuzz import fuzz
from app.services.pdf_extractor import normalize_text

CONFIDENCE_THRESHOLD = 0.80

def find_best_match(extracted: dict, open_orders: list) -> dict:
    """
    Find the best matching order for extracted label data.

    extracted: { customer_name, address }
    open_orders: list of { id, customer_name, address_line1, address_line2, city, state, zip }

    Returns: { matched_order_id, confidence, match_status, top_candidates }
    """
    if not open_orders:
        return {
            "matched_order_id": None,
            "confidence": 0.0,
            "match_status": "unmatched",
            "top_candidates": [],
        }

    ext_name = normalize_text(extracted.get("customer_name") or "")
    ext_addr = normalize_text(extracted.get("address") or "")

    candidates = []

    for order in open_orders:
        ord_name = normalize_text(order.get("customer_name") or "")
        # Build full address string
        parts = [order.get("address_line1") or "", order.get("city") or "",
                 order.get("state") or "", order.get("zip") or ""]
        ord_addr = normalize_text(", ".join(p for p in parts if p))

        # Name score (WRatio handles partial matching well)
        if ext_name and ord_name:
            name_score = fuzz.WRatio(ext_name, ord_name) / 100.0
        else:
            name_score = 0.0

        # Address score (token_sort_ratio handles word order variations)
        if ext_addr and ord_addr:
            addr_score = fuzz.token_sort_ratio(ext_addr, ord_addr) / 100.0
        else:
            addr_score = 0.0

        # Weighted confidence: 40% name, 60% address
        if ext_name and ext_addr:
            confidence = (name_score * 0.4) + (addr_score * 0.6)
        elif ext_name:
            confidence = name_score
        elif ext_addr:
            confidence = addr_score
        else:
            confidence = 0.0

        candidates.append({
            "order_id": str(order["id"]),
            "customer_name": order.get("customer_name"),
            "confidence": round(confidence, 3),
            "name_score": round(name_score, 3),
            "addr_score": round(addr_score, 3),
        })

    # Sort by confidence descending
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    top_candidates = candidates[:5]

    best = top_candidates[0] if top_candidates else None

    if best and best["confidence"] >= CONFIDENCE_THRESHOLD:
        return {
            "matched_order_id": best["order_id"],
            "confidence": best["confidence"],
            "match_status": "pending",
            "top_candidates": top_candidates,
        }
    else:
        return {
            "matched_order_id": None,
            "confidence": best["confidence"] if best else 0.0,
            "match_status": "unmatched",
            "top_candidates": top_candidates,
        }
