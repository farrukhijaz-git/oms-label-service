from rapidfuzz import fuzz
from app.services.pdf_extractor import normalize_text

CONFIDENCE_THRESHOLD = 0.80


def find_best_match(extracted: dict, open_orders: list) -> dict:
    """
    Find the best matching order for extracted label data.

    extracted: { customer_name, address, tracking_number }
    open_orders: list of { id, customer_name, address_line1, address_line2,
                           city, state, zip, tracking_number }

    Returns: { matched_order_id, confidence, match_status, top_candidates,
               tracking_match_order_id (optional) }

    match_status values:
      'pending'           – confident match, needs human review/confirm
      'unmatched'         – no confident match found
      'tracking_conflict' – tracking number points to a different order than
                           name/address matching does; needs explicit resolution
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
    ext_tracking = (extracted.get("tracking_number") or "").strip().upper()

    # --- Step 1: find which order (if any) already owns this tracking number ---
    tracking_owner_id = None
    if ext_tracking:
        for order in open_orders:
            ord_tracking = (order.get("tracking_number") or "").strip().upper()
            if ord_tracking and ord_tracking == ext_tracking:
                tracking_owner_id = str(order["id"])
                break

    # --- Step 2: score every order on name + address ---
    candidates = []

    for order in open_orders:
        ord_name = normalize_text(order.get("customer_name") or "")
        parts = [
            order.get("address_line1") or "",
            order.get("city") or "",
            order.get("state") or "",
            order.get("zip") or "",
        ]
        ord_addr = normalize_text(", ".join(p for p in parts if p))
        ord_tracking = (order.get("tracking_number") or "").strip().upper()
        order_id = str(order["id"])

        # Name score (WRatio handles partial matching well)
        name_score = fuzz.WRatio(ext_name, ord_name) / 100.0 if (ext_name and ord_name) else 0.0

        # Address score (token_sort_ratio handles word-order variations)
        addr_score = fuzz.token_sort_ratio(ext_addr, ord_addr) / 100.0 if (ext_addr and ord_addr) else 0.0

        # Weighted base confidence: 40% name, 60% address
        if ext_name and ext_addr:
            base_confidence = (name_score * 0.4) + (addr_score * 0.6)
        elif ext_name:
            base_confidence = name_score
        elif ext_addr:
            base_confidence = addr_score
        else:
            base_confidence = 0.0

        # Tracking bonus: exact tracking match boosts confidence
        tracking_exact = bool(ext_tracking and ord_tracking and ext_tracking == ord_tracking)
        if tracking_exact:
            # Add up to +0.20 boost; floor at 0.70 so it always enters review queue
            confidence = max(0.70, min(1.0, base_confidence + 0.20))
        else:
            confidence = base_confidence

        candidates.append({
            "order_id": order_id,
            "customer_name": order.get("customer_name"),
            "confidence": round(confidence, 3),
            "name_score": round(name_score, 3),
            "addr_score": round(addr_score, 3),
            "tracking_match": tracking_exact,
        })

    # Sort by confidence descending
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    top_candidates = candidates[:5]
    best = top_candidates[0] if top_candidates else None

    # --- Step 3: classify result ---

    if best and best["confidence"] >= CONFIDENCE_THRESHOLD:
        # Check for tracking conflict: tracking says order A, name/addr says order B
        if (
            tracking_owner_id
            and best["order_id"] != tracking_owner_id
            and ext_tracking
        ):
            return {
                "matched_order_id": best["order_id"],
                "confidence": best["confidence"],
                "match_status": "tracking_conflict",
                "top_candidates": top_candidates,
                "tracking_match_order_id": tracking_owner_id,
            }

        return {
            "matched_order_id": best["order_id"],
            "confidence": best["confidence"],
            "match_status": "pending",
            "top_candidates": top_candidates,
        }

    # Below threshold — check if tracking alone gives us a candidate to surface
    if tracking_owner_id:
        tracking_candidate = next(
            (c for c in candidates if c["order_id"] == tracking_owner_id), None
        )
        if tracking_candidate:
            # Surface it in review queue even though name/addr score is low
            return {
                "matched_order_id": tracking_owner_id,
                "confidence": tracking_candidate["confidence"],
                "match_status": "pending",
                "top_candidates": top_candidates,
            }

    return {
        "matched_order_id": None,
        "confidence": best["confidence"] if best else 0.0,
        "match_status": "unmatched",
        "top_candidates": top_candidates,
    }
