import os
import httpx
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from app.middleware.auth import get_current_user, require_admin
from app.services.pdf_extractor import extract_label_data, split_pdf_pages
from app.services.fuzzy_matcher import find_best_match
from app.services.storage import upload_pdf, get_signed_url
from datetime import datetime, timezone

router = APIRouter()

async def get_open_orders(orders_service_url: str, user_id: str) -> list:
    """Fetch open orders from the orders service for matching."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{orders_service_url}/orders",
                params={"status": "new,label_generated,inventory_ordered,packed,ready", "limit": 200},
                headers={"X-User-Id": user_id, "X-User-Role": "staff"},
                timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("orders", [])
        except Exception as e:
            print(f"Error fetching orders: {e}")
    return []


# POST /labels/upload
@router.post("/upload")
async def upload_labels(request: Request, files: List[UploadFile] = File(...)):
    user = get_current_user(request)
    db = request.app.state.db
    orders_service_url = os.environ.get("ORDERS_SERVICE_URL", "http://localhost:3002")

    # Fetch open orders once for all files
    open_orders = await get_open_orders(orders_service_url, user["user_id"])

    results = []

    for file in files:
        pdf_bytes = await file.read()
        filename = file.filename or "upload.pdf"

        # Debatch: split multi-page PDFs into individual pages so each label
        # gets its own storage record, extraction, and auto-match attempt.
        try:
            pages = split_pdf_pages(pdf_bytes)
        except Exception as e:
            print(f"PDF split failed for {filename}, treating as single page: {e}")
            pages = [pdf_bytes]

        for page_num, page_bytes in enumerate(pages):
            # Use p1_, p2_… prefix only when there are multiple pages
            page_filename = f"p{page_num + 1}_{filename}" if len(pages) > 1 else filename

            try:
                # Upload individual page to storage
                storage_path = upload_pdf(page_bytes, page_filename)

                # Extract text from this page
                extracted = extract_label_data(page_bytes)

                if extracted["is_image_pdf"]:
                    match_result = {
                        "matched_order_id": None,
                        "confidence": 0.0,
                        "match_status": "unmatched",
                        "top_candidates": [],
                    }
                else:
                    match_result = find_best_match(extracted, open_orders)

                # Insert one row per page
                row = await db.fetchrow(
                    """
                    INSERT INTO labels.shipping_labels
                      (storage_path, original_filename, extracted_name, extracted_address,
                       tracking_number, match_confidence, match_status, order_id, uploaded_by)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id, match_status, match_confidence, order_id
                    """,
                    storage_path,
                    page_filename,
                    extracted.get("customer_name"),
                    extracted.get("address"),
                    extracted.get("tracking_number"),
                    match_result["confidence"],
                    match_result["match_status"],
                    match_result.get("matched_order_id"),
                    user["user_id"],
                )

                results.append({
                    "filename": page_filename,
                    "label_id": str(row["id"]),
                    "extracted_name": extracted.get("customer_name"),
                    "extracted_address": extracted.get("address"),
                    "tracking_number": extracted.get("tracking_number"),
                    "is_image_pdf": extracted["is_image_pdf"],
                    "match_status": row["match_status"],
                    "match_confidence": float(row["match_confidence"]) if row["match_confidence"] else 0.0,
                    "matched_order_id": str(row["order_id"]) if row["order_id"] else None,
                    "top_candidates": match_result.get("top_candidates", []),
                })
            except Exception as e:
                print(f"Error processing {page_filename}: {e}")
                results.append({"filename": page_filename, "error": str(e)})

    return {"results": results}


# GET /labels/queue - pending labels
@router.get("/queue")
async def get_queue(request: Request):
    user = get_current_user(request)
    db = request.app.state.db

    rows = await db.fetch(
        """
        SELECT l.id, l.original_filename, l.extracted_name, l.extracted_address,
               l.tracking_number, l.match_confidence, l.match_status, l.order_id, l.uploaded_at,
               o.customer_name as matched_customer_name, o.external_id as matched_order_external_id
        FROM labels.shipping_labels l
        LEFT JOIN orders.orders o ON l.order_id = o.id
        WHERE l.match_status = 'pending'
        ORDER BY l.uploaded_at DESC
        """,
    )

    return {"labels": [dict(r) for r in rows]}


# GET /labels/unmatched - unmatched labels
@router.get("/unmatched")
async def get_unmatched(request: Request):
    user = get_current_user(request)
    db = request.app.state.db

    rows = await db.fetch(
        """
        SELECT id, original_filename, extracted_name, extracted_address,
               tracking_number, match_confidence, match_status, order_id, uploaded_at
        FROM labels.shipping_labels
        WHERE match_status = 'unmatched'
        ORDER BY uploaded_at DESC
        """,
    )

    return {"labels": [dict(r) for r in rows]}


# POST /labels/{label_id}/confirm
@router.post("/{label_id}/confirm")
async def confirm_label(label_id: str, request: Request):
    user = get_current_user(request)
    db = request.app.state.db
    body = await request.json()
    order_id = body.get("order_id")

    if not order_id:
        raise HTTPException(status_code=400, detail="order_id is required")

    orders_service_url = os.environ.get("ORDERS_SERVICE_URL", "http://localhost:3002")

    async with db.transaction():
        # Update label
        label = await db.fetchrow(
            """
            UPDATE labels.shipping_labels
            SET match_status = 'confirmed', order_id = $1,
                confirmed_by = $2, confirmed_at = now()
            WHERE id = $3
            RETURNING id, storage_path, tracking_number
            """,
            order_id, user["user_id"], label_id,
        )

        if not label:
            raise HTTPException(status_code=404, detail="Label not found")

    # Update order with label_id and tracking_number, then set status to label_generated
    try:
        async with httpx.AsyncClient() as client:
            # First update the order with label_id and tracking_number
            update_data = {"label_id": label_id}
            if label["tracking_number"]:
                update_data["tracking_number"] = label["tracking_number"]
            
            await client.patch(
                f"{orders_service_url}/orders/{order_id}",
                json=update_data,
                headers={"X-User-Id": user["user_id"], "X-User-Role": user["role"] or "staff"},
                timeout=10.0
            )
            # Then update status
            await client.patch(
                f"{orders_service_url}/orders/{order_id}/status",
                json={"status": "label_generated", "note": "Label confirmed"},
                headers={"X-User-Id": user["user_id"], "X-User-Role": user["role"] or "staff"},
                timeout=10.0
            )
    except Exception as e:
        print(f"Failed to update order: {e}")

    return {"ok": True, "label_id": label_id, "order_id": order_id}


# PATCH /labels/{label_id}/assign - manually assign label to order
@router.patch("/{label_id}/assign")
async def assign_label(label_id: str, request: Request):
    user = get_current_user(request)
    db = request.app.state.db
    orders_service_url = os.environ.get("ORDERS_SERVICE_URL", "http://localhost:3002")
    
    body = await request.json()
    order_id = body.get("order_id")

    if not order_id:
        raise HTTPException(status_code=400, detail="order_id is required")

    label = await db.fetchrow(
        """
        UPDATE labels.shipping_labels
        SET match_status = 'manually_assigned', order_id = $1,
            confirmed_by = $2, confirmed_at = now()
        WHERE id = $3
        RETURNING id, tracking_number
        """,
        order_id, user["user_id"], label_id,
    )

    if not label:
        raise HTTPException(status_code=404, detail="Label not found")

    # Update order with label_id and tracking_number
    try:
        async with httpx.AsyncClient() as client:
            update_data = {"label_id": label_id}
            if label["tracking_number"]:
                update_data["tracking_number"] = label["tracking_number"]
            
            await client.patch(
                f"{orders_service_url}/orders/{order_id}",
                json=update_data,
                headers={"X-User-Id": user["user_id"], "X-User-Role": user["role"] or "staff"},
                timeout=10.0
            )
    except Exception as e:
        print(f"Failed to update order with label_id: {e}")

    return {"ok": True, "label_id": label_id, "order_id": order_id}


# GET /labels/{label_id}/download - get signed URL
@router.get("/{label_id}/download")
async def download_label(label_id: str, request: Request):
    user = get_current_user(request)
    db = request.app.state.db

    row = await db.fetchrow(
        "SELECT storage_path FROM labels.shipping_labels WHERE id = $1",
        label_id,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Label not found")

    try:
        signed_url = get_signed_url(row["storage_path"], expires_in=3600)
        return {"url": signed_url, "expires_in": 3600}
    except Exception as e:
        print(f"Error generating signed URL: {e}")
        raise HTTPException(status_code=500, detail="Could not generate download URL")
