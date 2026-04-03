import os
import uuid
import asyncio
import httpx
from fastapi import APIRouter, BackgroundTasks, Request, UploadFile, File, HTTPException
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


# ---------------------------------------------------------------------------
# Background processing helper
# ---------------------------------------------------------------------------

async def _process_upload_job(
    job_id: str,
    files_data: list,       # list of (filename: str, pdf_bytes: bytes)
    user_id: str,
    db_pool,                # asyncpg pool
    orders_service_url: str,
    jobs_store: dict,       # reference to app.state.upload_jobs
):
    """Run in background: OCR, extract, match, insert — one page at a time."""
    job = jobs_store[job_id]
    try:
        # Fetch open orders once for all files (happens in background)
        open_orders = await get_open_orders(orders_service_url, user_id)

        results = []
        file_index = 0

        for filename, pdf_bytes in files_data:
            file_index += 1
            job["current_file"] = filename
            job["current"] = file_index

            # Split pages (CPU-bound → run in thread)
            try:
                pages = await asyncio.to_thread(split_pdf_pages, pdf_bytes)
            except Exception as e:
                print(f"PDF split failed for {filename}: {e}")
                pages = [pdf_bytes]

            for page_num, page_bytes in enumerate(pages):
                page_filename = f"p{page_num + 1}_{filename}" if len(pages) > 1 else filename

                try:
                    # Upload to storage (network IO — run in thread for supabase SDK)
                    storage_path = await asyncio.to_thread(upload_pdf, page_bytes, page_filename)

                    # Extract text / OCR (CPU-bound)
                    extracted = await asyncio.to_thread(extract_label_data, page_bytes)

                    if extracted["is_image_pdf"]:
                        match_result = {
                            "matched_order_id": None,
                            "confidence": 0.0,
                            "match_status": "unmatched",
                            "top_candidates": [],
                        }
                    else:
                        match_result = find_best_match(extracted, open_orders)

                    row = await db_pool.fetchrow(
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
                        user_id,
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
                        "tracking_conflict_order_id": match_result.get("tracking_match_order_id"),
                    })
                except Exception as e:
                    print(f"Error processing {page_filename}: {e}")
                    results.append({"filename": page_filename, "error": str(e)})

        job["status"] = "done"
        job["results"] = results

    except Exception as e:
        print(f"Upload job {job_id} failed: {e}")
        job["status"] = "error"
        job["error"] = str(e)


# ---------------------------------------------------------------------------
# POST /labels/upload — kick off background job, return job_id immediately
# ---------------------------------------------------------------------------
@router.post("/upload")
async def upload_labels(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    user = get_current_user(request)
    db = request.app.state.db
    jobs_store = request.app.state.upload_jobs
    orders_service_url = os.environ.get("ORDERS_SERVICE_URL", "http://localhost:3002")

    # Read all file bytes NOW (before request connection closes)
    files_data = []
    for file in files:
        pdf_bytes = await file.read()
        files_data.append((file.filename or "upload.pdf", pdf_bytes))

    job_id = str(uuid.uuid4())
    jobs_store[job_id] = {
        "status": "processing",
        "current": 0,
        "total": len(files_data),
        "current_file": None,
        "results": None,
        "error": None,
    }

    background_tasks.add_task(
        _process_upload_job,
        job_id,
        files_data,
        user["user_id"],
        db,
        orders_service_url,
        jobs_store,
    )

    return {
        "job_id": job_id,
        "status": "processing",
        "total_files": len(files_data),
    }


# ---------------------------------------------------------------------------
# GET /labels/jobs/{job_id} — poll upload job progress
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}")
async def get_upload_job(job_id: str, request: Request):
    get_current_user(request)  # auth check
    jobs_store = request.app.state.upload_jobs
    job = jobs_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ---------------------------------------------------------------------------
# GET /labels/queue - pending labels (with matched order address details)
# ---------------------------------------------------------------------------
@router.get("/queue")
async def get_queue(request: Request):
    get_current_user(request)
    db = request.app.state.db

    rows = await db.fetch(
        """
        SELECT
          l.id, l.original_filename, l.extracted_name, l.extracted_address,
          l.tracking_number, l.match_confidence, l.match_status, l.order_id, l.uploaded_at,
          o.customer_name   AS matched_customer_name,
          o.external_id     AS matched_order_external_id,
          o.address_line1   AS matched_address_line1,
          o.address_line2   AS matched_address_line2,
          o.city            AS matched_city,
          o.state           AS matched_state,
          o.zip             AS matched_zip,
          o.status          AS matched_order_status,
          o.tracking_number AS order_tracking_number
        FROM labels.shipping_labels l
        LEFT JOIN orders.orders o ON l.order_id = o.id
        WHERE l.match_status IN ('pending', 'tracking_conflict')
        ORDER BY l.uploaded_at DESC
        """,
    )

    return {"labels": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# GET /labels/unmatched - unmatched labels
# ---------------------------------------------------------------------------
@router.get("/unmatched")
async def get_unmatched(request: Request):
    get_current_user(request)
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


# ---------------------------------------------------------------------------
# POST /labels/{label_id}/confirm
# ---------------------------------------------------------------------------
async def _sync_order_after_confirm(orders_service_url: str, order_id: str, label_id: str, label_tracking: str, user_id: str, user_role: str):
    """Background task: update the order with label_id, tracking, and status after confirm."""
    headers = {"X-User-Id": user_id, "X-User-Role": user_role or "staff"}
    try:
        async with httpx.AsyncClient() as client:
            # Get existing order tracking in one call
            existing_tracking = None
            try:
                r = await client.get(f"{orders_service_url}/orders/{order_id}", headers=headers, timeout=15.0)
                if r.status_code == 200:
                    existing_tracking = r.json().get("order", {}).get("tracking_number")
            except Exception as e:
                print(f"[confirm bg] Could not fetch order tracking: {e}")

            update_data = {"label_id": label_id}
            if label_tracking and not existing_tracking:
                update_data["tracking_number"] = label_tracking

            await asyncio.gather(
                client.patch(f"{orders_service_url}/orders/{order_id}", json=update_data, headers=headers, timeout=15.0),
                client.patch(f"{orders_service_url}/orders/{order_id}/status", json={"status": "label_generated", "note": "Label confirmed"}, headers=headers, timeout=15.0),
            )
    except Exception as e:
        print(f"[confirm bg] Failed to sync order {order_id}: {e}")


@router.post("/{label_id}/confirm")
async def confirm_label(label_id: str, request: Request, background_tasks: BackgroundTasks):
    user = get_current_user(request)
    db = request.app.state.db
    body = await request.json()
    order_id = body.get("order_id")

    if not order_id:
        raise HTTPException(status_code=400, detail="order_id is required")

    orders_service_url = os.environ.get("ORDERS_SERVICE_URL", "http://localhost:3002")

    async with db.acquire() as conn:
        label = await conn.fetchrow(
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

    background_tasks.add_task(
        _sync_order_after_confirm,
        orders_service_url, order_id, label_id,
        label["tracking_number"], user["user_id"], user["role"],
    )

    return {"ok": True, "label_id": label_id, "order_id": order_id}


# ---------------------------------------------------------------------------
# PATCH /labels/{label_id}/assign - manually assign label to order
# ---------------------------------------------------------------------------
async def _sync_order_after_assign(orders_service_url: str, order_id: str, label_id: str, label_tracking: str, user_id: str, user_role: str):
    """Background task: update order with label_id and tracking after manual assign."""
    headers = {"X-User-Id": user_id, "X-User-Role": user_role or "staff"}
    try:
        async with httpx.AsyncClient() as client:
            existing_tracking = None
            try:
                r = await client.get(f"{orders_service_url}/orders/{order_id}", headers=headers, timeout=15.0)
                if r.status_code == 200:
                    existing_tracking = r.json().get("order", {}).get("tracking_number")
            except Exception as e:
                print(f"[assign bg] Could not fetch order tracking: {e}")

            update_data = {"label_id": label_id}
            if label_tracking and not existing_tracking:
                update_data["tracking_number"] = label_tracking

            await client.patch(f"{orders_service_url}/orders/{order_id}", json=update_data, headers=headers, timeout=15.0)
    except Exception as e:
        print(f"[assign bg] Failed to sync order {order_id}: {e}")


@router.patch("/{label_id}/assign")
async def assign_label(label_id: str, request: Request, background_tasks: BackgroundTasks):
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

    background_tasks.add_task(
        _sync_order_after_assign,
        orders_service_url, order_id, label_id,
        label["tracking_number"], user["user_id"], user["role"],
    )

    return {"ok": True, "label_id": label_id, "order_id": order_id}


# ---------------------------------------------------------------------------
# GET /labels/{label_id}/download - get signed URL
# ---------------------------------------------------------------------------
@router.get("/{label_id}/download")
async def download_label(label_id: str, request: Request):
    get_current_user(request)
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
