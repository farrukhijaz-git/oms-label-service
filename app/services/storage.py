import os
import uuid
from supabase import create_client, Client

def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)

BUCKET = os.environ.get("LABEL_STORAGE_BUCKET", "oms-labels")

def upload_pdf(pdf_bytes: bytes, filename: str) -> str:
    """Upload PDF to Supabase Storage. Returns storage_path."""
    client = get_supabase()
    unique_name = f"labels/{uuid.uuid4()}/{filename}"
    client.storage.from_(BUCKET).upload(
        path=unique_name,
        file=pdf_bytes,
        file_options={"content-type": "application/pdf"}
    )
    return unique_name

def get_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """Get a signed URL for downloading a PDF."""
    client = get_supabase()
    response = client.storage.from_(BUCKET).create_signed_url(storage_path, expires_in)
    return response["signedURL"]
