import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers import labels
import asyncpg

app = FastAPI(title="OMS Label Service")

# No CORS here — this service is behind the gateway which handles CORS
# (wildcard + credentials=True is rejected by browsers anyway)

@app.on_event("startup")
async def startup():
    app.state.db = await asyncpg.create_pool(
        dsn=os.environ["DATABASE_URL"],
        min_size=1,
        max_size=5,
        ssl="require",
    )

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()

app.include_router(labels.router, prefix="/labels")

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    from datetime import datetime, timezone
    return {"status": "ok", "service": "labels", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}})
