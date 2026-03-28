from fastapi import Request, HTTPException

def get_current_user(request: Request) -> dict:
    user_id = request.headers.get("x-user-id")
    user_role = request.headers.get("x-user-role")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {"user_id": user_id, "role": user_role}

def require_admin(request: Request) -> dict:
    user = get_current_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
