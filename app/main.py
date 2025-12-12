# secure_embedding_service.py
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, HTTPException, Request, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from dotenv import load_dotenv
import os
import json
import secrets
from typing import Optional
import time

# ============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ============================================================================

load_dotenv()

# Environment-based configuration
ENV = os.getenv("ENVIRONMENT", "development")  # development, staging, production
DEBUG = ENV == "development"

# Logging configuration (NEVER log request bodies in production)
logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("realxreal-api")

# Suppress unnecessary logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

app = FastAPI(
    title="RealxReal Secure Embedding Oracle",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,  # Disable docs in production
    redoc_url=None
)

# ============================================================================
# 2. CORS CONFIGURATION (Restrict to your iOS app)
# ============================================================================

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    logger.warning("⚠️ No ALLOWED_ORIGINS set. CORS will be wide open!")
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],  # Only POST for /embed
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600,
)

# ============================================================================
# 3. RATE LIMITING (Prevent abuse)
# ============================================================================

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# 4. FIREBASE ADMIN INITIALIZATION
# ============================================================================

def initialize_firebase():
    """Initialize Firebase Admin SDK with proper error handling."""
    try:
        # Check if already initialized (prevents double-init in tests)
        firebase_admin.get_app()
        logger.info("🔥 Firebase Admin already initialized")
        return
    except ValueError:
        pass  # Not initialized yet, continue

    cred_path = "firebase-credentials.json"
    json_env = os.getenv("FIREBASE_CREDENTIALS_JSON")
    
    if json_env:
        # Production: Read from Environment Variable
        try:
            cred_dict = json.loads(json_env)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            logger.info("🔥 Firebase Admin initialized via Environment Variable")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid Firebase credentials JSON: {e}")
            raise RuntimeError("Firebase initialization failed")
    
    elif os.path.exists(cred_path):
        # Local Development: Read from File
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        logger.info("🔥 Firebase Admin initialized via Local File")
    
    else:
        logger.error("❌ No Firebase credentials found!")
        raise RuntimeError("Firebase credentials missing. Set FIREBASE_CREDENTIALS_JSON or provide firebase-credentials.json")

initialize_firebase()

# ============================================================================
# 5. MODEL MANAGEMENT (Lazy loading + singleton)
# ============================================================================

_model_instance: Optional[SentenceTransformer] = None
_model_lock = False

def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model (singleton pattern).
    This prevents multiple model copies in memory.
    """
    global _model_instance, _model_lock
    
    if _model_instance is None:
        if _model_lock:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is still loading. Please retry in a few seconds."
            )
        
        _model_lock = True
        try:
            logger.info("🧠 Loading Sentence Transformer model...")
            start_time = time.time()
            _model_instance = SentenceTransformer('all-MiniLM-L6-v2')
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise RuntimeError("Failed to load embedding model")
        finally:
            _model_lock = False
    
    return _model_instance

# ============================================================================
# 6. SECURITY: FIREBASE TOKEN VERIFICATION
# ============================================================================

security = HTTPBearer()

async def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Verify the Firebase ID token sent by the iOS app.
    Returns the authenticated user's UID.
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials
    
    try:
        # Verify the token with Firebase Admin SDK
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        uid = decoded_token['uid']
        
        # Optional: Add additional checks
        # e.g., check if user is not disabled
        # user_record = auth.get_user(uid)
        # if user_record.disabled:
        #     raise HTTPException(status_code=403, detail="User account disabled")
        
        return uid
    
    except auth.ExpiredIdTokenError:
        logger.warning("⚠️ Expired Firebase token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please refresh your session.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except auth.RevokedIdTokenError:
        logger.warning("⚠️ Revoked Firebase token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except auth.InvalidIdTokenError:
        logger.warning("⚠️ Invalid Firebase token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except Exception as e:
        logger.error(f"❌ Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ============================================================================
# 7. REQUEST/RESPONSE MODELS (Input validation)
# ============================================================================

class EmbedRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to embed (1-5000 characters)"
    )
    
    @validator('text')
    def text_must_be_valid(cls, v):
        # Strip whitespace
        v = v.strip()
        
        # Check not empty after stripping
        if not v:
            raise ValueError('Text cannot be empty or only whitespace')
        
        # Optional: Check for suspicious patterns
        # (e.g., repeated characters, control characters)
        if len(set(v)) < 3:  # Less than 3 unique characters
            raise ValueError('Text appears to be spam or invalid')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "What was the name of our first pet?"
            }
        }


class EmbedResponse(BaseModel):
    success: bool
    vector: list[float]
    dimension: int
    request_id: str  # For debugging without logging PII
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "vector": [0.123, -0.456, 0.789],  # truncated for display
                "dimension": 384,
                "request_id": "a1b2c3d4"
            }
        }


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    request_id: str

# ============================================================================
# 8. MAIN ENDPOINT: /embed
# ============================================================================

@app.post(
    "/embed",
    response_model=EmbedResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
@limiter.limit("20/minute")  # Max 20 embeddings per minute per IP
async def create_embedding(
    request: Request,
    body: EmbedRequest,
    uid: str = Depends(verify_firebase_token)
):
    """
    Generate a semantic embedding vector for the provided text.
    
    **Security:**
    - Requires valid Firebase authentication token
    - Rate limited to 20 requests/minute per IP
    - Input validated (1-5000 characters)
    
    **Privacy:**
    - Text is NOT logged or stored
    - Only anonymous request IDs are logged
    - Model inference happens in-memory only
    """
    # Generate anonymous request ID for debugging
    request_id = secrets.token_hex(8)
    
    try:
        # Privacy-preserving log (no PII, no text content)
        logger.info(f"Embedding request: {request_id}")
        
        # Get model instance
        model = get_embedding_model()
        
        # Generate embedding
        start_time = time.time()
        embedding = model.encode(
            body.text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        inference_time = time.time() - start_time
        
        # Log performance metrics only
        logger.info(f"Request {request_id} completed in {inference_time:.3f}s")
        
        return EmbedResponse(
            success=True,
            vector=embedding.tolist(),
            dimension=len(embedding),
            request_id=request_id
        )
    
    except Exception as e:
        logger.error(f"❌ Request {request_id} failed: {type(e).__name__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Embedding generation failed",
                "request_id": request_id
            }
        )

# ============================================================================
# 9. HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Basic health check - is the service running?"""
    return {
        "status": "healthy",
        "service": "realxreal-embedding-api",
        "version": "1.0.0"
    }


@app.get("/ready")
async def readiness_check():
    """
    Readiness check - is the service ready to accept requests?
    Checks if model is loaded and Firebase is initialized.
    """
    try:
        # Check Firebase
        firebase_admin.get_app()
        
        # Check model (will load if not already loaded)
        get_embedding_model()
        
        return {
            "status": "ready",
            "firebase": "connected",
            "model": "loaded"
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

# ============================================================================
# 10. STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Warm up the model on startup to avoid cold start latency."""
    logger.info("🚀 Starting RealxReal Embedding Service...")
    logger.info(f"Environment: {ENV}")
    logger.info(f"Debug mode: {DEBUG}")
    
    # Warm up model with dummy request
    try:
        model = get_embedding_model()
        _ = model.encode("warmup", show_progress_bar=False)
        logger.info("✅ Model warmed up and ready")
    except Exception as e:
        logger.error(f"❌ Startup warmup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down RealxReal Embedding Service...")

# ============================================================================
# 11. MAIN (for local development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "secure_embedding_service:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        log_level="info"
    )