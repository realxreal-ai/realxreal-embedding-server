# secure_embedding_service.py
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, HTTPException, Request, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
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
import torch
from contextlib import asynccontextmanager

# ============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ============================================================================

load_dotenv()

# Environment-based configuration
ENV = os.getenv("ENVIRONMENT", "development")
DEBUG = ENV == "development"

# Logging configuration
logging.basicConfig(
    level=logging.INFO if not DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("realxreal-api")

# Suppress unnecessary logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ============================================================================
# 2. MODEL MANAGEMENT (Optimized: Quantized + Singleton)
# ============================================================================

_model_instance: Optional[SentenceTransformer] = None
_model_lock = False

def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model (singleton pattern).
    Applies Dynamic Quantization to speed up CPU inference by ~2x.
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
            
            # 1. Load the base model on CPU
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            # 2. PERFORMANCE OPTIMIZATION: Dynamic Quantization
            # Converts weights from float32 to int8.
            logger.info("⚡ Quantizing model for CPU speed boost...")
            
            quantized_backend = torch.quantization.quantize_dynamic(
                model._first_module().auto_model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            model._first_module().auto_model = quantized_backend
            
            _model_instance = model
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded & quantized in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise RuntimeError("Failed to load embedding model")
        finally:
            _model_lock = False
    
    return _model_instance

# ============================================================================
# 3. FIREBASE ADMIN INITIALIZATION
# ============================================================================

def initialize_firebase():
    """Initialize Firebase Admin SDK."""
    try:
        firebase_admin.get_app()
        logger.info("🔥 Firebase Admin already initialized")
        return
    except ValueError:
        pass

    cred_path = "firebase-credentials.json"
    json_env = os.getenv("FIREBASE_CREDENTIALS_JSON")
    
    if json_env:
        try:
            cred_dict = json.loads(json_env)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            logger.info("🔥 Firebase Admin initialized via Environment Variable")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid Firebase credentials JSON: {e}")
            raise RuntimeError("Firebase initialization failed")
    elif os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        logger.info("🔥 Firebase Admin initialized via Local File")
    else:
        logger.error("❌ No Firebase credentials found!")
        raise RuntimeError("Firebase credentials missing.")

initialize_firebase()

# ============================================================================
# 4. LIFESPAN MANAGEMENT (Startup/Shutdown Replacement)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events using the modern FastAPI lifespan format.
    Replaces the deprecated @app.on_event.
    """
    # --- STARTUP LOGIC ---
    logger.info("🚀 Starting RealxReal Embedding Service...")
    logger.info(f"Environment: {ENV}")
    
    # Warm up model to avoid cold start latency
    try:
        model = get_embedding_model()
        _ = model.encode("warmup", show_progress_bar=False)
        logger.info("✅ Model warmed up and ready")
    except Exception as e:
        logger.error(f"❌ Startup warmup failed: {e}")
    
    yield  # The application runs here
    
    # --- SHUTDOWN LOGIC ---
    logger.info("🛑 Shutting down RealxReal Embedding Service...")

# ============================================================================
# 5. FASTAPI APP DEFINITION
# ============================================================================

app = FastAPI(
    title="RealxReal Secure Embedding Oracle",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url=None,
    lifespan=lifespan  # Inject the lifespan manager here
)

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    logger.warning("⚠️ No ALLOWED_ORIGINS set. CORS will be wide open!")
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600,
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================================
# 6. SECURITY: FIREBASE TOKEN VERIFICATION
# ============================================================================

security = HTTPBearer()

async def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        return decoded_token['uid']
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
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
# 7. REQUEST/RESPONSE MODELS (Pydantic V2 Updated)
# ============================================================================

class EmbedRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to embed (1-5000 characters)"
    )
    
    @field_validator('text')
    @classmethod
    def text_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Text cannot be empty or only whitespace')
        if len(set(v)) < 3: 
            raise ValueError('Text appears to be spam or invalid')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {"text": "What was the name of our first pet?"}
        }

class EmbedResponse(BaseModel):
    success: bool
    vector: list[float]
    dimension: int
    request_id: str

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
@limiter.limit("20/minute")
def create_embedding(
    request: Request,
    body: EmbedRequest,
    uid: str = Depends(verify_firebase_token)
):
    """
    Generate a semantic embedding vector.
    Runs synchronously to utilize threadpool for CPU-bound tasks.
    """
    request_id = secrets.token_hex(8)
    
    try:
        # Privacy-preserving log
        logger.info(f"Embedding request: {request_id}")
        
        # Get model instance
        model = get_embedding_model()
        
        # Generate embedding
        start_time = time.time()
        
        embedding = model.encode(
            body.text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        inference_time = time.time() - start_time
        
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
    return {
        "status": "healthy",
        "service": "realxreal-embedding-api",
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    try:
        firebase_admin.get_app()
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
# 10. MAIN (for local development)
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

