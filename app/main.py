import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, HTTPException, Request, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import logging
from dotenv import load_dotenv
import os
import json

# 1. SETUP & CONFIG
app = FastAPI(title="RealxReal Secure Oracle")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Initialize Firebase Admin
# We handle two cases: Local Dev (File) vs Cloud Prod (Env Variable)
cred_path = "firebase-credentials.json"
json_env = os.getenv("FIREBASE_CREDENTIALS_JSON")

if json_env:
    # Production: Read from Environment Variable
    cred_dict = json.loads(json_env)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    logger.info("🔥 Firebase Admin initialized via Environment Variable")
elif os.path.exists(cred_path):
    # Local: Read from File
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    logger.info("🔥 Firebase Admin initialized via Local File")
else:
    logger.warning("⚠️ No Firebase Credentials found! Auth will fail.")

# Load Model
logger.info("🧠 Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("✅ Model Loaded.")

# 2. SECURITY DEPENDENCY
security = HTTPBearer()

async def verify_token(token: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verifies the Firebase ID Token sent by the iOS app.
    Returns the User UID if valid.
    """
    try:
        decoded_token = auth.verify_id_token(token.credentials)
        uid = decoded_token['uid']
        return uid
    except Exception as e:
        logger.error(f"Auth Failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# 3. ENDPOINTS

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(
    request: Request, 
    body: EmbedRequest, 
    uid: str = Depends(verify_token) # <-- The Guard Logic
):
    """
    Only accessible if the request has a valid Firebase Token.
    """
    try:
        # Privacy Log: We know WHO asked, but we don't log WHAT they asked.
        logger.info(f"Embed Request from User: {uid}")

        embedding = model.encode(body.text)
        return {
            "success": True,
            "vector": embedding.tolist()
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Inference Failed")