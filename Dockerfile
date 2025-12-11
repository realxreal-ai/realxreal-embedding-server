# 1. Use slim python image (Debian based, small footprint)
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# Ensure Firebase credentials work
ENV FIREBASE_CREDENTIALS_JSON=$FIREBASE_CREDENTIALS_JSON

# 3. CRITICAL STEP: Install CPU-Only PyTorch first
# This prevents pip from downloading the 5GB NVIDIA CUDA version
# We use the official PyTorch CPU wheel index
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy App Code
COPY app/ app/

# 6. Pre-warm the Model (Downloads ~200MB model to cache)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 7. Network Configuration
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]