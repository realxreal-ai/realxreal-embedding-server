# 1. Use slim python image (Debian based, small footprint)
FROM python:3.11-slim

# 2. Set environment variables for optimization
# Prevents Python from writing pyc files to disc (saves I/O)
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures logs are flushed immediately (helps with debugging in DigitalOcean logs)
ENV PYTHONUNBUFFERED=1
# Pass the Firebase credentials env var through
ENV FIREBASE_CREDENTIALS_JSON=$FIREBASE_CREDENTIALS_JSON

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies
# We install 'build-essential' to ensure any optimized C-extensions can compile,
# then we clean up the apt cache to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. CRITICAL STEP: Install CPU-Only PyTorch first
# This prevents pip from downloading the 5GB NVIDIA CUDA version
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# 6. Install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 7. Pre-warm the Model (Downloads ~80MB model to cache)
# This ensures the model is baked into the image, preventing download at startup.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 8. Copy App Code
COPY app/ app/

# 9. Security: Create a non-root user
# Running as root is a security risk. We create 'appuser' and switch to it.
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# 10. Network Configuration
EXPOSE 8080

# 11. Start Command
# Added '--workers 2' to utilize both vCPUs on your new DigitalOcean plan.
# If you stick to 1 vCPU, change this back to 1.
CMD ["uvicorn", "app.secure_embedding_service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]