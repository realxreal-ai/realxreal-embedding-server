# Use a Python base image optimized for small size
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Ensure that the Firebase Admin SDK can access credentials via environment variable
# The application code will look for this variable
ENV FIREBASE_CREDENTIALS_JSON=$FIREBASE_CREDENTIALS_JSON

# Copy requirements file first to take advantage of Docker caching
COPY requirements.txt .

# Install dependencies (no-cache-dir keeps the final image smaller)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application source code
COPY app/ app/

# --- Model Pre-warming (Speeds up deployment) ---
# This downloads the model during the BUILD phase so the server starts fast.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose the port (Standard for cloud platforms)
EXPOSE 8080

# Command to run the application (uvicorn)
# Binds to the port defined in the environment (8080)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]