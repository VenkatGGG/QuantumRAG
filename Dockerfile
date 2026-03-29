# Dockerfile for RAG Application
# Uses python:3.11-slim base with PyTorch CPU for smaller image size

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - build-essential: Required for compiling some Python packages
# - libhdf5-dev: Required for h5py (HDF5 persistence)
# - curl: Required for healthcheck in docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Install PyTorch CPU version explicitly for smaller image size
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY static/ ./static/
COPY data/ ./data/

# Create data directory if it doesn't exist (for volume mount)
RUN mkdir -p /app/data

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
