#!/bin/bash
set -e

echo "=== RAG Application Initialization ==="

# Create necessary directories
mkdir -p data
mkdir -p tests
mkdir -p scripts
mkdir -p static
mkdir -p src

# Install dependencies if requirements.txt exists
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

echo "=== Initialization Complete ==="
echo "Project structure ready."
echo "Run 'python scripts/scrape_articles.py' to fetch Wikipedia articles."
echo "Run 'python scripts/run_full_pipeline.py' to process all data."
echo "Run 'uvicorn src.main:app --host 0.0.0.0 --port 8000' to start the API."
