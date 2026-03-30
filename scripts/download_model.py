"""Script to pre-download the Hugging Face model during Docker build.

This script downloads the sentence-transformers/all-MiniLM-L6-v2 model
to the Hugging Face cache directory, so it's available at runtime
without needing to download during the first query.
"""

import os
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Use the cache location set in Dockerfile
HF_HOME = os.environ.get("HF_HOME", "/app/.cache/huggingface")
os.makedirs(HF_HOME, exist_ok=True)

print(f"Pre-downloading model: {MODEL_NAME}")
print(f"Cache location: {HF_HOME}")
print("This may take a few minutes...")

# Download both tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

print(f"Model {MODEL_NAME} successfully downloaded and cached!")
print("The model is now available at runtime without additional downloads.")
