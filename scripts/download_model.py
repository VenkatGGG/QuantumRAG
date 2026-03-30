"""Script to pre-download the Hugging Face model during Docker build.

This script downloads the sentence-transformers/all-MiniLM-L6-v2 model
to the Hugging Face cache directory, so it's available at runtime
without needing to download during the first query.
"""

import os
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Use the cache location set in Dockerfile, or fall back to local cache
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(HF_HOME, exist_ok=True)

print(f"Pre-downloading model: {MODEL_NAME}")
print(f"Cache location: {HF_HOME}")
print("This may take a few minutes...")

# Download both tokenizer and model
# Use local_files_only=False to allow download during build
# Do NOT use cache_dir parameter - rely on HF_HOME environment variable
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

print(f"Model {MODEL_NAME} successfully downloaded and cached!")
print("The model is now available at runtime without additional downloads.")
