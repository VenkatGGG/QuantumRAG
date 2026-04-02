"""Script to pre-download the Hugging Face model during Docker build.

This script downloads the sentence-transformers/all-MiniLM-L6-v2 model
to the Hugging Face cache directory, so it's available at runtime
without needing to download during the first query.
"""

import os
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207a35e9c6575db85a3dc6c09659f"  # nosec: B105 - public model revision hash, not a secret

# Use the cache location set in Dockerfile, or fall back to local cache
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(HF_HOME, exist_ok=True)

print(f"Pre-downloading model: {MODEL_NAME}")
print(f"Cache location: {HF_HOME}")
print("This may take a few minutes...")

# Download both tokenizer and model with pinned revision for security
# nosec: B615 - revision is pinned below
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

# Verify the model can be loaded with local_files_only=True (runtime mode)
print("Verifying model can be loaded in offline mode...")
# nosec: B615 - revision is pinned, local_files_only for verification
tokenizer_verify = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, revision=MODEL_REVISION)
model_verify = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True, revision=MODEL_REVISION)
print("Offline verification successful!")

print(f"Model {MODEL_NAME} successfully downloaded and cached!")
print("The model is now available at runtime without additional downloads.")
