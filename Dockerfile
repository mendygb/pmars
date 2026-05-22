FROM python:3.13-slim

WORKDIR /app

# libgomp1: required by PyTorch on Linux for OpenMP multi-threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch before the rest of requirements.
# The default PyTorch index serves the CUDA build (~2GB); the CPU wheel is ~300MB.
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the safety classifier weights into this image layer (~400MB).
# Without this, the first container startup would trigger a HuggingFace Hub download.
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='KoalaAI/Text-Moderation')"

# Copy application code last — it changes most often, so keeping it at the end
# means the expensive layers above (deps, model download) are reused from cache.
# NOTE: rag/cleaned_chunks.json and rag/cleaned_posts.json are gitignored but required
# at runtime. They must be present in the build context (i.e. on disk locally) before
# running `docker build`. In CI/CD, restore them from artifact storage first.
COPY . .

EXPOSE 8000

# One worker per container. Scale horizontally via multiple replicas/pods instead —
# each pod gets a clean memory footprint and Kubernetes handles load distribution.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
