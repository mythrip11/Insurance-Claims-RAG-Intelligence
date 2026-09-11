# Insurance Claims RAG Intelligence -- Dockerfile
#
# Builds a self-contained image for the full pipeline + Streamlit dashboard.
# Data (data/) is expected to be a mounted volume, not baked into the image --
# see docker-compose.yml. docker-entrypoint.sh bootstraps the pipeline
# (synthetic data -> ingestion -> embeddings) automatically on first run if
# that volume is empty, so `docker compose up` works with zero manual setup
# beyond supplying ANTHROPIC_API_KEY.
#
# No PyTorch dependency anywhere in this project (see README "Known
# Limitations" for why) -- embeddings run on ONNX Runtime, which has
# prebuilt wheels for linux/amd64, so this image stays small and doesn't
# need a GPU or a PyTorch-specific base image.

FROM python:3.11-slim

# build-essential covers the rare case where a transitive dependency (e.g.
# chromadb's HNSW bindings) has no prebuilt wheel for this exact
# python/platform combination and needs to compile from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the requirements file first so dependency installation is
# cached across rebuilds unless requirements.txt actually changes.
COPY ["Python/requirements.txt", "Python/requirements.txt"]
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r "Python/requirements.txt"

# Now copy the application code.
COPY ["Python/", "Python/"]
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# data/ is created here as a mount point; docker-compose.yml maps it to a
# host directory so generated data persists across container restarts.
RUN mkdir -p /app/data

EXPOSE 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]
