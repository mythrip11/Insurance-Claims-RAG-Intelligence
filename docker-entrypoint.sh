#!/usr/bin/env bash
# Insurance Claims RAG Intelligence -- container entrypoint
#
# Bootstraps the data pipeline on first run (idempotent -- each stage is
# skipped if its output already exists, so a fresh volume runs everything
# once and a warm volume starts in seconds), then launches the dashboard.
set -euo pipefail

cd /app

# Streamlit's CLI asks an interactive "want onboarding emails?" question on
# its very first ever invocation. There's no interactive stdin in a
# container, so pre-write an empty consent file to skip it rather than
# have the container hang waiting for input.
mkdir -p /root/.streamlit
if [ ! -f /root/.streamlit/credentials.toml ]; then
  printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml
fi

if [ ! -f "data/processed/processed_claims.csv" ]; then
  echo "[entrypoint] No processed claims found -- running Step 1 (synthetic data) + Step 2 (ingestion)..."
  python3 "Python/synthetic_data_generator.py"
  python3 "Python/data_ingestion_pipeline.py"
else
  echo "[entrypoint] Found existing processed claims, skipping Steps 1-2."
fi

if [ ! -d "data/chroma_db" ] || [ -z "$(ls -A data/chroma_db 2>/dev/null)" ]; then
  echo "[entrypoint] No ChromaDB store found -- running Step 3 (embeddings)..."
  python3 "Python/embeddings_vectorstore_pipeline.py"
else
  echo "[entrypoint] Found existing ChromaDB store, skipping Step 3."
fi

mkdir -p data/risk_assessments

echo "[entrypoint] Starting Streamlit dashboard on port 8501..."
exec streamlit run "Python/claims_adjuster_dashboard.py" \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true
