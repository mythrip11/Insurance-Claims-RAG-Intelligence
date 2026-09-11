#!/usr/bin/env python3
"""
Insurance Claims RAG Intelligence System
Step 3: Embeddings & ChromaDB Vector Store
==================================================

Loads the chunked policy documents (Step 2 output), generates sentence
embeddings for each chunk, and indexes them into a persistent local
ChromaDB collection. Finishes with a semantic-search smoke test that
proves retrieval actually works, before Step 4 builds the full RAG query
layer on top of this store.

Deliberate scope decision (see PROJECT_STATUS_AND_HANDOFF.md): this step
uses the raw `chromadb` client directly, NOT LlamaIndex. LlamaIndex is
deferred to Step 4, where its query engine is actually needed for hybrid
retrieval across this vector store AND the structured claims data.

Embedding backend -- changed from the original plan, see below:
  Step 1's tech stack called for Hugging Face `sentence-transformers`
  (`all-MiniLM-L6-v2`). That package pulls in PyTorch, and PyPI stopped
  publishing PyTorch wheels for Intel macOS (x86_64) after version 2.2.2 --
  while current `transformers`/`sentence-transformers` require PyTorch
  >= 2.5. On an Intel Mac, `pip install --upgrade torch` can never get past
  2.2.2; there is no wheel to upgrade to. That's a hard platform dead end,
  not a stale-install problem.
  Fix: this script instead uses ChromaDB's OWN bundled export of the exact
  same all-MiniLM-L6-v2 model (`chromadb.utils.embedding_functions.
  ONNXMiniLM_L6_V2`), run through `onnxruntime` instead of PyTorch. Same
  model family, no PyTorch involved anywhere, and `onnxruntime` /
  `tokenizers` / `tqdm` are already hard dependencies of `chromadb` itself
  -- so this needs ZERO new packages beyond what Step 3 already required.

Inputs (produced by Step 2's data_ingestion_pipeline.py):
  - ./data/processed/chunked_policies.json

Outputs:
  - ./data/chroma_db/   (persistent ChromaDB store, written to disk)
  - Console: embedding stats + smoke-test query results

Heads up before running:
  First run downloads the ONNX model (~90MB) from a Chroma-hosted S3
  bucket -- needs internet access once; it's then cached locally at
  ~/.cache/chroma/onnx_models/ and every run after that is fully offline.

Author: Insurance Claims RAG System
Version: 1.1.0
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Dependency check (fail fast with a clear, actionable message)
# ----------------------------------------------------------------------------
_MISSING: List[str] = []

try:
    import numpy as np
except ImportError:
    _MISSING.append("numpy")

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    _MISSING.append("pydantic")

try:
    import chromadb
except ImportError:
    _MISSING.append("chromadb")

try:
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
except ImportError:
    # Ships inside chromadb itself -- if this import fails but plain
    # `chromadb` succeeded above, something is unusual about this install.
    _MISSING.append("chromadb (embedding_functions submodule not found)")

if _MISSING:
    print("=" * 80)
    print(f"ERROR: Missing required dependenc{'y' if len(_MISSING) == 1 else 'ies'}: "
          f"{', '.join(_MISSING)}")
    print("=" * 80)
    print("\nFix -- run this in the SAME terminal/venv you used for Steps 1-2:\n")
    print(f"    python3 -m pip install chromadb\n")
    print("(python3 -m pip install ... guarantees it installs into the exact")
    print(" interpreter that will run this script -- avoids pip/python3")
    print(" pointing at two different environments.)\n")
    print("Then re-run this script.")
    print("=" * 80)
    sys.exit(1)


# ============================================================================
# Configuration & Logging
# ============================================================================

@dataclass
class IndexingConfig:
    """Configuration for the embedding + vector store pipeline."""
    chunked_policies_path: str = "./data/processed/chunked_policies.json"
    chroma_db_path: str = "./data/chroma_db"
    collection_name: str = "policy_documents"
    embedding_model_name: str = "all-MiniLM-L6-v2 (ONNX, via chromadb)"
    distance_metric: str = "cosine"  # one of: "cosine", "l2", "ip"
    smoke_test_top_k: int = 3
    log_level: str = "INFO"


def setup_logging(config: IndexingConfig) -> logging.Logger:
    """Configure structured logging (same format as Steps 1-2)."""
    logger = logging.getLogger(__name__)
    logger.setLevel(config.log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


# ============================================================================
# Schema
# ============================================================================

class PolicyChunkRecord(BaseModel):
    """Validated schema for a single policy chunk (Step 2's JSON output)."""

    chunk_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    policy_title: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    word_count: int = Field(ge=0)


@dataclass
class ChunkValidationIssue:
    """Records why one chunk record failed validation."""
    index: int
    chunk_id: str
    error: str


# ============================================================================
# Loader: chunked_policies.json -> validated PolicyChunkRecord list
# ============================================================================

class PolicyChunkLoader:
    """Loads and validates Step 2's chunked_policies.json output."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def load(
        self, chunked_policies_path: str
    ) -> Tuple[List[PolicyChunkRecord], List[ChunkValidationIssue]]:
        """Read the chunk JSON file and validate every record.

        Raises:
            FileNotFoundError: if the input file doesn't exist.
            ValueError: if the file isn't valid JSON, isn't a list, or
                every single record fails validation (nothing to index).
        """
        path = Path(chunked_policies_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Chunked policies file not found: '{path}'. "
                f"Did you run Step 2's data_ingestion_pipeline.py first?"
            )

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"'{path}' is not valid JSON: {exc}") from exc

        if not isinstance(raw, list):
            raise ValueError(
                f"Expected '{path}' to contain a JSON list of chunk objects, "
                f"got {type(raw).__name__}."
            )

        records: List[PolicyChunkRecord] = []
        issues: List[ChunkValidationIssue] = []
        for i, item in enumerate(raw):
            try:
                records.append(PolicyChunkRecord(**item))
            except ValidationError as exc:
                issues.append(
                    ChunkValidationIssue(
                        index=i,
                        chunk_id=str(item.get("chunk_id", "<unknown>")),
                        error=str(exc.errors()[0]["msg"]) if exc.errors() else str(exc),
                    )
                )

        if not records:
            raise ValueError(
                f"No valid chunk records found in '{path}' "
                f"({len(issues)} record(s) failed validation)."
            )

        self.logger.info(f"Loaded {len(records)} valid policy chunks from {path}")
        return records, issues


# ============================================================================
# Embedding generation
# ============================================================================

class EmbeddingGenerator:
    """Wraps ChromaDB's bundled ONNX export of all-MiniLM-L6-v2.

    See the module docstring for why this replaces `sentence-transformers`:
    in short, that package needs PyTorch >= 2.5, and PyPI has no PyTorch
    wheel newer than 2.2.2 for Intel macOS -- an unfixable dead end on that
    platform. This class keeps the exact same external interface
    (.load() / .encode() / .embedding_dim) that a `sentence-transformers`
    -backed version would have, so the rest of this pipeline doesn't care
    which backend is underneath.
    """

    EMBEDDING_DIM = 384  # fixed output size of all-MiniLM-L6-v2

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self._embedding_fn: Optional[ONNXMiniLM_L6_V2] = None

    def load(self) -> None:
        """Load the embedding model, downloading it on first run if needed."""
        self.logger.info(
            "Loading ChromaDB's bundled ONNX all-MiniLM-L6-v2 model "
            "(first run downloads ~90MB from a Chroma-hosted S3 bucket; "
            "cached locally after that at ~/.cache/chroma/onnx_models/)..."
        )
        start = time.time()
        try:
            self._embedding_fn = ONNXMiniLM_L6_V2()
            self._embedding_fn(["warmup"])  # forces the download/load now,
            # rather than lazily on the first real encode() call, so any
            # failure surfaces here with full context.
        except Exception as exc:  # broad on purpose: network/disk/runtime errors
            raise RuntimeError(
                f"Failed to load the embedding model. If this is the first "
                f"run, check your internet connection (the model is "
                f"downloaded once from Chroma's S3 bucket, then cached "
                f"locally). Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed = time.time() - start
        self.logger.info(f"Model loaded in {elapsed:.1f}s "
                          f"(embedding dimension: {self.embedding_dim})")

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    def encode(self, texts: List[str]) -> "np.ndarray":
        """Encode a list of texts into normalized embedding vectors.

        (Batching is handled internally by ONNXMiniLM_L6_V2 at a fixed
        batch size of 32; there's no batch_size parameter to pass through.)
        """
        if self._embedding_fn is None:
            raise RuntimeError("Call .load() before encode().")
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self._embedding_fn(texts)  # already mean-pooled + L2-normalized
        return np.asarray(embeddings, dtype=np.float32)


# ============================================================================
# Vector store (ChromaDB)
# ============================================================================

class VectorStoreManager:
    """Wraps a persistent local ChromaDB collection for the policy chunks."""

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        distance_metric: str,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.collection_name = collection_name
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        # We always supply our own precomputed embeddings (see
        # EmbeddingGenerator) at add/query time, so no embedding_function is
        # attached to the collection itself -- Chroma never tries to compute
        # embeddings on its own.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        self.logger.info(
            f"Connected to ChromaDB at '{db_path}' "
            f"(collection '{collection_name}', metric='{distance_metric}', "
            f"existing items: {self.collection.count()})"
        )

    def reset(self) -> None:
        """Delete and recreate the collection (used with --rebuild)."""
        metadata = dict(self.collection.metadata or {})
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=metadata
        )
        self.logger.info(f"Collection '{self.collection_name}' reset (0 items).")

    def upsert_chunks(
        self, chunks: List[PolicyChunkRecord], embeddings: "np.ndarray"
    ) -> None:
        """Insert or update chunks in the collection (safe to re-run)."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"length mismatch."
            )
        if not chunks:
            self.logger.warning("No chunks to index -- skipping upsert.")
            return

        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_file": c.source_file,
                    "policy_title": c.policy_title,
                    "chunk_index": c.chunk_index,
                    "word_count": c.word_count,
                }
                for c in chunks
            ],
        )
        self.logger.info(
            f"Upserted {len(chunks)} chunk(s) into '{self.collection_name}' "
            f"(total items now: {self.collection.count()})"
        )

    def count(self) -> int:
        return int(self.collection.count())

    def query(self, query_embedding: "np.ndarray", top_k: int) -> List[Dict[str, Any]]:
        """Run a similarity search and return flattened result dicts."""
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for rank, (chunk_id, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, dists), start=1
        ):
            hits.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "distance": dist,
                    "text": doc,
                    **meta,
                }
            )
        return hits


# ============================================================================
# Smoke test: prove semantic search actually retrieves the right content
# ============================================================================

@dataclass
class SmokeTestQuery:
    """A single semantic-search smoke test case."""
    query_text: str
    acceptable_source_files: List[str] = field(default_factory=list)
    description: str = ""


DEFAULT_SMOKE_TESTS: List[SmokeTestQuery] = [
    SmokeTestQuery(
        query_text="What happens if I file an insurance claim too late after the loss?",
        acceptable_source_files=[
            "policy_01_commercial_auto_insurance.txt",
            "policy_02_homeowners_insurance.txt",
            "policy_08_health_disability_claims.txt",
        ],
        description="Late-filing exclusions / time limits",
    ),
    SmokeTestQuery(
        query_text="What red flags do investigators look for to detect insurance fraud?",
        acceptable_source_files=["policy_03_fraud_investigation_sop.txt"],
        description="Fraud red flags / SOP",
    ),
    SmokeTestQuery(
        query_text="What file formats and photo requirements apply when submitting a claim online?",
        acceptable_source_files=["policy_06_digital_documentation_guide.txt"],
        description="Digital submission requirements",
    ),
    SmokeTestQuery(
        query_text="What criteria are used to underwrite and approve a new insurance policy?",
        acceptable_source_files=["policy_05_underwriting_standards.txt"],
        description="Underwriting standards",
    ),
]


def run_smoke_tests(
    vector_store: VectorStoreManager,
    embedding_generator: EmbeddingGenerator,
    queries: List[SmokeTestQuery],
    top_k: int,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """Run each smoke-test query and print/verify the retrieved results.

    A test "passes" if the top-ranked result's source_file is one of the
    query's acceptable_source_files. Returns (passed, total).
    """
    passed = 0
    logger.info("=" * 80)
    logger.info("SEMANTIC SEARCH SMOKE TEST")
    logger.info("=" * 80)

    for test in queries:
        query_embedding = embedding_generator.encode([test.query_text])[0]
        hits = vector_store.query(query_embedding, top_k=top_k)

        print(f"\nQuery: \"{test.query_text}\"  [{test.description}]")
        if not hits:
            print("  (no results)")
            continue

        for hit in hits:
            snippet = hit["text"].replace("\n", " ")[:140]
            print(
                f"  #{hit['rank']} dist={hit['distance']:.4f} "
                f"| {hit['source_file']} (chunk {hit['chunk_index']}) "
                f"| \"{snippet}...\""
            )

        top_source = hits[0]["source_file"]
        if not test.acceptable_source_files or top_source in test.acceptable_source_files:
            print(f"  -> PASS (top result: {top_source})")
            passed += 1
        else:
            print(
                f"  -> FAIL (top result: {top_source}, "
                f"expected one of {test.acceptable_source_files})"
            )
            logger.warning(
                f"Smoke test mismatch for query '{test.query_text}': "
                f"got '{top_source}', expected one of {test.acceptable_source_files}"
            )

    return passed, len(queries)


# ============================================================================
# Main Execution
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: Embed policy chunks and index them into ChromaDB."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the collection before indexing "
             "(use after changing chunking logic; normal re-runs are safe "
             "without this since upsert is idempotent).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = IndexingConfig()
    logger = setup_logging(config)

    logger.info("=" * 80)
    logger.info("Insurance Claims RAG System - Step 3: Embeddings & Vector Store")
    logger.info("=" * 80)

    # ---- Load chunks ----
    loader = PolicyChunkLoader(logger)
    chunks, issues = loader.load(config.chunked_policies_path)
    if issues:
        logger.warning(f"{len(issues)} chunk(s) failed validation -- showing first 5:")
        for issue in issues[:5]:
            logger.warning(f"  Record {issue.index} ({issue.chunk_id}): {issue.error}")

    # ---- Generate embeddings ----
    embedder = EmbeddingGenerator(logger)
    embedder.load()

    start = time.time()
    embeddings = embedder.encode([c.text for c in chunks])
    elapsed = time.time() - start
    logger.info(
        f"Encoded {len(chunks)} chunks into {embeddings.shape[1]}-dim vectors "
        f"in {elapsed:.2f}s"
    )

    # ---- Index into ChromaDB ----
    vector_store = VectorStoreManager(
        db_path=config.chroma_db_path,
        collection_name=config.collection_name,
        distance_metric=config.distance_metric,
        logger=logger,
    )
    if args.rebuild:
        vector_store.reset()
    vector_store.upsert_chunks(chunks, embeddings)

    # ---- Smoke test ----
    passed, total = run_smoke_tests(
        vector_store, embedder, DEFAULT_SMOKE_TESTS, config.smoke_test_top_k, logger
    )

    # ---- Summary ----
    logger.info("=" * 80)
    logger.info("STEP 3 COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Policy chunks embedded & indexed: {vector_store.count()}")
    logger.info(f"✅ Embedding model: {config.embedding_model_name} "
                f"({embedder.embedding_dim} dimensions)")
    logger.info(f"✅ Vector store path: {config.chroma_db_path}")
    logger.info(f"✅ Collection name: {config.collection_name}")
    logger.info(f"{'✅' if passed == total else '⚠️ '} Smoke test: {passed}/{total} queries "
                f"retrieved an expected source document")
    logger.info("=" * 80)

    if passed < total:
        logger.warning(
            "Not all smoke test queries hit their expected document. Retrieval "
            "still ran successfully -- review the results above before treating "
            "this as a blocker for Step 4."
        )


if __name__ == "__main__":
    main()
