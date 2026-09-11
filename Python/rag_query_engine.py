#!/usr/bin/env python3
"""
Insurance Claims RAG Intelligence System
Step 4: LlamaIndex Hybrid RAG Query Layer
==================================================

Wires together the two things already built in Steps 2-3 into a single
question-answering layer:

  1. UNSTRUCTURED side: the ChromaDB collection of 9 policy/SOP chunks from
     Step 3 (semantic search -- "what does the policy say about X?").
  2. STRUCTURED side: the 1,000-row processed claims CSV from Step 2
     (structured lookups -- "show me claim CLM-00123" / "list high-risk
     claims over $10,000").

A LlamaIndex `RouterQueryEngine` sits on top of both and uses an LLM call
to decide which one a given question should go to -- that's the "hybrid
retrieval" Step 4 was scoped for.

Reused from Step 3 (NOT rebuilt):
  - The persistent ChromaDB collection at ./data/chroma_db/ (9 chunks,
    384-dim embeddings). This script connects to it read-only; it does
    NOT re-embed or re-index anything.

Embedding model -- same choice as Step 3, same reason:
  LlamaIndex needs to embed each incoming QUERY the same way the policy
  chunks were embedded, or similarity scores are meaningless. The obvious
  default (`HuggingFaceEmbedding`, via `sentence-transformers`) pulls in
  PyTorch -- and PyPI has no PyTorch wheel newer than 2.2.2 for Intel
  macOS, which is a hard platform ceiling on this machine (see Step 3).
  So this script defines its own small LlamaIndex embedding class that
  wraps the exact same ChromaDB-bundled ONNX all-MiniLM-L6-v2 model Step 3
  used -- zero PyTorch, and the vectors match what's already stored.

LLM: Anthropic Claude API. This is the first step in this project that
  makes real API calls and therefore the first one that needs an
  ANTHROPIC_API_KEY. See the "Before you run this" section in the
  accompanying guide for how to get one.

The LLM's role is deliberately scoped and safe in both halves:
  - Policy side: standard RAG synthesis -- answer using ONLY the retrieved
    chunks, which are always shown alongside the answer as citations.
  - Claims side: the LLM ONLY picks which of a small, fixed set of filter
    parameters to apply (claim_id, is_fraud, min score, etc.) via Claude's
    tool-use feature. It never generates or executes arbitrary SQL/pandas
    code -- the actual data access is 100% deterministic Python, run by
    this script, not the LLM. This keeps structured lookups safe and
    testable independent of any LLM call.

Inputs:
  - ./data/chroma_db/                        (Step 3's persistent store)
  - ./data/processed/processed_claims.csv     (Step 2's cleaned claims)
  - ANTHROPIC_API_KEY in a .env file or environment variable

Outputs:
  - Console: demo query results (routed engine, answer, sources), and
    optionally an interactive Q&A loop (--interactive).

Author: Insurance Claims RAG System
Version: 1.0.0
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------
# Dependency check (fail fast with a clear, actionable message)
# ----------------------------------------------------------------------------
_MISSING: List[str] = []

try:
    import pandas as pd
except ImportError:
    _MISSING.append("pandas")

try:
    from pydantic import PrivateAttr
except ImportError:
    _MISSING.append("pydantic")

try:
    import chromadb
except ImportError:
    _MISSING.append("chromadb")

try:
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
except ImportError:
    _MISSING.append("chromadb (embedding_functions submodule not found)")

try:
    from dotenv import load_dotenv
except ImportError:
    _MISSING.append("python-dotenv")

try:
    import anthropic
except ImportError:
    _MISSING.append("anthropic")

try:
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.base.response.schema import Response
except ImportError:
    _MISSING.append("llama-index-core")

try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
except ImportError:
    _MISSING.append("llama-index-vector-stores-chroma")

try:
    from llama_index.llms.anthropic import Anthropic as LlamaIndexAnthropic
    # llama-index-llms-anthropic hard-codes a whitelist of "known" model names
    # (llama_index.llms.anthropic.utils.CLAUDE_MODELS) and raises ValueError at
    # call time -- "Unknown model: ...` -- for any name not in it, including
    # current model names newer than whatever version of the package got
    # resolved. Confirmed against the library's own source: there is NO
    # constructor override for this (no `context_window=` kwarg exists on this
    # class, despite that being a common pattern on other LlamaIndex LLM
    # wrappers -- tried that first, it raised TypeError). Hit for real on a
    # fresh Streamlit Community Cloud deploy of this app, which resolved a
    # version predating "claude-sonnet-5". Fixed by patching the whitelist
    # dict IN PLACE (mutating the object, not reassigning the name) so it
    # works regardless of which module imported/aliased the lookup function.
    from llama_index.llms.anthropic.utils import CLAUDE_MODELS as _CLAUDE_MODELS
    _CLAUDE_MODELS.setdefault("claude-sonnet-5", 200_000)
except ImportError:
    _MISSING.append("llama-index-llms-anthropic")

if _MISSING:
    print("=" * 80)
    print(f"ERROR: Missing required dependenc{'y' if len(_MISSING) == 1 else 'ies'}: "
          f"{', '.join(_MISSING)}")
    print("=" * 80)
    print("\nFix -- run this in the SAME terminal/venv you used for Steps 1-3:\n")
    print("    python3 -m pip install pandas pydantic chromadb python-dotenv \\")
    print("        anthropic llama-index-core llama-index-vector-stores-chroma \\")
    print("        llama-index-llms-anthropic\n")
    print("(python3 -m pip install ... guarantees it installs into the exact")
    print(" interpreter that will run this script.)\n")
    print("NOTE: deliberately NOT installing 'llama-index-embeddings-huggingface'")
    print("or 'sentence-transformers' -- those pull in PyTorch, which cannot be")
    print("upgraded past 2.2.2 on Intel macOS (see Step 3). This script uses")
    print("ChromaDB's own ONNX embedding model instead -- no new install for that.\n")
    print("Then re-run this script.")
    print("=" * 80)
    sys.exit(1)


# ============================================================================
# Configuration & Logging
# ============================================================================

@dataclass
class QueryEngineConfig:
    """Configuration for the hybrid RAG query layer."""
    chroma_db_path: str = "./data/chroma_db"
    collection_name: str = "policy_documents"
    processed_claims_path: str = "./data/processed/processed_claims.csv"
    anthropic_model: str = "claude-sonnet-5"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    policy_similarity_top_k: int = 3
    claims_default_limit: int = 10
    log_level: str = "INFO"


def setup_logging(config: QueryEngineConfig) -> logging.Logger:
    """Configure structured logging (same format as Steps 1-3)."""
    logger = logging.getLogger(__name__)
    logger.setLevel(config.log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


def load_api_key(logger: logging.Logger) -> str:
    """Load ANTHROPIC_API_KEY from .env / environment, with a clear error
    (and sign-up instructions) if it's missing."""
    load_dotenv()  # looks for a .env file in the current working directory
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("=" * 80)
        print("ERROR: ANTHROPIC_API_KEY not found")
        print("=" * 80)
        print("\nThis step makes real calls to the Anthropic Claude API, which")
        print("needs an API key. To get one:\n")
        print("  1. Go to https://console.anthropic.com and sign up / log in.")
        print("  2. Open 'API Keys' in the left sidebar and create a new key.")
        print("  3. Add billing details if prompted (check current pricing at")
        print("     https://anthropic.com/pricing -- this project's usage is")
        print("     tiny: a handful of short calls per question asked).\n")
        print("Then, in your project root, create (or edit) a file named '.env'")
        print("with this line (replace with your real key):\n")
        print("    ANTHROPIC_API_KEY=sk-ant-...\n")
        print("Then re-run this script.")
        print("=" * 80)
        sys.exit(1)
    return api_key


# ============================================================================
# Embedding: reuse Step 3's ONNX model so query vectors match stored ones
# ============================================================================

class ChromaONNXEmbedding(BaseEmbedding):
    """LlamaIndex embedding wrapper around ChromaDB's bundled ONNX
    all-MiniLM-L6-v2 model (onnxruntime-based -- no PyTorch).

    This MUST be the embedding model LlamaIndex uses here: it guarantees
    query vectors are computed the exact same way the policy chunks were
    embedded and stored in Step 3, so cosine similarity scores are
    meaningful. Using a different embedding model (e.g. the default
    HuggingFaceEmbedding) would silently produce nonsense similarity
    scores even if it ran without error.
    """

    _embed_fn: ONNXMiniLM_L6_V2 = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._embed_fn = ONNXMiniLM_L6_V2()

    @classmethod
    def class_name(cls) -> str:
        return "ChromaONNXEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return list(self._embed_fn([query])[0])

    def _get_text_embedding(self, text: str) -> List[float]:
        return list(self._embed_fn([text])[0])

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [list(v) for v in self._embed_fn(texts)]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


# ============================================================================
# Policy vector query engine (semantic search over Step 3's ChromaDB store)
# ============================================================================

def load_policy_query_engine(
    config: QueryEngineConfig,
    llm: "LlamaIndexAnthropic",
    logger: logging.Logger,
) -> Any:
    """Connect to Step 3's existing ChromaDB collection and build a
    LlamaIndex query engine over it. Does NOT re-embed or rebuild anything.

    Raises:
        FileNotFoundError: if the Step 3 database directory doesn't exist.
        ValueError: if the collection exists but is empty.
    """
    db_path = Path(config.chroma_db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChromaDB store not found at '{db_path}'. Did you run Step 3's "
            f"embeddings_vectorstore_pipeline.py first?"
        )

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(name=config.collection_name)
    except Exception as exc:
        raise ValueError(
            f"Could not open collection '{config.collection_name}' in "
            f"'{db_path}'. Did you run Step 3's "
            f"embeddings_vectorstore_pipeline.py first? Underlying error: {exc}"
        ) from exc

    count = collection.count()
    if count == 0:
        raise ValueError(
            f"Collection '{config.collection_name}' exists but is empty. "
            f"Re-run Step 3's embeddings_vectorstore_pipeline.py to index "
            f"the policy chunks."
        )
    logger.info(
        f"Connected to Step 3's ChromaDB collection '{config.collection_name}' "
        f"({count} chunks)"
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = ChromaONNXEmbedding()
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=config.policy_similarity_top_k,
    )


# ============================================================================
# Claims structured query engine (deterministic filtering, LLM only picks
# which filter to apply -- never generates or runs arbitrary code)
# ============================================================================

FILTER_CLAIMS_TOOL: Dict[str, Any] = {
    "name": "filter_claims",
    "description": (
        "Select which structured filters to apply to the insurance claims "
        "dataset in order to answer the user's question. Only use fields "
        "that are actually relevant to the question; omit the rest."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "Exact claim_id to look up a single specific claim.",
            },
            "is_fraud": {
                "type": "boolean",
                "description": "true = only fraudulent claims, false = only legitimate claims.",
            },
            "min_fraud_confidence_score": {
                "type": "number",
                "description": "Minimum fraud_confidence_score, 0.0-1.0.",
            },
            "policy_type": {
                "type": "string",
                "description": "Exact policy_type to filter to (e.g. 'Commercial Auto', 'Homeowners').",
            },
            "min_claim_amount": {"type": "number"},
            "max_claim_amount": {"type": "number"},
            "limit": {
                "type": "integer",
                "description": "Max number of matching claims to list (default 10).",
            },
        },
    },
}

_CLAIMS_DISPLAY_COLUMNS = [
    "claim_id",
    "policy_type",
    "claim_amount",
    "is_fraud",
    "fraud_confidence_score",
    "days_to_report",
]


class ClaimsQueryEngine(CustomQueryEngine):
    """Structured lookups over the processed claims CSV.

    The LLM here ONLY selects which of the fixed `filter_claims` parameters
    to apply, via Claude's tool-use feature -- it never writes or executes
    SQL/pandas code itself. `_apply_filters` (plain, type-hinted, unit-
    testable Python) does the actual data access. See the module docstring
    for why this design was chosen over an LLM-generated-code approach.
    """

    claims_df: Any
    client: Any
    model_name: str
    default_limit: int = 10

    def _apply_filters(self, **filters: Any) -> "pd.DataFrame":
        """Apply a validated subset of FILTER_CLAIMS_TOOL's parameters to
        the claims DataFrame. Unknown/empty filters are ignored."""
        df = self.claims_df

        claim_id = filters.get("claim_id")
        if claim_id:
            return df[df["claim_id"] == claim_id]

        mask = pd.Series(True, index=df.index)
        if "is_fraud" in filters and filters["is_fraud"] is not None:
            mask &= df["is_fraud"] == bool(filters["is_fraud"])
        if filters.get("min_fraud_confidence_score") is not None:
            mask &= df["fraud_confidence_score"] >= float(
                filters["min_fraud_confidence_score"]
            )
        if filters.get("policy_type"):
            mask &= df["policy_type"].str.casefold() == str(
                filters["policy_type"]
            ).casefold()
        if filters.get("min_claim_amount") is not None:
            mask &= df["claim_amount"] >= float(filters["min_claim_amount"])
        if filters.get("max_claim_amount") is not None:
            mask &= df["claim_amount"] <= float(filters["max_claim_amount"])

        limit = int(filters.get("limit") or self.default_limit)
        return df[mask].sort_values("fraud_confidence_score", ascending=False).head(limit)

    def _extract_filters(self, query_str: str) -> Dict[str, Any]:
        """Ask Claude to pick which filter_claims parameters apply. Raises
        RuntimeError with a clear message on any API failure."""
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=512,
                tools=[FILTER_CLAIMS_TOOL],
                tool_choice={"type": "tool", "name": "filter_claims"},
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract structured filter parameters for this "
                            f"question about an insurance claims dataset: "
                            f"{query_str!r}"
                        ),
                    }
                ],
            )
        except Exception as exc:  # broad on purpose: auth/network/rate-limit errors
            raise RuntimeError(
                f"Anthropic API call failed while extracting claim filters "
                f"({type(exc).__name__}: {exc}). Check that ANTHROPIC_API_KEY "
                f"is set and valid."
            ) from exc

        for block in message.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        return {}

    def custom_query(self, query_str: str) -> str:
        filters = self._extract_filters(query_str)
        results = self._apply_filters(**filters)

        if results.empty:
            return (
                f"No claims matched the extracted filters {filters}. "
                f"(0 of {len(self.claims_df)} claims matched.)"
            )

        lines = [
            f"Found {len(results)} matching claim(s) (filters used: {filters}):",
            "",
        ]
        for _, row in results[_CLAIMS_DISPLAY_COLUMNS].iterrows():
            lines.append(
                f"  - {row['claim_id']} | {row['policy_type']} | "
                f"${row['claim_amount']:,.2f} | "
                f"fraud={row['is_fraud']} (score={row['fraud_confidence_score']:.2f}) | "
                f"reported after {row['days_to_report']} day(s)"
            )
        return "\n".join(lines)


def load_claims_query_engine(
    config: QueryEngineConfig,
    anthropic_client: "anthropic.Anthropic",
    logger: logging.Logger,
) -> ClaimsQueryEngine:
    """Load the processed claims CSV (Step 2 output) and wrap it in a
    ClaimsQueryEngine.

    Raises:
        FileNotFoundError: if Step 2's output CSV doesn't exist.
    """
    path = Path(config.processed_claims_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed claims file not found: '{path}'. Did you run Step "
            f"2's data_ingestion_pipeline.py first?"
        )

    df = pd.read_csv(path)
    required_cols = {
        "claim_id",
        "policy_type",
        "claim_amount",
        "is_fraud",
        "fraud_confidence_score",
        "days_to_report",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"'{path}' is missing expected column(s): {sorted(missing_cols)}. "
            f"Was it produced by Step 2's data_ingestion_pipeline.py?"
        )

    # is_fraud may load as the string "True"/"False" depending on how it was
    # saved -- normalize to an actual bool so filtering works correctly.
    if df["is_fraud"].dtype != bool:
        df["is_fraud"] = df["is_fraud"].astype(str).str.strip().str.lower() == "true"

    logger.info(f"Loaded {len(df)} processed claims from {path}")

    return ClaimsQueryEngine(
        claims_df=df,
        client=anthropic_client,
        model_name=config.anthropic_model,
        default_limit=config.claims_default_limit,
    )


# ============================================================================
# Hybrid router: combines both engines, LLM decides which one to use
# ============================================================================

def build_hybrid_query_engine(
    policy_engine: Any,
    claims_engine: ClaimsQueryEngine,
    llm: "LlamaIndexAnthropic",
) -> RouterQueryEngine:
    """Combine the policy (unstructured) and claims (structured) engines
    into a single router that picks the right one per question."""
    policy_tool = QueryEngineTool.from_defaults(
        query_engine=policy_engine,
        name="policy_documents",
        description=(
            "Answers questions about insurance POLICY RULES, COVERAGE, "
            "EXCLUSIONS, claims PROCEDURES, underwriting standards, and "
            "fraud investigation SOPs -- general rules and requirements, "
            "not specific claim records."
        ),
    )
    claims_tool = QueryEngineTool.from_defaults(
        query_engine=claims_engine,
        name="claims_data",
        description=(
            "Answers questions about SPECIFIC CLAIM RECORDS in the claims "
            "database -- looking up a claim by ID, listing claims matching "
            "criteria (e.g. high fraud score, claim amount, policy type), "
            "or counting/filtering actual claims data."
        ),
    )

    # IMPORTANT: explicitly use LLMSingleSelector rather than letting
    # RouterQueryEngine.from_defaults() auto-pick a selector. Left to its
    # own defaults, it prefers PydanticSingleSelector for any LLM that
    # reports function-calling support (which Anthropic models do) -- and
    # that selector's FunctionCallingProgram path sends `tool_choice=None`
    # explicitly to the Anthropic API instead of omitting the field, which
    # the API rejects with "tool_choice: Input should be an object" (a
    # confirmed llama-index-llms-anthropic bug, reproduced and traced by
    # intercepting the actual request kwargs). LLMSingleSelector uses a
    # plain text completion instead -- no tool_choice involved at all --
    # so it sidesteps the bug entirely while still doing LLM-based routing.
    selector = LLMSingleSelector.from_defaults(llm=llm)

    return RouterQueryEngine.from_defaults(
        query_engine_tools=[policy_tool, claims_tool],
        llm=llm,
        selector=selector,
        select_multi=False,
    )


def describe_routing(response: Any) -> str:
    """Extract a human-readable 'which engine + why' string from a
    RouterQueryEngine response, if that metadata is present."""
    selector_result = (response.metadata or {}).get("selector_result")
    if selector_result is None or not getattr(selector_result, "selections", None):
        return "unknown"
    selection = selector_result.selections[0]
    return f"index {selection.index} (reason: {selection.reason})"


# ============================================================================
# Demo queries (mirrors Step 3's smoke-test pattern)
# ============================================================================

DEFAULT_DEMO_QUERIES: List[str] = [
    "What happens if a claim is filed too late after the loss?",
    "What are the red flags investigators look for when detecting fraud?",
    "List the 5 claims with the highest fraud confidence score.",
    "What is the status of claim CLM-000001?",
]


def run_demo_queries(
    hybrid_engine: RouterQueryEngine, queries: List[str], logger: logging.Logger
) -> None:
    logger.info("=" * 80)
    logger.info("HYBRID QUERY DEMO")
    logger.info("=" * 80)

    for query_str in queries:
        print(f"\nQuestion: \"{query_str}\"")
        try:
            response = hybrid_engine.query(query_str)
        except Exception as exc:
            print(f"  -> ERROR: {type(exc).__name__}: {exc}")
            logger.warning(f"Query failed: {query_str!r} -- {exc}")
            continue

        print(f"  Routed to: {describe_routing(response)}")
        print(f"  Answer: {str(response).strip()}")

        source_nodes = getattr(response, "source_nodes", None)
        if source_nodes:
            print("  Sources:")
            for node in source_nodes:
                meta = node.node.metadata
                print(
                    f"    - {meta.get('source_file', '?')} "
                    f"(chunk {meta.get('chunk_index', '?')}, "
                    f"score={node.score:.4f})"
                )


def interactive_loop(hybrid_engine: RouterQueryEngine, logger: logging.Logger) -> None:
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE -- type a question, or 'quit' to exit")
    print("=" * 80)
    while True:
        try:
            query_str = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not query_str or query_str.lower() in {"quit", "exit"}:
            print("Exiting.")
            return
        try:
            response = hybrid_engine.query(query_str)
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            logger.warning(f"Interactive query failed: {query_str!r} -- {exc}")
            continue
        print(f"[routed to: {describe_routing(response)}]")
        print(str(response).strip())


# ============================================================================
# Main Execution
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: Hybrid RAG query layer over policy docs + claims data."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After the demo queries, drop into an interactive Q&A loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = QueryEngineConfig()
    logger = setup_logging(config)

    logger.info("=" * 80)
    logger.info("Insurance Claims RAG System - Step 4: Hybrid RAG Query Layer")
    logger.info("=" * 80)

    api_key = load_api_key(logger)
    anthropic_client = anthropic.Anthropic(api_key=api_key)
    llm = LlamaIndexAnthropic(
        model=config.anthropic_model,
        api_key=api_key,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
    )

    start = time.time()
    policy_engine = load_policy_query_engine(config, llm, logger)
    claims_engine = load_claims_query_engine(config, anthropic_client, logger)
    hybrid_engine = build_hybrid_query_engine(policy_engine, claims_engine, llm)
    logger.info(f"Hybrid query engine ready in {time.time() - start:.2f}s")

    run_demo_queries(hybrid_engine, DEFAULT_DEMO_QUERIES, logger)

    logger.info("=" * 80)
    logger.info("STEP 4 COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Policy documents engine: {config.policy_similarity_top_k} chunks/query, "
                f"backed by Step 3's ChromaDB collection")
    logger.info(f"✅ Claims data engine: {len(claims_engine.claims_df)} claims loaded, "
                f"LLM picks filters only (no generated code execution)")
    logger.info(f"✅ Router LLM: {config.anthropic_model}")
    logger.info(f"✅ Demo queries run: {len(DEFAULT_DEMO_QUERIES)}")
    logger.info("=" * 80)

    if args.interactive:
        interactive_loop(hybrid_engine, logger)


if __name__ == "__main__":
    main()
