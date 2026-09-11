#!/usr/bin/env python3
"""
Insurance Claims RAG Intelligence System
Step 2: Data Ingestion & Preprocessing Pipeline
==================================================

Loads and validates the synthetic claims CSV (Step 1 output) and the policy
documents, then produces clean, chunked artifacts ready for embedding in
Step 3.

Inputs (produced by Step 1's synthetic_data_generator.py):
  - ./data/synthetic/synthetic_claims.csv
  - ./data/synthetic/policies/*.txt

Outputs:
  - ./data/processed/processed_claims.csv   (validated, cleaned claims)
  - ./data/processed/chunked_policies.json  (policy text split into chunks)

Author: Insurance Claims RAG System
Version: 1.0.0
"""

import sys
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------------
# Dependency check (fail fast with a clear, actionable message)
# ----------------------------------------------------------------------------
_MISSING: List[str] = []

try:
    import pandas as pd
except ImportError:
    _MISSING.append("pandas")

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError:
    _MISSING.append("pydantic")

if _MISSING:
    print("=" * 80)
    print(f"ERROR: Missing required dependenc{'y' if len(_MISSING) == 1 else 'ies'}: "
          f"{', '.join(_MISSING)}")
    print("=" * 80)
    print("\nFix -- run this in the SAME terminal/venv you used for Step 1:\n")
    print(f"    python3 -m pip install {' '.join(_MISSING)}\n")
    print("(python3 -m pip install ... guarantees it installs into the exact")
    print(" interpreter that will run this script -- avoids pip/python3")
    print(" pointing at two different environments, which is what caused the")
    print(" 'faker' error in Step 1.)\n")
    print("Then re-run this script.")
    print("=" * 80)
    sys.exit(1)


# ============================================================================
# Configuration & Logging
# ============================================================================

@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    claims_csv_path: str = "./data/synthetic/synthetic_claims.csv"
    policies_dir: str = "./data/synthetic/policies"
    processed_claims_path: str = "./data/processed/processed_claims.csv"
    chunked_policies_path: str = "./data/processed/chunked_policies.json"
    chunk_size_words: int = 150
    chunk_overlap_words: int = 30
    log_level: str = "INFO"


def setup_logging(config: IngestionConfig) -> logging.Logger:
    """Configure structured logging."""
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
# Schemas
# ============================================================================

class ClaimRecord(BaseModel):
    """Validated schema for a single insurance claim record."""

    claim_id: str
    claimant_name: str
    claimant_email: str
    claimant_phone: str
    policy_id: str
    policy_type: str
    policy_start_date: date
    loss_date: date
    loss_description: str
    report_date: date
    claim_amount: float = Field(gt=0)
    policy_limit: float = Field(gt=0)
    claim_as_pct_of_limit: float = Field(ge=0)
    days_to_report: int = Field(ge=0)
    estimated_loss_value: float = Field(gt=0)
    num_previous_claims: int = Field(ge=0)
    repair_vendor: str
    repair_estimate: float = Field(gt=0)
    supporting_docs_count: int = Field(ge=0)
    adjuster_notes: str
    fraud_indicators: List[str] = Field(default_factory=list)
    is_fraud: bool
    fraud_confidence_score: float = Field(ge=0.0, le=1.0)

    @field_validator("fraud_indicators", mode="before")
    @classmethod
    def _parse_fraud_indicators(cls, v: Any) -> List[str]:
        """Convert the pipe-separated CSV string into a list."""
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped or stripped.upper() == "NONE":
                return []
            return [item.strip() for item in stripped.split("|") if item.strip()]
        return v if isinstance(v, list) else []

    @field_validator("is_fraud", mode="before")
    @classmethod
    def _parse_bool(cls, v: Any) -> bool:
        """Convert 'TRUE'/'FALSE' strings into real booleans."""
        if isinstance(v, str):
            return v.strip().upper() == "TRUE"
        return bool(v)


class PolicyChunk(BaseModel):
    """A single chunk of a policy/SOP document, ready for embedding."""

    chunk_id: str
    source_file: str
    policy_title: str
    chunk_index: int
    text: str
    word_count: int


@dataclass
class ValidationIssue:
    """A single row that failed schema validation."""
    row_index: int
    claim_id: str
    error: str


# ============================================================================
# Claims Loader
# ============================================================================

REQUIRED_CLAIM_COLUMNS = [
    "claim_id", "claimant_name", "claimant_email", "claimant_phone",
    "policy_id", "policy_type", "policy_start_date", "loss_date",
    "loss_description", "report_date", "claim_amount", "policy_limit",
    "claim_as_pct_of_limit", "days_to_report", "estimated_loss_value",
    "num_previous_claims", "repair_vendor", "repair_estimate",
    "supporting_docs_count", "adjuster_notes", "fraud_indicators",
    "is_fraud", "fraud_confidence_score",
]


class ClaimsDataLoader:
    """Loads, validates, and cleans the structured claims CSV."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load(self, csv_path: str) -> "pd.DataFrame":
        """Load the raw claims CSV into a DataFrame."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Claims CSV not found at '{path}'. Did you run Step 1's "
                f"synthetic_data_generator.py from the project root first?"
            )
        df = pd.read_csv(path)
        self.logger.info(f"Loaded {len(df)} raw claim rows from {path}")
        return df

    def check_schema(self, df: "pd.DataFrame") -> None:
        """Raise a clear error if required columns are missing."""
        missing = set(REQUIRED_CLAIM_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Claims CSV is missing required columns: {sorted(missing)}. "
                f"This usually means an outdated CSV -- re-run Step 1's "
                f"generator to regenerate it."
            )

    def check_data_quality(self, df: "pd.DataFrame") -> Dict[str, Any]:
        """Run data-quality checks and return a summary report."""
        null_counts = {
            col: int(count) for col, count in df.isnull().sum().items() if count > 0
        }
        report: Dict[str, Any] = {
            "total_rows": len(df),
            "duplicate_claim_ids": int(df["claim_id"].duplicated().sum()),
            "null_counts_by_column": null_counts,
            "negative_claim_amounts": int((df["claim_amount"] <= 0).sum()),
            "fraud_score_out_of_range": int(
                ((df["fraud_confidence_score"] < 0) | (df["fraud_confidence_score"] > 1)).sum()
            ),
        }
        return report

    def validate_and_convert(
        self, df: "pd.DataFrame"
    ) -> Tuple[List[ClaimRecord], List[ValidationIssue]]:
        """Validate every row against ClaimRecord; collect (not crash on) failures."""
        records: List[ClaimRecord] = []
        issues: List[ValidationIssue] = []

        for idx, row in df.iterrows():
            try:
                record = ClaimRecord(**row.to_dict())
                records.append(record)
            except ValidationError as exc:
                issues.append(
                    ValidationIssue(
                        row_index=int(idx),
                        claim_id=str(row.get("claim_id", "UNKNOWN")),
                        error=str(exc).replace("\n", " | "),
                    )
                )

        self.logger.info(
            f"Validated {len(records)}/{len(df)} claims successfully "
            f"({len(issues)} failed validation)"
        )
        return records, issues

    def save_processed(self, records: List[ClaimRecord], output_path: str) -> str:
        """Write validated, cleaned claims back out to CSV."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in records:
            row = r.model_dump()
            row["fraud_indicators"] = (
                "|".join(row["fraud_indicators"]) if row["fraud_indicators"] else "NONE"
            )
            rows.append(row)

        pd.DataFrame(rows).to_csv(out_path, index=False)
        self.logger.info(f"✅ Saved {len(rows)} validated claims to {out_path}")
        return str(out_path)


# ============================================================================
# Policy Document Loader & Chunker
# ============================================================================

class PolicyDocumentLoader:
    """Loads policy/SOP .txt files and splits them into embedding-ready chunks."""

    def __init__(
        self,
        logger: logging.Logger,
        chunk_size_words: int = 150,
        chunk_overlap_words: int = 30,
    ):
        if chunk_overlap_words >= chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        self.logger = logger
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

    def load_documents(self, policies_dir: str) -> List[Dict[str, str]]:
        """Read every .txt file in policies_dir into memory."""
        dir_path = Path(policies_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Policies directory not found: '{dir_path}'")

        txt_files = sorted(dir_path.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in '{dir_path}'. Did you run Step 1's "
                f"generator first?"
            )

        documents = []
        for file_path in txt_files:
            text = file_path.read_text(encoding="utf-8").strip()
            title = text.splitlines()[0].strip() if text else file_path.stem
            documents.append(
                {"source_file": file_path.name, "title": title, "text": text}
            )

        self.logger.info(f"Loaded {len(documents)} policy documents from {dir_path}")
        return documents

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping word-count windows."""
        words = text.split()
        if len(words) <= self.chunk_size_words:
            return [text]

        chunks: List[str] = []
        step = self.chunk_size_words - self.chunk_overlap_words
        start = 0
        while start < len(words):
            window = words[start : start + self.chunk_size_words]
            if not window:
                break
            chunks.append(" ".join(window))
            if start + self.chunk_size_words >= len(words):
                break
            start += step
        return chunks

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[PolicyChunk]:
        """Chunk every loaded document, attaching source metadata to each chunk."""
        all_chunks: List[PolicyChunk] = []
        for doc in documents:
            text_chunks = self._chunk_text(doc["text"])
            stem = Path(doc["source_file"]).stem
            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append(
                    PolicyChunk(
                        chunk_id=f"{stem}_chunk_{i:02d}",
                        source_file=doc["source_file"],
                        policy_title=doc["title"],
                        chunk_index=i,
                        text=chunk_text,
                        word_count=len(chunk_text.split()),
                    )
                )

        self.logger.info(
            f"Created {len(all_chunks)} chunks from {len(documents)} documents "
            f"(~{self.chunk_size_words} words/chunk, {self.chunk_overlap_words} overlap)"
        )
        return all_chunks

    def save_chunks(self, chunks: List[PolicyChunk], output_path: str) -> str:
        """Write chunks to a JSON file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, indent=2)
        self.logger.info(f"✅ Saved {len(chunks)} policy chunks to {out_path}")
        return str(out_path)


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    config = IngestionConfig()
    logger = setup_logging(config)

    logger.info("=" * 80)
    logger.info("Insurance Claims RAG System - Step 2: Data Ingestion & Preprocessing")
    logger.info("=" * 80)

    # ---- Claims pipeline ----
    claims_loader = ClaimsDataLoader(logger)
    df = claims_loader.load(config.claims_csv_path)
    claims_loader.check_schema(df)

    quality_report = claims_loader.check_data_quality(df)
    logger.info(f"Data quality report: {quality_report}")

    records, issues = claims_loader.validate_and_convert(df)
    if issues:
        logger.warning(f"{len(issues)} claim(s) failed validation -- showing first 5:")
        for issue in issues[:5]:
            logger.warning(f"  Row {issue.row_index} ({issue.claim_id}): {issue.error}")

    processed_claims_path = claims_loader.save_processed(
        records, config.processed_claims_path
    )

    # ---- Policy document pipeline ----
    doc_loader = PolicyDocumentLoader(
        logger,
        chunk_size_words=config.chunk_size_words,
        chunk_overlap_words=config.chunk_overlap_words,
    )
    documents = doc_loader.load_documents(config.policies_dir)
    chunks = doc_loader.chunk_documents(documents)
    chunked_policies_path = doc_loader.save_chunks(chunks, config.chunked_policies_path)

    # ---- Summary ----
    fraud_count = sum(1 for r in records if r.is_fraud)
    avg_chunk_words = (
        round(sum(c.word_count for c in chunks) / len(chunks), 1) if chunks else 0
    )

    logger.info("=" * 80)
    logger.info("INGESTION COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Claims validated: {len(records)}/{len(df)} ({len(issues)} rejected)")
    logger.info(f"   - Fraudulent: {fraud_count} | Legitimate: {len(records) - fraud_count}")
    logger.info(f"✅ Processed claims saved to: {processed_claims_path}")
    logger.info(f"✅ Policy documents loaded: {len(documents)}")
    logger.info(f"✅ Policy chunks created: {len(chunks)} (avg {avg_chunk_words} words/chunk)")
    logger.info(f"✅ Chunked policies saved to: {chunked_policies_path}")
    logger.info("=" * 80)

    if issues:
        logger.warning(
            f"⚠️  {len(issues)} claim row(s) were rejected during validation. "
            f"Review the warnings above before proceeding to Step 3."
        )


if __name__ == "__main__":
    main()
