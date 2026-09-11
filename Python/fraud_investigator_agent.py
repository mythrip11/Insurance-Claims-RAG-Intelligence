#!/usr/bin/env python3
"""
Insurance Claims RAG Intelligence System
Step 5: AI Co-Investigator Agent
==================================================

Builds a genuine multi-step AGENT on top of Step 4's hybrid query layer:
given a claim, it decides for itself which tools to call (and in what
order) to gather evidence, then produces a structured, schema-validated
fraud risk assessment -- not a single LLM call, a reasoning loop.

Reused from Step 4 (NOT rebuilt):
  - `load_policy_query_engine()` -- Step 3's ChromaDB policy/SOP index,
    wrapped for semantic search (used here as the `search_policy_guidance`
    tool).
  - `load_claims_query_engine()` -- used only to get its already-validated,
    already-normalized claims DataFrame (`.claims_df`); this script does
    NOT reuse `ClaimsQueryEngine` itself, because the agent needs richer,
    multi-claim pattern-matching (shared vendors/contacts across records)
    that the Step 4 single-claim filter tool wasn't scoped for.
  - `QueryEngineConfig`, `setup_logging`, `load_api_key` -- unchanged.

Why a hand-rolled tool loop instead of a LlamaIndex agent class:
  Step 4's debugging journey (see PROJECT_STATUS_AND_HANDOFF.md) found a
  real bug in how llama-index-llms-anthropic constructs `tool_choice` for
  its function-calling agent paths. A manual loop against the raw
  `anthropic` SDK -- the same approach already proven safe in Step 4's
  `ClaimsQueryEngine._extract_filters` -- is fully transparent, easy to
  debug, and sidesteps that integration layer entirely for the one part
  of this project (multi-turn agentic tool use) that would exercise it
  most.

Safety design (same philosophy as Step 4, extended to a full agent):
  - Every tool the agent can call is a deterministic, type-hinted Python
    function over pandas/ChromaDB -- the agent selects WHICH tool and
    WHAT arguments, exactly as in Step 4's `filter_claims`, but it never
    generates or executes arbitrary code.
  - The agent's final answer is not free text: it must call
    `submit_risk_assessment` with arguments that are validated against a
    strict Pydantic schema (`RiskAssessment`). A validation failure is
    fed back to the model as a tool error so it can self-correct, with a
    bounded number of retries -- it can't "succeed" with a malformed or
    incomplete assessment.
  - The system prompt explicitly instructs the model to ground every red
    flag/finding in a tool result and to reason only about objective
    behavioral/financial signals (timing, amounts, vendor/contact
    overlaps, deviation from policy rules) -- never claimant name or any
    demographic-adjacent attribute. This is a portfolio project, but a
    real fraud-scoring system that could be shown to skew on protected
    characteristics is a real regulatory and ethical problem, and it's
    worth designing that constraint in from the start rather than bolting
    it on later.

Inputs:
  - ./data/chroma_db/                        (Step 3's persistent store)
  - ./data/processed/processed_claims.csv     (Step 2's cleaned claims)
  - ANTHROPIC_API_KEY in a .env file or environment variable

Outputs:
  - Console: one investigation report per claim (routing/tool trail,
    risk level, red flags, recommendation).
  - ./data/risk_assessments/<claim_id>_risk_assessment.json -- one
    schema-validated JSON file per investigated claim. This is the
    hand-off format Step 6's Streamlit UI will read.

Author: Insurance Claims RAG System
Version: 1.0.0
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
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
    from pydantic import BaseModel, Field, ValidationError, field_validator
except ImportError:
    _MISSING.append("pydantic")

try:
    import anthropic
except ImportError:
    _MISSING.append("anthropic")

# This script deliberately imports Step 4's module rather than duplicating
# its retrieval logic (per the project hand-off notes). Add this script's
# own directory to sys.path so the import works regardless of the caller's
# current working directory, as long as both files stay side by side in
# "Python/".
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from rag_query_engine import (
        QueryEngineConfig,
        load_api_key,
        load_claims_query_engine,
        load_policy_query_engine,
        setup_logging,
    )
except ImportError as exc:
    _MISSING.append(f"rag_query_engine.py (Step 4 script) -- {exc}")

try:
    from llama_index.llms.anthropic import Anthropic as LlamaIndexAnthropic
except ImportError:
    _MISSING.append("llama-index-llms-anthropic")

if _MISSING:
    print("=" * 80)
    print(f"ERROR: Missing required dependenc{'y' if len(_MISSING) == 1 else 'ies'}: "
          f"{', '.join(_MISSING)}")
    print("=" * 80)
    print("\nIf the missing item is a package, run this in the SAME terminal/venv")
    print("used for Steps 1-4:\n")
    print("    python3 -m pip install pandas pydantic anthropic chromadb \\")
    print("        python-dotenv llama-index-core llama-index-vector-stores-chroma \\")
    print("        llama-index-llms-anthropic\n")
    print("If the missing item is 'rag_query_engine.py', make sure Step 4's script")
    print("is sitting in the same 'Python/' folder as this one -- Step 5")
    print("reuses its retrieval code directly instead of duplicating it.\n")
    print("Then re-run this script.")
    print("=" * 80)
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AgentConfig:
    """Configuration for the AI Co-Investigator agent.

    NOTE: no `temperature` field. The installed `anthropic` SDK (>=1.x, as
    pulled in transitively by llama-index-llms-anthropic) removed
    `temperature` from `Messages.create()`'s signature entirely -- it is
    no longer an accepted keyword argument for this model generation at
    all (confirmed via `inspect.signature()` against the real installed
    client, not assumed). Passing it raises a client-side TypeError before
    any network call is made. There is no direct sampling-temperature
    equivalent in the current signature to substitute it with.
    """
    anthropic_model: str = "claude-sonnet-5"
    # 1536 was too tight in practice: a turn that reasons through evidence
    # and/or emits more than one tool call can run past it, and the model's
    # own extended-thinking behavior (see the `thinking` param on the
    # installed SDK) adds further output-token overhead before any tool_use
    # block. Raised after the first real run hit max_tokens mid-turn and
    # left a dangling tool_use in the conversation history (see
    # PROJECT_STATUS_AND_HANDOFF.md). _handle_response() now also survives
    # a max_tokens cutoff gracefully regardless of this value, but a more
    # generous budget makes hitting it in the first place much rarer.
    llm_max_tokens: int = 4096
    max_tool_iterations: int = 8
    top_n_default: int = 5
    output_dir: str = "./data/risk_assessments"
    log_level: str = "INFO"


# ============================================================================
# Structured output schema
# ============================================================================

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Observed on a real run (CLM-000459, CLM-000826): the model occasionally
# leaks tool-calling-style markup INTO a field's text -- e.g. `summary` ends
# with "...</summary>\n<parameter name=\"recommended_action\">...</parameter>"
# instead of putting `recommended_action` in its own top-level JSON key. When
# this swallows a required field entirely (as it did for CLM-000459) pydantic
# already rejects it as "missing" -- but when the model ALSO happens to supply
# proper copies of every field (CLM-000826), that malformed text still slips
# through validation and pollutes the final report. This pattern catches it
# either way and forces a resubmission through the existing validation-retry
# loop, rather than accepting cosmetically-broken output.
_STRAY_TOOL_MARKUP_PATTERN = re.compile(r"</?parameter\b|</summary>|</?invoke\b", re.IGNORECASE)


def _reject_stray_tool_markup(value: str) -> str:
    match = _STRAY_TOOL_MARKUP_PATTERN.search(value)
    if match:
        raise ValueError(
            f"Field contains stray tool-call-style markup ({match.group(0)!r}) -- each "
            f"field must be provided as its own separate JSON value, never as text "
            f"embedded inside another field. Remove all XML/markup tags and resubmit "
            f"plain prose."
        )
    return value


class RiskAssessment(BaseModel):
    """The agent's final, schema-validated verdict for one claim.

    This is what `submit_risk_assessment` tool calls are validated against
    -- the agent's investigation does not count as complete until an
    instance of this model is produced successfully.
    """

    claim_id: str = Field(min_length=1)
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100, description="0 = no concern, 100 = near-certain fraud")
    summary: str = Field(min_length=1)
    red_flags_identified: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    related_claims: List[str] = Field(default_factory=list)
    recommended_action: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("summary", "recommended_action")
    @classmethod
    def _summary_and_action_have_no_stray_markup(cls, v: str) -> str:
        return _reject_stray_tool_markup(v)

    @field_validator("red_flags_identified", "supporting_evidence", "related_claims")
    @classmethod
    def _list_items_have_no_stray_markup(cls, v: List[str]) -> List[str]:
        for item in v:
            _reject_stray_tool_markup(item)
        return v


# ============================================================================
# System prompt
# ============================================================================

SYSTEM_PROMPT = """You are an AI Co-Investigator supporting human insurance claims \
adjusters. Your job is to investigate ONE claim at a time for signs of fraud and \
produce a structured risk assessment that a human adjuster will review -- you are \
a decision-support tool, not the decision-maker.

Rules you must follow:
1. Use the available tools to gather evidence before forming any conclusion. Start \
by pulling the claim's own record, then check for related claims and relevant \
policy/SOP guidance as needed. Do not rely on prior knowledge about any specific \
claim -- only on what the tools return.
2. Every red flag or piece of supporting evidence in your final assessment must be \
traceable to something a tool actually returned. Do not invent or assume details.
3. Reason ONLY about objective behavioral and financial signals: timing (e.g. days \
to report a loss), amounts relative to baseline/policy limits, vendor or contact \
overlaps across claims, inconsistencies with documented policy rules, and patterns \
described in the fraud investigation SOP. NEVER factor in claimant name or any \
demographic-adjacent attribute -- doing so is both unreliable and a fairness \
violation in this domain.
4. When you have gathered enough evidence, call `submit_risk_assessment` exactly \
once with your final findings. Do not call it speculatively partway through, and do \
not call it more than once.
5. Be honest about uncertainty -- a claim with no red flags should get a low \
risk_score and a low/medium risk_level, not an inflated one. Reserve "critical" for \
claims with multiple, clearly corroborated red flags.
6. When calling `submit_risk_assessment`, every field (summary, recommended_action, \
red_flags_identified, supporting_evidence, related_claims, etc.) must be its own \
separate, plain-text JSON value. Never embed one field's content as text inside \
another field, and never include XML/markup tags of any kind (for example \
`<parameter>`, `</summary>`) anywhere in a field's value -- write plain prose only."""


# ============================================================================
# Tool schemas (Anthropic tool-use format)
# ============================================================================

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_claim_record",
        "description": (
            "Retrieve the full record for one specific claim by its claim_id, "
            "including fields not shown elsewhere (loss description, adjuster "
            "notes, fraud indicator flags, repair vendor, contact details). "
            "Always call this first for the claim under investigation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string", "description": "e.g. 'CLM-000123'"}
            },
            "required": ["claim_id"],
        },
    },
    {
        "name": "find_related_claims",
        "description": (
            "Search the claims dataset for OTHER claims that share an "
            "identifying detail with a given claim: the same repair vendor, "
            "the same claimant email or phone number, or the same policy_id. "
            "Overlapping vendors or contact details across otherwise-unrelated "
            "claims can indicate a staged-claim ring; multiple claims on the "
            "same policy_id can be entirely normal but is worth checking for "
            "unusual frequency."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "match_on": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["repair_vendor", "claimant_email", "claimant_phone", "policy_id"],
                    },
                    "description": "Which fields to match on. Omit to check all four.",
                },
                "max_results": {"type": "integer", "description": "Default 10."},
            },
            "required": ["claim_id"],
        },
    },
    {
        "name": "search_policy_guidance",
        "description": (
            "Semantic search over the company's policy documents and fraud "
            "investigation SOPs. Use this to check the official red flags, "
            "coverage rules, or exclusions relevant to the claim type you're "
            "investigating."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language question, e.g. 'red flags for staged auto accidents'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_baseline_statistics",
        "description": (
            "Get baseline statistics (average/median claim amount, average "
            "days-to-report, historical fraud rate) for a given policy_type, "
            "or dataset-wide if policy_type is omitted. Use this to judge "
            "whether a claim's amount or reporting delay is an outlier for "
            "its category, rather than guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "policy_type": {
                    "type": "string",
                    "description": "Optional; omit for a dataset-wide baseline.",
                }
            },
        },
    },
    {
        "name": "submit_risk_assessment",
        "description": (
            "Submit your final, evidence-backed fraud risk assessment for the "
            "claim under investigation. Call this exactly once, after you have "
            "gathered sufficient evidence with the other tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "risk_score": {"type": "integer", "description": "0-100"},
                "summary": {
                    "type": "string",
                    "description": (
                        "2-4 sentence plain-English summary of the finding, as plain "
                        "prose only. Do not include any other field's content here, "
                        "and do not include XML/markup tags of any kind."
                    ),
                },
                "red_flags_identified": {"type": "array", "items": {"type": "string"}},
                "supporting_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific facts pulled from tool results that back up the red flags.",
                },
                "related_claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "claim_ids from find_related_claims that factored into the assessment, if any.",
                },
                "recommended_action": {"type": "string"},
                "confidence": {"type": "number", "description": "0.0-1.0"},
            },
            "required": [
                "claim_id",
                "risk_level",
                "risk_score",
                "summary",
                "recommended_action",
                "confidence",
            ],
        },
    },
]

_RELATABLE_FIELDS = ("repair_vendor", "claimant_email", "claimant_phone", "policy_id")


# ============================================================================
# The agent
# ============================================================================

class FraudInvestigatorAgent:
    """Runs a bounded, tool-using investigation loop for one claim at a time.

    All data-access tools are deterministic pandas/LlamaIndex operations;
    the LLM only ever chooses which tool to call and with what arguments,
    and its final output is validated against `RiskAssessment` before it is
    accepted. See the module docstring for the full safety rationale.
    """

    def __init__(
        self,
        claims_df: "pd.DataFrame",
        policy_engine: Any,
        client: "anthropic.Anthropic",
        config: AgentConfig,
        logger: logging.Logger,
    ) -> None:
        self.claims_df = claims_df
        self.policy_engine = policy_engine
        self.client = client
        self.config = config
        self.logger = logger

    # -- deterministic tool implementations ---------------------------------

    @staticmethod
    def _row_to_dict(row: "pd.Series") -> Dict[str, Any]:
        """Convert a pandas row to plain JSON-serializable Python types."""
        out: Dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                out[key] = None
            elif hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
                out[key] = value.item()
            else:
                out[key] = value
        return out

    def get_claim_record(self, claim_id: str) -> Dict[str, Any]:
        matches = self.claims_df[self.claims_df["claim_id"] == claim_id]
        if matches.empty:
            return {"error": f"No claim found with claim_id='{claim_id}'."}
        return self._row_to_dict(matches.iloc[0])

    def find_related_claims(
        self,
        claim_id: str,
        match_on: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        source_rows = self.claims_df[self.claims_df["claim_id"] == claim_id]
        if source_rows.empty:
            return {"error": f"No claim found with claim_id='{claim_id}'."}
        source = source_rows.iloc[0]

        fields = [f for f in (match_on or list(_RELATABLE_FIELDS)) if f in _RELATABLE_FIELDS]
        if not fields:
            fields = list(_RELATABLE_FIELDS)

        matches: List[Dict[str, Any]] = []
        seen_claim_ids = {claim_id}
        for field in fields:
            value = source.get(field)
            if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
                continue
            same_value = self.claims_df[
                (self.claims_df[field] == value) & (~self.claims_df["claim_id"].isin(seen_claim_ids))
            ]
            for _, row in same_value.iterrows():
                seen_claim_ids.add(row["claim_id"])
                matches.append(
                    {
                        "claim_id": row["claim_id"],
                        "matched_field": field,
                        "matched_value": str(value),
                        "policy_type": row.get("policy_type"),
                        "claim_amount": float(row.get("claim_amount")),
                        "is_fraud": bool(row.get("is_fraud")),
                        "fraud_confidence_score": float(row.get("fraud_confidence_score")),
                    }
                )
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        return {"claim_id": claim_id, "related_claims_found": len(matches), "matches": matches}

    def search_policy_guidance(self, query: str) -> Dict[str, Any]:
        try:
            response = self.policy_engine.query(query)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            return {"error": f"Policy search failed: {type(exc).__name__}: {exc}"}

        sources: List[Dict[str, Any]] = []
        for node in getattr(response, "source_nodes", None) or []:
            meta = node.node.metadata
            sources.append(
                {
                    "source_file": meta.get("source_file", "?"),
                    "chunk_index": meta.get("chunk_index", "?"),
                    "relevance_score": round(float(node.score), 4) if node.score is not None else None,
                }
            )
        return {"answer": str(response).strip(), "sources": sources}

    def get_baseline_statistics(self, policy_type: Optional[str] = None) -> Dict[str, Any]:
        df = self.claims_df
        if policy_type:
            df = df[df["policy_type"].str.casefold() == policy_type.casefold()]
            if df.empty:
                return {"error": f"No claims found for policy_type='{policy_type}'."}

        return {
            "scope": policy_type or "dataset-wide",
            "claim_count": int(len(df)),
            "claim_amount": {
                "mean": round(float(df["claim_amount"].mean()), 2),
                "median": round(float(df["claim_amount"].median()), 2),
                "std": round(float(df["claim_amount"].std() or 0.0), 2),
            },
            "days_to_report": {
                "mean": round(float(df["days_to_report"].mean()), 2),
                "median": round(float(df["days_to_report"].median()), 2),
            },
            "claim_as_pct_of_limit": {
                "mean": round(float(df["claim_as_pct_of_limit"].mean()), 4),
            },
            "avg_num_previous_claims": round(float(df["num_previous_claims"].mean()), 2),
            "fraud_rate": round(float(df["is_fraud"].mean()), 4),
        }

    def _execute_tool(self, name: str, tool_input: Dict[str, Any]) -> str:
        """Dispatch a non-terminal tool call and return its result as a JSON
        string. Any internal exception is caught and returned as an error
        message rather than raised, so a single bad tool call doesn't crash
        the whole investigation -- the model can see the error and adjust."""
        try:
            if name == "get_claim_record":
                result = self.get_claim_record(**tool_input)
            elif name == "find_related_claims":
                result = self.find_related_claims(**tool_input)
            elif name == "search_policy_guidance":
                result = self.search_policy_guidance(**tool_input)
            elif name == "get_baseline_statistics":
                result = self.get_baseline_statistics(**tool_input)
            else:
                result = {"error": f"Unknown tool '{name}'."}
        except Exception as exc:  # noqa: BLE001 - deliberately broad, fed back to the model
            self.logger.warning(f"Tool '{name}' raised {type(exc).__name__}: {exc}")
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, default=str)

    # -- the agent loop -------------------------------------------------------

    def _call_model(
        self, messages: List[Dict[str, Any]], force_submit: bool = False
    ) -> Any:
        kwargs: Dict[str, Any] = dict(
            model=self.config.anthropic_model,
            max_tokens=self.config.llm_max_tokens,
            # No `temperature` here -- see the AgentConfig docstring: the
            # installed anthropic SDK no longer accepts it as a keyword
            # argument to Messages.create() for this model generation.
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        if force_submit:
            kwargs["tool_choice"] = {"type": "tool", "name": "submit_risk_assessment"}
        try:
            return self.client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Anthropic API call failed: {type(exc).__name__}: {exc}. "
                f"Check that ANTHROPIC_API_KEY is set and valid."
            ) from exc

    def _handle_response(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        tool_call_log: List[str],
        validation_errors_seen: List[str],
        claim_id: str = "?",
    ) -> Optional[RiskAssessment]:
        """Append the assistant turn, execute any tool_use blocks, append the
        resulting tool_result turn, and return a validated RiskAssessment if
        `submit_risk_assessment` succeeded this turn -- otherwise None.

        IMPORTANT: branch on whether `response.content` actually CONTAINS any
        tool_use blocks, not on `response.stop_reason == "tool_use"`. The API
        requires that every tool_use block be immediately followed by a
        matching tool_result in the next message -- full stop, regardless of
        why generation ended. If a turn runs long (e.g. several tool calls
        plus reasoning) and gets cut off by `max_tokens`, `stop_reason` comes
        back as `"max_tokens"` even though one or more COMPLETE tool_use
        blocks are already sitting in `response.content`. Checking
        `stop_reason` alone would skip building tool_results for those,
        leaving a dangling tool_use in the conversation history -- which the
        very next API call then rejects with a 400
        ("tool_use ids were found without tool_result blocks immediately
        after"). This is exactly what happened on the first real run against
        the live API (see PROJECT_STATUS_AND_HANDOFF.md) -- every one of the
        5 investigations failed this way after ~25-28s, consistent with
        `max_tokens` being hit partway through a multi-tool-call turn.
        """
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        if not tool_use_blocks:
            if response.stop_reason == "max_tokens":
                self.logger.warning(
                    "Response hit max_tokens with no complete tool_use block produced -- "
                    "nudging the model to continue. If this recurs, raise AgentConfig.llm_max_tokens."
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please continue the investigation using the available "
                        "tools, and finish by calling submit_risk_assessment."
                    ),
                }
            )
            return None

        if response.stop_reason == "max_tokens":
            self.logger.warning(
                f"Response was truncated by max_tokens after {len(tool_use_blocks)} complete "
                f"tool_use block(s); processing those normally so the API's tool_use/tool_result "
                f"pairing requirement is satisfied. Consider raising AgentConfig.llm_max_tokens "
                f"if this recurs often."
            )

        tool_results: List[Dict[str, Any]] = []
        final_assessment: Optional[RiskAssessment] = None

        for block in tool_use_blocks:
            tool_call_log.append(block.name)

            if block.name == "submit_risk_assessment":
                try:
                    final_assessment = RiskAssessment(**block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Risk assessment accepted.",
                        }
                    )
                except ValidationError as exc:
                    validation_errors_seen.append(str(exc))
                    # Log the moment this happens (not just at the end, in a
                    # final RuntimeError, which never fires if a later retry
                    # succeeds) -- this is what let us see exactly which
                    # field(s) the model got wrong on CLM-000459's 4 failed
                    # attempts instead of only knowing "it eventually worked".
                    self.logger.warning(
                        f"{claim_id}: submit_risk_assessment failed validation "
                        f"(attempt {len(validation_errors_seen)}). Rejected input: "
                        f"{json.dumps(dict(block.input), default=str)}. Errors: {exc}"
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Validation error -- fix and resubmit: {exc}",
                            "is_error": True,
                        }
                    )
            else:
                result_text = self._execute_tool(block.name, dict(block.input))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
                )

        messages.append({"role": "user", "content": tool_results})
        return final_assessment

    def investigate(self, claim_id: str) -> RiskAssessment:
        """Run the full investigation loop for one claim and return a
        validated RiskAssessment.

        Raises:
            ValueError: if claim_id doesn't exist in the dataset.
            RuntimeError: if the API fails, or no validated assessment is
                produced within the configured retry/iteration budget.
        """
        if self.claims_df[self.claims_df["claim_id"] == claim_id].empty:
            raise ValueError(f"Claim '{claim_id}' not found in the claims dataset.")

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Investigate claim {claim_id} for potential fraud. Use the "
                    f"available tools to gather evidence before concluding. When "
                    f"ready, call submit_risk_assessment exactly once."
                ),
            }
        ]
        tool_call_log: List[str] = []
        validation_errors_seen: List[str] = []

        # Phase 1: free exploration -- the model chooses tools/order itself.
        for _ in range(self.config.max_tool_iterations):
            response = self._call_model(messages)
            result = self._handle_response(
                response, messages, tool_call_log, validation_errors_seen, claim_id=claim_id
            )
            if result is not None:
                retry_note = (
                    f" ({len(validation_errors_seen)} validation retry/retries along the way)"
                    if validation_errors_seen
                    else ""
                )
                self.logger.info(
                    f"{claim_id}: assessed in {len(tool_call_log)} tool call(s){retry_note}: {tool_call_log}"
                )
                return result

        # Phase 2: exploration budget spent -- force a final answer, allowing
        # one validation-error retry so the model can self-correct.
        self.logger.warning(f"{claim_id}: exploration budget exhausted, forcing submission")
        for attempt in range(2):
            response = self._call_model(messages, force_submit=True)
            result = self._handle_response(
                response, messages, tool_call_log, validation_errors_seen, claim_id=claim_id
            )
            if result is not None:
                self.logger.info(
                    f"{claim_id}: assessed after forced submission (attempt {attempt + 1}), "
                    f"{len(validation_errors_seen)} total validation retry/retries, "
                    f"tool calls: {tool_call_log}"
                )
                return result

        raise RuntimeError(
            f"Investigation of {claim_id} did not produce a validated risk assessment "
            f"within {self.config.max_tool_iterations} exploration iteration(s) + 2 forced "
            f"attempts. Tool calls made: {tool_call_log}. "
            f"Validation errors seen: {validation_errors_seen or 'none'}."
        )


# ============================================================================
# Console reporting
# ============================================================================

def print_assessment(assessment: RiskAssessment) -> None:
    print(f"\n{'=' * 80}")
    print(f"Claim {assessment.claim_id} -- risk: {assessment.risk_level.value.upper()} "
          f"({assessment.risk_score}/100, confidence {assessment.confidence:.2f})")
    print("=" * 80)
    print(f"Summary: {assessment.summary}")
    if assessment.red_flags_identified:
        print("Red flags:")
        for flag in assessment.red_flags_identified:
            print(f"  - {flag}")
    if assessment.supporting_evidence:
        print("Supporting evidence:")
        for evidence in assessment.supporting_evidence:
            print(f"  - {evidence}")
    if assessment.related_claims:
        print(f"Related claims: {', '.join(assessment.related_claims)}")
    print(f"Recommended action: {assessment.recommended_action}")


def print_batch_summary(assessments: List[RiskAssessment]) -> None:
    if not assessments:
        print("\nNo assessments were produced.")
        return
    print(f"\n{'=' * 80}")
    print("BATCH SUMMARY (sorted by risk_score, descending)")
    print("=" * 80)
    for a in sorted(assessments, key=lambda x: x.risk_score, reverse=True):
        print(f"  {a.claim_id:<14} {a.risk_level.value.upper():<9} score={a.risk_score:>3}  "
              f"confidence={a.confidence:.2f}")


# ============================================================================
# Main Execution
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5: AI Co-Investigator fraud risk agent."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--claim-id", type=str, default=None, help="Investigate a single claim by ID.")
    group.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Investigate the top-N highest fraud-confidence-score claims (default 5 if neither flag is given).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory for risk assessment JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_config = AgentConfig()
    if args.output_dir:
        agent_config.output_dir = args.output_dir

    qe_config = QueryEngineConfig(anthropic_model=agent_config.anthropic_model)
    logger = setup_logging(qe_config)

    logger.info("=" * 80)
    logger.info("Insurance Claims RAG System - Step 5: AI Co-Investigator Agent")
    logger.info("=" * 80)

    api_key = load_api_key(logger)
    anthropic_client = anthropic.Anthropic(api_key=api_key)
    llm = LlamaIndexAnthropic(
        model=qe_config.anthropic_model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
        # See rag_query_engine.py's load_policy_query_engine() for why this is
        # needed -- llama-index-llms-anthropic's own model whitelist can lag
        # behind current model names and raise ValueError at construction time.
        context_window=200_000,
    )

    start = time.time()
    policy_engine = load_policy_query_engine(qe_config, llm, logger)
    claims_df = load_claims_query_engine(qe_config, anthropic_client, logger).claims_df
    agent = FraudInvestigatorAgent(
        claims_df=claims_df,
        policy_engine=policy_engine,
        client=anthropic_client,
        config=agent_config,
        logger=logger,
    )
    logger.info(f"Agent ready in {time.time() - start:.2f}s ({len(claims_df)} claims loaded)")

    if args.claim_id:
        claim_ids = [args.claim_id]
    else:
        top_n = args.top_n or agent_config.top_n_default
        claim_ids = (
            claims_df.sort_values("fraud_confidence_score", ascending=False)
            .head(top_n)["claim_id"]
            .tolist()
        )
        logger.info(f"No --claim-id given: investigating top {top_n} claims by fraud_confidence_score")

    output_dir = Path(agent_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assessments: List[RiskAssessment] = []
    for claim_id in claim_ids:
        logger.info(f"Investigating {claim_id}...")
        try:
            assessment = agent.investigate(claim_id)
        except Exception as exc:
            logger.error(f"Investigation of {claim_id} failed: {type(exc).__name__}: {exc}")
            continue

        out_path = output_dir / f"{claim_id}_risk_assessment.json"
        out_path.write_text(assessment.model_dump_json(indent=2))
        assessments.append(assessment)
        print_assessment(assessment)

    if len(claim_ids) > 1:
        print_batch_summary(assessments)

    logger.info("=" * 80)
    logger.info("STEP 5 COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Claims investigated: {len(claim_ids)}, succeeded: {len(assessments)}, "
                f"failed: {len(claim_ids) - len(assessments)}")
    logger.info(f"Risk assessment JSON files written to: {output_dir.resolve()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
