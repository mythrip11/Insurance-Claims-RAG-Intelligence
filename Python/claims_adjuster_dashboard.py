#!/usr/bin/env python3
"""
Insurance Claims RAG Intelligence System
Step 6: Claims Adjuster Dashboard (Streamlit UI)
==================================================

An interactive dashboard for claims adjusters, built on top of Steps 1-5.
It does NOT duplicate any retrieval or fraud-reasoning logic -- every data
lookup and every risk assessment shown here comes from calling the exact
same code Steps 4-5 already built and verified against the real Anthropic
API:

  - `rag_query_engine.py` (Step 4): `QueryEngineConfig`, `setup_logging`,
    `load_api_key`, `load_policy_query_engine`, `load_claims_query_engine`.
  - `fraud_investigator_agent.py` (Step 5): `AgentConfig`,
    `FraudInvestigatorAgent`, `RiskAssessment`. This dashboard calls
    `agent.get_claim_record()` and `agent.investigate()` directly -- the
    same tool implementations and the same bounded, self-correcting
    tool-use loop that were debugged and confirmed live in Step 5. This
    script adds no new fraud-reasoning logic of its own; it is a
    presentation layer over what already exists.

What it does:
  1. Loads the full processed claims book (Step 2's output, ~1,000 claims)
     and every existing risk assessment JSON file Step 5 has written to
     `./data/risk_assessments/`.
  2. Gives an adjuster a filterable overview (policy type, investigation
     status, risk level, minimum fraud_confidence_score, claim ID search)
     plus summary KPIs and two charts.
  3. Lets the adjuster drill into any single claim: full claim record,
     and -- if investigated -- the AI Co-Investigator's structured risk
     assessment (risk level/score, grounded red flags, supporting
     evidence, related claims, recommended action, confidence).
  4. Lets the adjuster trigger a LIVE investigation for any claim (already
     assessed or not) with one click. This makes a real call to
     `FraudInvestigatorAgent.investigate()` -- expect ~20-60+ seconds per
     claim, the same real-world latency observed in Step 5's live runs
     (multi-turn tool use, occasional validation-retry round trips). The
     result is written to disk in the exact same format/location Step 5's
     CLI uses (`<claim_id>_risk_assessment.json`), so the two stay fully
     interoperable -- you can run the CLI in batch overnight and review
     results here, or investigate one-off claims here and see them if you
     later run the CLI.
  5. Includes an optional "model validation" panel that compares
     investigated claims' AI risk level against the dataset's synthetic
     `is_fraud` ground-truth label. This ONLY works because the claims
     book is synthetic (Step 1) -- a real deployment would not have
     ground truth to compare against -- but it's a genuinely useful
     portfolio artifact for demonstrating the system's practical accuracy.

Inputs (all reused, none rebuilt):
  - ./data/chroma_db/                     (Step 3's persistent policy index)
  - ./data/processed/processed_claims.csv (Step 2's cleaned claims)
  - ./data/risk_assessments/*.json        (Step 5's prior outputs, if any)
  - ANTHROPIC_API_KEY in .env or environment (only needed to run live
    investigations -- browsing existing assessments works without it,
    though the dashboard still needs it at startup since the policy
    engine and Anthropic client are built eagerly; see Design notes.)

Design notes:
  - `FraudInvestigatorAgent`, the policy query engine, and the Anthropic
    client are built once via `@st.cache_resource` and shared across
    reruns/sessions for the life of the Streamlit process -- rebuilding
    the ChromaDB connection and LLM client on every widget interaction
    would be slow and pointless, and all three are safe to share
    read-only.
  - Risk assessment JSON files are re-read from disk on every rerun
    (there are at most a few thousand small JSON files -- this is cheap)
    rather than cached, specifically so a fresh investigation triggered
    from this dashboard, or a batch run of Step 5's CLI in another
    terminal, shows up immediately without a stale cache to invalidate.
  - The claims table/filters deliberately do not expose claimant name as a
    sort/filter field -- the same fairness constraint Step 5's system
    prompt enforces on the agent (reason on objective behavioral/
    financial signals, not identity) is echoed here in what the UI makes
    easy to do. Claimant name is still shown in the per-claim detail view,
    where an adjuster legitimately needs it to work the case.

Run (same pattern as Steps 1-5):
    cd ~/Desktop/Insurance\ Claims\ RAG\ Intelligence
    source .venv/bin/activate
    streamlit run "Python/claims_adjuster_dashboard.py"

This opens a browser tab (usually http://localhost:8501). Leave the
terminal open while using the dashboard -- both Streamlit's own messages
and this script's logger (same format as Steps 1-5) print there, which is
useful if something errors.

Author: Insurance Claims RAG System
Version: 1.0.0
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Dependency check (fail fast with a clear, actionable message -- same
# pattern as Steps 4-5)
# ----------------------------------------------------------------------------
_MISSING: List[str] = []

try:
    import pandas as pd
except ImportError:
    _MISSING.append("pandas")

try:
    import streamlit as st
except ImportError:
    _MISSING.append("streamlit")

try:
    import plotly.express as px
except ImportError:
    _MISSING.append("plotly")

try:
    import anthropic
except ImportError:
    _MISSING.append("anthropic")

try:
    from llama_index.llms.anthropic import Anthropic as LlamaIndexAnthropic
    # See rag_query_engine.py's import block for why this is needed --
    # llama-index-llms-anthropic's own hardcoded model whitelist can lag
    # behind current model names and raise ValueError at call time.
    from llama_index.llms.anthropic.utils import CLAUDE_MODELS as _CLAUDE_MODELS
    _CLAUDE_MODELS.setdefault("claude-sonnet-5", 200_000)
except ImportError:
    _MISSING.append("llama-index-llms-anthropic")

# This script imports Steps 4 and 5's modules directly rather than
# duplicating any retrieval or agent logic. Add this script's own directory
# to sys.path so the import works regardless of the caller's current
# working directory, as long as all three files stay side by side in
# "Python/" (same convention as Step 5 importing Step 4).
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
    from fraud_investigator_agent import (
        AgentConfig,
        FraudInvestigatorAgent,
        RiskAssessment,
    )
except ImportError as exc:
    _MISSING.append(f"fraud_investigator_agent.py (Step 5 script) -- {exc}")

if _MISSING:
    print("=" * 80)
    print(f"ERROR: Missing required dependenc{'y' if len(_MISSING) == 1 else 'ies'}: "
          f"{', '.join(_MISSING)}")
    print("=" * 80)
    print("\nIf the missing item is a package, run this in the SAME terminal/venv")
    print("used for Steps 1-5:\n")
    print("    python3 -m pip install streamlit plotly pandas anthropic \\")
    print("        llama-index-llms-anthropic\n")
    print("If the missing item is 'rag_query_engine.py' or")
    print("'fraud_investigator_agent.py', make sure both scripts are sitting in")
    print("the same 'Python/' folder as this one -- this dashboard reuses")
    print("their code directly instead of duplicating it.\n")
    print("Then re-run: streamlit run \"Python/claims_adjuster_dashboard.py\"")
    print("=" * 80)
    sys.exit(1)


# ============================================================================
# Constants & page config
# ============================================================================

RISK_ORDER: List[str] = ["low", "medium", "high", "critical"]
RISK_COLORS: Dict[str, str] = {
    "low": "#2e7d32",
    "medium": "#f9a825",
    "high": "#ef6c00",
    "critical": "#c62828",
}

# Single source of truth for paths/model settings -- reuses Step 4's and
# Step 5's own config dataclasses (with their own sensible defaults)
# instead of redefining any of these values here.
_QE_CONFIG = QueryEngineConfig()
_AGENT_CONFIG = AgentConfig()

st.set_page_config(
    page_title="Claims Fraud Risk Dashboard",
    page_icon="\U0001F50E",
    layout="wide",
)


# ============================================================================
# Cached, expensive setup (ChromaDB connection, Anthropic client, claims df)
# ============================================================================

@st.cache_resource(show_spinner="Connecting to the policy index and claims data, and starting the Anthropic client (one-time per server session)...")
def _build_agent() -> Tuple[FraudInvestigatorAgent, logging.Logger]:
    """Build the FraudInvestigatorAgent exactly the way Step 5's own
    `main()` does, reusing Step 4's loaders -- so there is exactly one
    place (Step 4/5's own code) that knows how to construct the policy
    engine, the claims DataFrame, and the Anthropic client.

    Cached for the life of the Streamlit server process: the ChromaDB
    connection, the LlamaIndex query engine, and the Anthropic client are
    all safe to share read-only across reruns and across browser sessions.

    Raises:
        RuntimeError: if ANTHROPIC_API_KEY is missing (converted from the
            SystemExit that `load_api_key` raises directly, so a missing
            key shows a Streamlit error instead of killing the server).
        FileNotFoundError: if Step 3's ChromaDB store or Step 2's
            processed claims CSV don't exist yet.
        ValueError: if the ChromaDB collection or claims CSV exist but are
            empty/malformed.
    """
    logger = setup_logging(_QE_CONFIG)

    try:
        api_key = load_api_key(logger)
    except SystemExit as exc:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Setup instructions were printed "
            "to the terminal window running `streamlit run` -- add the key "
            "to your .env file and restart the dashboard."
        ) from exc

    anthropic_client = anthropic.Anthropic(api_key=api_key)
    llm = LlamaIndexAnthropic(
        model=_QE_CONFIG.anthropic_model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
    )

    policy_engine = load_policy_query_engine(_QE_CONFIG, llm, logger)
    claims_df = load_claims_query_engine(_QE_CONFIG, anthropic_client, logger).claims_df

    agent = FraudInvestigatorAgent(
        claims_df=claims_df,
        policy_engine=policy_engine,
        client=anthropic_client,
        config=_AGENT_CONFIG,
        logger=logger,
    )
    logger.info(f"Dashboard agent ready ({len(claims_df)} claims loaded)")
    return agent, logger


# ============================================================================
# Risk assessment loading & overview construction (dashboard-only, no
# business logic -- purely reshapes what Steps 4/5 already produced)
# ============================================================================

def load_risk_assessments(output_dir: Path) -> Dict[str, RiskAssessment]:
    """Read every '<claim_id>_risk_assessment.json' file in `output_dir`
    (written by Step 5's CLI and/or this dashboard's own 'Investigate'
    button) and parse each into a validated RiskAssessment, keyed by the
    claim_id inside the file (not the filename, for robustness).

    Malformed files are skipped with a logged warning rather than crashing
    the whole dashboard load -- one bad file should not take down the
    review of every other claim.
    """
    assessments: Dict[str, RiskAssessment] = {}
    if not output_dir.exists():
        return assessments

    logger = logging.getLogger(__name__)
    for path in sorted(output_dir.glob("*_risk_assessment.json")):
        try:
            assessment = RiskAssessment.model_validate_json(path.read_text())
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any bad file is skippable
            logger.warning(f"Skipping unreadable assessment file '{path.name}': {exc}")
            continue
        assessments[assessment.claim_id] = assessment
    return assessments


def build_overview(claims_df: "pd.DataFrame", assessments: Dict[str, RiskAssessment]) -> "pd.DataFrame":
    """Left-join the claims book with the current risk-assessment cache
    for the dashboard's own table/chart rendering. Does not mutate the
    original claims_df and does not perform any fraud reasoning itself --
    all scoring happens inside FraudInvestigatorAgent; this only presents
    its output alongside the underlying claim data.
    """
    overview = claims_df.copy()
    overview["investigated"] = overview["claim_id"].isin(assessments)
    overview["risk_level"] = overview["claim_id"].map(
        lambda cid: assessments[cid].risk_level.value if cid in assessments else None
    )
    overview["risk_score"] = overview["claim_id"].map(
        lambda cid: assessments[cid].risk_score if cid in assessments else None
    )
    overview["agent_confidence"] = overview["claim_id"].map(
        lambda cid: assessments[cid].confidence if cid in assessments else None
    )
    return overview


# ============================================================================
# Rendering helpers
# ============================================================================

def _format_currency(value: Optional[float]) -> str:
    return f"${value:,.2f}" if value is not None else "—"


def _render_claim_record(record: Dict[str, object]) -> None:
    """Render one claim's raw record fields for an adjuster to review."""
    info: Dict[str, object] = {
        "Claimant": record.get("claimant_name"),
        "Claimant contact": f"{record.get('claimant_email')} / {record.get('claimant_phone')}",
        "Policy": f"{record.get('policy_id')} ({record.get('policy_type')})",
        "Policy start": record.get("policy_start_date"),
        "Loss date": record.get("loss_date"),
        "Report date": record.get("report_date"),
        "Days to report": record.get("days_to_report"),
        "Claim amount": _format_currency(record.get("claim_amount")),
        "Policy limit": _format_currency(record.get("policy_limit")),
        "Claim as % of limit": (
            f"{record.get('claim_as_pct_of_limit') * 100:.1f}%"
            if record.get("claim_as_pct_of_limit") is not None
            else "—"
        ),
        "Estimated loss value": _format_currency(record.get("estimated_loss_value")),
        "Repair vendor": record.get("repair_vendor"),
        "Repair estimate": _format_currency(record.get("repair_estimate")),
        "Prior claims by claimant": record.get("num_previous_claims"),
        "Supporting docs on file": record.get("supporting_docs_count"),
        "System fraud indicators": record.get("fraud_indicators") or "none",
        "fraud_confidence_score (dataset)": record.get("fraud_confidence_score"),
    }
    for label, value in info.items():
        st.write(f"**{label}:** {value}")

    with st.expander("Loss description & adjuster notes"):
        st.write(record.get("loss_description") or "—")
        st.write(record.get("adjuster_notes") or "—")


def _render_assessment(assessment: RiskAssessment) -> None:
    """Render one validated RiskAssessment (Step 5's structured output)."""
    color = RISK_COLORS.get(assessment.risk_level.value, "#616161")
    st.markdown(
        f"<span style='background-color:{color};color:white;padding:4px 10px;"
        f"border-radius:12px;font-weight:600;'>{assessment.risk_level.value.upper()}</span>"
        f"&nbsp;&nbsp;Risk score: <b>{assessment.risk_score}/100</b>"
        f"&nbsp;&nbsp;Agent confidence: <b>{assessment.confidence:.0%}</b>",
        unsafe_allow_html=True,
    )
    st.progress(assessment.risk_score / 100)
    st.write(assessment.summary)

    if assessment.red_flags_identified:
        st.markdown("**Red flags identified**")
        for flag in assessment.red_flags_identified:
            st.markdown(f"- {flag}")

    if assessment.supporting_evidence:
        with st.expander("Supporting evidence (tool-grounded)"):
            for item in assessment.supporting_evidence:
                st.markdown(f"- {item}")

    if assessment.related_claims:
        st.markdown(f"**Related claims:** {', '.join(assessment.related_claims)}")

    st.markdown(f"**Recommended action:** {assessment.recommended_action}")


def _render_claim_detail(
    agent: FraudInvestigatorAgent,
    assessments: Dict[str, RiskAssessment],
    claim_id: str,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Render the full detail panel (claim record + assessment + the
    'investigate now' action) for one selected claim.

    Reuses `agent.get_claim_record()` and `agent.investigate()` directly --
    the same deterministic lookup and the same bounded tool-use loop Step 5
    already built and verified. No fraud logic is re-implemented here.
    """
    record = agent.get_claim_record(claim_id)
    if "error" in record:
        st.error(record["error"])
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Claim record**")
        _render_claim_record(record)

    with right:
        st.markdown("**AI Co-Investigator assessment**")
        assessment = assessments.get(claim_id)
        button_label = "Re-investigate this claim" if assessment else "Investigate this claim now"
        if st.button(button_label, key=f"investigate_{claim_id}", type="primary"):
            with st.spinner(
                f"Investigating {claim_id} -- the agent is calling tools and reasoning "
                f"across multiple turns, this typically takes 20-60+ seconds..."
            ):
                try:
                    new_assessment = agent.investigate(claim_id)
                except Exception as exc:  # noqa: BLE001 - surfaced to the adjuster, not swallowed
                    st.error(f"Investigation failed: {type(exc).__name__}: {exc}")
                    new_assessment = None

            if new_assessment is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"{claim_id}_risk_assessment.json"
                out_path.write_text(new_assessment.model_dump_json(indent=2))
                logger.info(f"Dashboard: wrote {out_path}")
                st.success(f"Investigation complete -- saved to {out_path.name}")
                st.rerun()

        if assessment:
            _render_assessment(assessment)
        else:
            st.info(
                "This claim has not been investigated yet. Click the button "
                "above to run the AI Co-Investigator live against the real "
                "Anthropic API."
            )


def _render_validation_panel(overview: "pd.DataFrame") -> None:
    """Compare investigated claims' AI risk level against the dataset's
    synthetic `is_fraud` ground-truth label. Only meaningful because Step
    1's claims are synthetic with a known label -- a real deployment would
    not have this available -- but it's a genuinely useful way to show
    this system actually works, not just that it produces plausible text.
    """
    investigated = overview[overview["investigated"]]
    if investigated.empty:
        st.info("Investigate at least one claim to see this panel.")
        return

    flagged = investigated["risk_level"].isin(["high", "critical"])
    actual_fraud = investigated["is_fraud"].astype(bool)

    true_positive = int((flagged & actual_fraud).sum())
    false_positive = int((flagged & ~actual_fraud).sum())
    false_negative = int((~flagged & actual_fraud).sum())
    true_negative = int((~flagged & ~actual_fraud).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True positives", true_positive, help="Flagged high/critical AND actually fraud")
    c2.metric("False positives", false_positive, help="Flagged high/critical but NOT actually fraud")
    c3.metric("False negatives", false_negative, help="Not flagged high/critical but IS actually fraud")
    c4.metric("True negatives", true_negative, help="Not flagged AND not actually fraud")

    st.dataframe(
        investigated[["claim_id", "policy_type", "is_fraud", "risk_level", "risk_score", "agent_confidence"]]
        .rename(columns={"is_fraud": "actual_is_fraud"})
        .sort_values("risk_score", ascending=False),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "\"Flagged\" = AI risk level is high or critical. Ground truth is the "
        "synthetic `is_fraud` label from the claims generator -- available "
        "here only because this is a portfolio dataset with known labels."
    )


# ============================================================================
# Main app
# ============================================================================

def main() -> None:
    st.title("Claims Fraud Risk Dashboard")
    st.caption(
        "AI Co-Investigator risk assessments over the claims book. Every "
        "score below is traceable to a real tool call (claim lookup, "
        "cross-claim matching, policy search, or baseline statistics) -- "
        "not just plausible-sounding text."
    )

    try:
        agent, logger = _build_agent()
    except Exception as exc:  # noqa: BLE001 - shown to the adjuster, app should not crash
        st.error(
            f"Could not start the dashboard: {type(exc).__name__}: {exc}\n\n"
            f"Check the terminal window running `streamlit run` for full setup "
            f"instructions (same environment checks as Steps 4-5)."
        )
        st.stop()
        return

    output_dir = Path(_AGENT_CONFIG.output_dir)
    assessments = load_risk_assessments(output_dir)
    overview = build_overview(agent.claims_df, assessments)

    # ---- Sidebar: filters + info ----
    st.sidebar.header("Filters")
    policy_types = sorted(overview["policy_type"].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect("Policy type", policy_types, default=policy_types)

    status_choice = st.sidebar.radio(
        "Investigation status", ["All", "Investigated", "Not yet investigated"], index=0
    )

    risk_levels_present = [lvl for lvl in RISK_ORDER if lvl in set(overview["risk_level"].dropna().unique())]
    selected_levels = st.sidebar.multiselect(
        "Risk level (investigated claims only)", risk_levels_present, default=risk_levels_present
    )

    min_score = st.sidebar.slider("Min. fraud_confidence_score (dataset)", 0.0, 1.0, 0.0, 0.05)
    search_term = st.sidebar.text_input("Search claim ID").strip()

    if st.sidebar.button("Refresh assessments from disk"):
        st.rerun()

    st.sidebar.caption(f"Risk assessments read from: `{output_dir}`")
    st.sidebar.caption(f"Loaded at {datetime.now().strftime('%H:%M:%S')}")

    with st.sidebar.expander("About this system"):
        st.markdown(
            "- **Data:** 1,000 synthetic claims + 8 policy/SOP docs\n"
            "- **Retrieval:** ChromaDB + ONNX MiniLM embeddings, "
            "with a LlamaIndex hybrid router over policy docs + structured "
            "claims\n"
            "- **Agent:** hand-rolled multi-turn tool-use loop over the raw "
            "Anthropic SDK -- 4 deterministic read tools + a Pydantic-"
            "validated `submit_risk_assessment` terminal tool\n"
            "- **This dashboard:** a presentation layer only -- it calls "
            "the same agent and retrieval code directly, no separate logic "
            "of its own"
        )

    # ---- Filtering ----
    mask = overview["policy_type"].isin(selected_types) & (overview["fraud_confidence_score"] >= min_score)
    if status_choice == "Investigated":
        mask &= overview["investigated"]
    elif status_choice == "Not yet investigated":
        mask &= ~overview["investigated"]
    if selected_levels:
        mask &= overview["risk_level"].isin(selected_levels) | (~overview["investigated"])
    if search_term:
        mask &= overview["claim_id"].str.contains(search_term, case=False, na=False)
    filtered = overview[mask].copy()

    # ---- KPIs ----
    st.divider()
    total_claims = len(overview)
    total_investigated = int(overview["investigated"].sum())
    high_critical = int(overview["risk_level"].isin(["high", "critical"]).sum())
    avg_score = overview.loc[overview["investigated"], "risk_score"].mean() if total_investigated else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Claims in book", f"{total_claims:,}")
    k2.metric("Investigated by AI Co-Investigator", f"{total_investigated:,}")
    k3.metric("High/critical risk flagged", f"{high_critical:,}")
    k4.metric("Avg. risk score (investigated)", f"{avg_score:.0f}/100" if avg_score is not None else "—")

    # ---- Charts ----
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Risk level breakdown")
        if total_investigated:
            level_counts = (
                overview.loc[overview["investigated"], "risk_level"]
                .value_counts()
                .reindex(RISK_ORDER, fill_value=0)
            )
            level_counts.index.name = "risk_level"
            level_counts = level_counts.rename("count").reset_index()
            fig = px.bar(
                level_counts,
                x="risk_level",
                y="count",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                category_orders={"risk_level": RISK_ORDER},
            )
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Claims")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No claims investigated yet. Select one below and click 'Investigate now'.")

    with chart_col2:
        st.subheader("fraud_confidence_score distribution (full book)")
        fig2 = px.histogram(overview, x="fraud_confidence_score", nbins=30)
        fig2.update_layout(xaxis_title="fraud_confidence_score", yaxis_title="Claims")
        st.plotly_chart(fig2, width="stretch")

    # ---- Table ----
    st.subheader(f"Claims ({len(filtered):,} shown of {total_claims:,})")
    display_cols = [
        "claim_id", "policy_type", "claim_amount", "fraud_confidence_score",
        "days_to_report", "num_previous_claims", "investigated", "risk_level",
        "risk_score", "agent_confidence",
    ]
    st.dataframe(
        filtered[display_cols].sort_values("fraud_confidence_score", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "claim_amount": st.column_config.NumberColumn("Claim amount", format="$%.2f"),
            "fraud_confidence_score": st.column_config.NumberColumn("Fraud score", format="%.2f"),
            "risk_score": st.column_config.NumberColumn("Risk score", format="%d"),
            "agent_confidence": st.column_config.NumberColumn("Agent confidence", format="%.2f"),
        },
    )

    # ---- Claim detail / investigate ----
    st.divider()
    st.subheader("Investigate / review a claim")

    options = filtered.sort_values("fraud_confidence_score", ascending=False)["claim_id"].tolist()
    if not options:
        st.warning("No claims match the current filters.")
        return

    selected_claim_id = st.selectbox("Claim ID", options)
    _render_claim_detail(agent, assessments, selected_claim_id, output_dir, logger)

    # ---- Validation panel ----
    st.divider()
    with st.expander("Model validation against synthetic ground-truth labels (portfolio-only feature)"):
        _render_validation_panel(overview)


if __name__ == "__main__":
    main()
