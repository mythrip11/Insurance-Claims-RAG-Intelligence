#!/usr/bin/env python3
"""
Insurance Claims Synthetic Data Generator
==========================================

Production-grade script to generate realistic synthetic insurance claims data
with seeded fraud patterns for RAG pipeline testing and development.

Generates:
  - 1,000 synthetic insurance claims (CSV)
  - 8 realistic policy/SOP documents (TXT)
  - Reproducible fraud patterns via random seed

Author: Insurance Claims RAG System
Version: 1.0.0
"""

import os
import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

try:
    from faker import Faker
except ImportError:
    print("=" * 80)
    print("ERROR: Missing required dependency 'faker'")
    print("=" * 80)
    print(
        "\nThis script needs the 'faker' package to generate synthetic data, "
        "but it isn't installed in the Python environment you're using to run "
        "this script.\n"
    )
    print("Fix -- run ONE of the following in your terminal:\n")
    print("  Option A (install just this package):")
    print("    pip install faker\n")
    print("  Option B (install everything this project needs -- recommended):")
    print("    pip install -r requirements.txt")
    print("    (adjust the path if requirements.txt lives in a different folder)\n")
    print(
        "If you're running inside a conda 'base' environment, consider creating "
        "a dedicated virtual environment first to avoid dependency conflicts:\n"
    )
    print("    python3 -m venv venv")
    print("    source venv/bin/activate   # Windows: venv\\Scripts\\activate")
    print("    pip install -r requirements.txt\n")
    print("Then re-run this script.")
    print("=" * 80)
    sys.exit(1)


# ============================================================================
# Configuration & Setup
# ============================================================================

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    random_seed: int = 42
    num_claims: int = 1000
    num_policies: int = 8
    fraud_rate: float = 0.06  # 6% fraud prevalence
    output_dir: str = "./data/synthetic"
    log_level: str = "INFO"


def setup_logging(config: SyntheticDataConfig) -> logging.Logger:
    """Configure structured logging."""
    logger = logging.getLogger(__name__)
    logger.setLevel(config.log_level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ============================================================================
# Fraud Pattern Definitions
# ============================================================================

class FraudPatterns:
    """Defines realistic fraud patterns for synthetic data generation."""

    # Suspicious claim amounts (% over policy limits)
    HIGH_CLAIM_AMOUNTS = [5000, 8000, 12000, 15000, 20000, 25000]

    # Common fraud indicators
    FRAUD_INDICATORS = {
        "multiple_claims": {
            "description": "Claimant filed 3+ claims in 6 months",
            "weight": 0.15,
        },
        "short_policy_coverage": {
            "description": "Claim filed within 90 days of policy start",
            "weight": 0.12,
        },
        "inflated_loss_value": {
            "description": "Claim amount is 150%+ of estimated value",
            "weight": 0.18,
        },
        "round_numbers": {
            "description": "Claim amount is exact round number (1000, 5000, etc.)",
            "weight": 0.08,
        },
        "missing_documentation": {
            "description": "Incomplete or inconsistent supporting docs",
            "weight": 0.10,
        },
        "inconsistent_narrative": {
            "description": "Loss description conflicts with damage evidence",
            "weight": 0.14,
        },
        "delayed_reporting": {
            "description": "Claim filed 30+ days after loss date",
            "weight": 0.12,
        },
        "suspicious_repair_quotes": {
            "description": "Repair estimates from unknown/unverified vendors",
            "weight": 0.11,
        },
    }


class ClaimsDataGenerator:
    """Generates synthetic insurance claims with realistic fraud patterns."""

    def __init__(self, config: SyntheticDataConfig, logger: logging.Logger):
        """Initialize generator with config."""
        self.config = config
        self.logger = logger
        self.faker = Faker()

        # Seed for reproducibility
        random.seed(config.random_seed)
        Faker.seed(config.random_seed)

        self.logger.info(f"Initialized generator with seed: {config.random_seed}")

    def generate_claims(self) -> List[Dict[str, Any]]:
        """Generate synthetic claims with seeded fraud patterns."""
        self.logger.info(f"Generating {self.config.num_claims} synthetic claims...")

        claims = []
        num_frauds = int(self.config.num_claims * self.config.fraud_rate)

        # Determine which claims will be fraudulent (seeded)
        fraud_indices = set(random.sample(range(self.config.num_claims), num_frauds))

        for claim_id in range(1, self.config.num_claims + 1):
            is_fraud = claim_id in fraud_indices
            claim = self._generate_single_claim(claim_id, is_fraud)
            claims.append(claim)

        self.logger.info(f"✅ Generated {len(claims)} claims ({num_frauds} fraudulent)")
        return claims

    def _generate_single_claim(self, claim_id: int, is_fraud: bool) -> Dict[str, Any]:
        """Generate a single claim record."""

        # Base claim data
        loss_date = self.faker.date_between(start_date="-2y", end_date="today")
        report_date = loss_date + timedelta(
            days=random.randint(1 if is_fraud else 1, 60 if is_fraud else 7)
        )

        claim_amount = self._generate_claim_amount(is_fraud)
        policy_limit = 50000 if random.random() < 0.5 else 100000

        # Fraud-specific patterns
        fraud_indicators = []
        if is_fraud:
            fraud_indicators = self._select_fraud_indicators()

        # Metadata for fraud detection
        days_to_report = (report_date - loss_date).days
        claim_as_pct_of_limit = (claim_amount / policy_limit) * 100

        return {
            "claim_id": f"CLM-{claim_id:06d}",
            "claimant_name": self.faker.name(),
            "claimant_email": self.faker.email(),
            "claimant_phone": self.faker.phone_number(),
            "policy_id": f"POL-{random.randint(100000, 999999)}",
            "policy_type": random.choice(["auto", "home", "umbrella"]),
            "policy_start_date": (loss_date - timedelta(days=random.randint(30, 1000))).strftime("%Y-%m-%d"),
            "loss_date": loss_date.strftime("%Y-%m-%d"),
            "loss_description": self.faker.sentence(nb_words=10),
            "report_date": report_date.strftime("%Y-%m-%d"),
            "claim_amount": claim_amount,
            "policy_limit": policy_limit,
            "claim_as_pct_of_limit": round(claim_as_pct_of_limit, 2),
            "days_to_report": days_to_report,
            "estimated_loss_value": self._generate_estimated_value(claim_amount, is_fraud),
            "num_previous_claims": random.randint(0 if not is_fraud else 2, 5),
            "repair_vendor": self.faker.company(),
            "repair_estimate": round(claim_amount * random.uniform(0.9, 1.2), 2),
            "supporting_docs_count": random.randint(1 if is_fraud else 3, 8),
            "adjuster_notes": self.faker.sentence(nb_words=15),
            "fraud_indicators": "|".join(fraud_indicators) if fraud_indicators else "NONE",
            "is_fraud": "TRUE" if is_fraud else "FALSE",
            "fraud_confidence_score": self._calculate_fraud_score(fraud_indicators),
        }

    def _generate_claim_amount(self, is_fraud: bool) -> float:
        """Generate claim amount with fraud-specific logic."""
        if is_fraud:
            # Fraudulent claims often use specific round numbers
            if random.random() < 0.4:
                return float(random.choice(FraudPatterns.HIGH_CLAIM_AMOUNTS))
            else:
                return round(random.uniform(5000, 25000) / 1000) * 1000
        else:
            return round(random.uniform(1000, 20000), 2)

    def _generate_estimated_value(self, claim_amount: float, is_fraud: bool) -> float:
        """Generate estimated loss value (often inflated in fraud)."""
        if is_fraud:
            # Claim amount is 150%+ of estimated value
            estimated = claim_amount / random.uniform(1.2, 1.8)
        else:
            # Legitimate: close to actual claim
            estimated = claim_amount * random.uniform(0.8, 1.1)

        return round(estimated, 2)

    def _select_fraud_indicators(self) -> List[str]:
        """Select realistic fraud indicators for fraudulent claims."""
        selected = []
        for indicator, properties in FraudPatterns.FRAUD_INDICATORS.items():
            if random.random() < properties["weight"]:
                selected.append(indicator)

        # Ensure at least 1 fraud indicator per fraudulent claim
        if not selected:
            selected.append(random.choice(list(FraudPatterns.FRAUD_INDICATORS.keys())))

        return selected

    def _calculate_fraud_score(self, fraud_indicators: List[str]) -> float:
        """Calculate fraud confidence score based on indicators."""
        if not fraud_indicators:
            return 0.0

        score = sum(
            FraudPatterns.FRAUD_INDICATORS[ind]["weight"]
            for ind in fraud_indicators
        )
        return round(min(score, 1.0), 3)


class PolicyDocumentGenerator:
    """Generates synthetic policy and SOP documents."""

    POLICY_TEMPLATES = [
        """COMMERCIAL AUTO INSURANCE POLICY
Version: 2024-Q1
Effective Date: January 1, 2024

COVERAGE LIMITS:
- Bodily Injury Liability: $100,000 per person / $300,000 per accident
- Property Damage Liability: $50,000 per accident
- Uninsured Motorist: $50,000 per person

EXCLUSIONS:
1. Claims filed more than 90 days after loss date
2. Intentional loss or damage
3. Racing, speed contests, or illegal activity
4. Mechanical or electrical failure not caused by collision

CLAIMS PROCEDURE:
1. Report claim within 7 days of loss
2. Submit complete documentation (photos, police report)
3. Schedule adjuster inspection within 14 days
4. Submit repair estimates from licensed vendors

DEDUCTIBLES:
- Collision: $500
- Comprehensive: $250
- Liability: $0
""",
        """HOMEOWNERS INSURANCE POLICY
Contract Number: HOP-2024-001
Policy Period: 12 months

COVERED PERILS:
- Fire and smoke damage
- Theft and vandalism
- Weather damage (hail, wind)
- Liability for bodily injury

NOT COVERED:
- Flood damage (separate policy required)
- Earthquake
- Wear and tear
- Claims without supporting evidence

COVERAGE AMOUNTS:
- Dwelling: $300,000
- Personal Property: $150,000
- Liability: $100,000

REPORTING REQUIREMENTS:
- Call within 24 hours of loss
- Preserve all damage evidence
- Obtain written estimates (minimum 2)
- File claim within 6 months
""",
        """FRAUD INVESTIGATION & CLAIMS SOP
Version: 3.2 | Updated: 2024-01-15

RED FLAGS FOR INVESTIGATION:
1. Multiple claims within 12 months (3+)
2. Claim filed within 90 days of policy inception
3. Claim amount significantly exceeds policy estimate
4. Missing or inconsistent documentation
5. Unusually high repair costs from unverified vendors
6. Claimant unavailable for inspection
7. Loss description conflicts with photographic evidence

INVESTIGATIVE STEPS:
1. INITIAL REVIEW (Day 1-2)
   - Verify policy status and coverage
   - Validate claimant information
   - Obtain loss description narrative

2. DOCUMENTATION REVIEW (Day 3-5)
   - Collect supporting documents
   - Verify third-party reports (police, fire)
   - Obtain repair estimates (3+ vendors)

3. ON-SITE INSPECTION (Day 6-10)
   - Photograph all damage
   - Compare damage to claim narrative
   - Interview claimant and witnesses

4. RISK ASSESSMENT (Day 11-14)
   - Calculate fraud risk score
   - Review prior claim history
   - Identify similar patterns in database

5. DECISION & RESOLUTION (Day 15-21)
   - Approve, deny, or request additional info
   - Document all findings
   - Close case with summary report

FRAUD RISK SCORING:
- Score 0.0-0.2: Low risk - Process normally
- Score 0.2-0.5: Medium risk - Extended investigation
- Score 0.5-0.8: High risk - Referral to SIU
- Score 0.8-1.0: Critical - Deny and report to authorities
""",
        """CLAIMS ADJUSTER GUIDELINES
Revision: 2.1

DOCUMENTATION REQUIREMENTS:
All claims must include:
✓ Completed claim form (signed and dated)
✓ Photo evidence (before/after, damage, serial numbers)
✓ Receipt or proof of ownership
✓ Police report (if applicable)
✓ Repair estimates (2-3 quotes minimum)
✓ Receipts for completed repairs

LOSS VALUATION PRINCIPLES:
- Actual Cash Value (ACV) = Replacement Cost - Depreciation
- Depreciation applies to items over 5 years old
- No coverage for labor if DIY work performed
- Parts replacement price + reasonable labor only

COMMON CLAIM ERRORS TO AVOID:
1. Accepting round-number claims without verification
2. Approving claims without proper photo documentation
3. Processing claims from unverified vendors
4. Ignoring prior claim history flags
5. Skipping on-site inspection for high-value claims

DISPUTE RESOLUTION:
If claimant disputes denial:
1. Provide detailed written explanation
2. Reference specific policy language
3. Offer appeal process information
4. Document all communications
""",
        """INSURANCE POLICY UNDERWRITING STANDARDS
Effective: 2024-01-01

RISK ASSESSMENT CRITERIA:
1. APPLICANT HISTORY
   - No more than 2 claims in 5 years (new policies)
   - Minimum 3 months since last claim
   - Clean driving record or equivalent

2. PROPERTY CONDITIONS
   - Maintained in good repair
   - No evidence of previous loss
   - Security systems for high-value items

3. COVERAGE RECOMMENDATIONS
   - Auto: Minimum 100/300/50 liability limits
   - Home: Replacement cost endorsement advised
   - Umbrella: Recommended if assets exceed $500K

EXCLUSIONS SUMMARY:
- Intentional damage or loss
- Violation of policy conditions
- Use for commercial purposes (unless endorsed)
- Claims involving fraud or misrepresentation

PREMIUM CALCULATION FACTORS:
- Loss history (past 5 years)
- Age and condition of property
- Geographic location and risk
- Type and amount of coverage
- Claims frequency (key variable)
""",
        """DIGITAL DOCUMENTATION & CLAIMS SUBMISSION GUIDE
Version: 1.0

ACCEPTED FILE FORMATS:
✓ PDF documents
✓ JPG/PNG photos (2MB max per file)
✓ MP4 video evidence
✓ Scanned receipts
✗ Faxes (poor quality)
✗ Original documents without scans

PHOTO SUBMISSION GUIDELINES:
1. Wide shots showing full extent of damage
2. Close-ups of specific damage areas
3. Evidence of ownership (serial numbers, labels)
4. Timestamp or date stamp on photos
5. Include reference objects for scale

ONLINE PORTAL REQUIREMENTS:
- Username must match policy holder name
- All dates in MM/DD/YYYY format
- File size limit: 50MB per submission
- Keep password secure (never share)
- Save confirmation number for reference

SECURITY & PRIVACY:
- All data encrypted in transit (HTTPS)
- PII protected per CCPA regulations
- Access logs maintained for audit
- Third-party access requires authorization
""",
        """AUTO INSURANCE CLAIM SPECIFICS
Policy Type: CA, UT, WA, CO

COVERAGE TYPES:
- Liability: Injury/damage you cause to others
- Collision: Damage from collision (regardless of fault)
- Comprehensive: Non-collision damage (theft, weather)
- Uninsured Motorist: Hit by uninsured driver

COMMON AUTO CLAIM ISSUES:
1. Lack of police report (required for theft)
2. Contradictory accident descriptions
3. Inflated repair estimates
4. Failure to pursue uninsured motorist recovery
5. Claims involving commercial use

REPAIR REQUIREMENTS:
- Use network or pre-approved shops
- Get estimate approval before repairs
- Submit repair invoices within 30 days
- Keep damaged parts for inspection

RENTAL CAR COVERAGE:
- Typically $30/day for 30 days maximum
- Must be through approved rental companies
- Provided at no cost if comprehensive/collision
""",
        """HEALTH & DISABILITY INSURANCE CLAIMS
Effective: 2024-Q1

CLAIM DOCUMENTATION:
Medical claims require:
- Provider invoice
- Itemized bill with diagnosis/procedure codes
- Proof of payment
- EOB from primary insurance (if applicable)

DISABILITY CLAIMS require:
- Attending physician statement
- Proof of inability to work
- Tax return (last 2 years)
- Detailed treatment plan

COMMON DENIAL REASONS:
1. Pre-existing condition exclusion
2. Exceeds annual benefit limit
3. Non-covered service
4. Claim filed outside time limit

APPEAL PROCESS:
1. Request detailed denial reason (5 days)
2. Gather additional medical evidence
3. Submit written appeal with documentation
4. Review committee decision (30 days)
5. Independent external review (if denied again)
""",
    ]

    # Descriptive filename slugs, one per entry in POLICY_TEMPLATES (same order).
    # Kept as a parallel list rather than embedded in each template string so
    # the templates themselves stay clean and easy to edit.
    POLICY_FILENAME_SLUGS = [
        "commercial_auto_insurance",
        "homeowners_insurance",
        "fraud_investigation_sop",
        "claims_adjuster_guidelines",
        "underwriting_standards",
        "digital_documentation_guide",
        "auto_insurance_claim_specifics",
        "health_disability_claims",
    ]

    def __init__(self, logger: logging.Logger):
        """Initialize policy generator."""
        self.logger = logger

        if len(self.POLICY_FILENAME_SLUGS) != len(self.POLICY_TEMPLATES):
            raise ValueError(
                "POLICY_FILENAME_SLUGS and POLICY_TEMPLATES must be the same "
                f"length (got {len(self.POLICY_FILENAME_SLUGS)} slugs vs "
                f"{len(self.POLICY_TEMPLATES)} templates). Add or remove a "
                "slug to match any template you add or remove."
            )

    def generate_policies(self, output_dir: str) -> List[str]:
        """Generate synthetic policy documents with descriptive filenames."""
        self.logger.info(f"Generating {len(self.POLICY_TEMPLATES)} policy documents...")

        policies_dir = Path(output_dir) / "policies"
        policies_dir.mkdir(parents=True, exist_ok=True)

        # Clear out previously generated policy files first. This matters
        # because an earlier version of this script used generic filenames
        # (policy_01_underwriting_guide.txt, etc.) -- without this cleanup,
        # re-running after a filename scheme change would leave old and new
        # files sitting side by side instead of replacing them.
        removed = 0
        for existing_file in policies_dir.glob("policy_*.txt"):
            existing_file.unlink()
            removed += 1
        if removed:
            self.logger.info(f"Removed {removed} previously generated policy file(s)")

        policy_files = []
        for i, (slug, template) in enumerate(
            zip(self.POLICY_FILENAME_SLUGS, self.POLICY_TEMPLATES), 1
        ):
            filename = f"policy_{i:02d}_{slug}.txt"
            filepath = policies_dir / filename

            with open(filepath, "w") as f:
                f.write(template)

            policy_files.append(str(filepath))

        self.logger.info(f"✅ Generated {len(policy_files)} policy documents")
        return policy_files


class SyntheticDataExporter:
    """Exports generated data to files."""

    def __init__(self, logger: logging.Logger):
        """Initialize exporter."""
        self.logger = logger

    def export_claims_csv(
        self,
        claims: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """Export claims to CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not claims:
            self.logger.warning("No claims to export")
            return ""

        fieldnames = claims[0].keys()

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(claims)

        self.logger.info(f"✅ Exported {len(claims)} claims to {output_path}")
        return str(output_path)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""

    # Configuration
    config = SyntheticDataConfig(
        random_seed=42,
        num_claims=1000,
        num_policies=8,
        fraud_rate=0.06,
        output_dir="./data/synthetic",
    )

    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("Insurance Claims Synthetic Data Generator - Step 1")
    logger.info("=" * 80)

    # Ensure output directory exists
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate claims
    claims_generator = ClaimsDataGenerator(config, logger)
    synthetic_claims = claims_generator.generate_claims()

    # Export claims
    exporter = SyntheticDataExporter(logger)
    claims_csv_path = exporter.export_claims_csv(
        synthetic_claims,
        output_dir / "synthetic_claims.csv"
    )

    # Generate policies
    policy_generator = PolicyDocumentGenerator(logger)
    policy_files = policy_generator.generate_policies(config.output_dir)

    # Summary report
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Claims generated: {len(synthetic_claims)}")
    logger.info(f"   - Fraudulent: {sum(1 for c in synthetic_claims if c['is_fraud'] == 'TRUE')}")
    logger.info(f"   - Legitimate: {sum(1 for c in synthetic_claims if c['is_fraud'] == 'FALSE')}")
    logger.info(f"✅ Claims CSV exported to: {claims_csv_path}")
    logger.info(f"✅ Policy documents generated: {len(policy_files)}")
    logger.info(f"   Location: {output_dir}/policies/")
    logger.info("=" * 80)

    return synthetic_claims, policy_files


if __name__ == "__main__":
    synthetic_claims, policy_files = main()
