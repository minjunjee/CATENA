from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ClaimDecision:
    claim_id: str
    allowed: bool
    reason: str
    tier: str


def evaluate_claims(evidence: dict[str, Any]) -> list[ClaimDecision]:
    h1 = bool(evidence.get("h1", False)); h2 = bool(evidence.get("h2", False)); h3 = bool(evidence.get("h3", False)); h4 = bool(evidence.get("h4", False)); h5 = bool(evidence.get("h5_direction", False)) and bool(evidence.get("h5_audit", False)); h6 = bool(evidence.get("h6", False)); rqt = bool(evidence.get("rqt", False))
    return [
        ClaimDecision("constrained_behavioral_reachability",h1,"Requires unseen-geometry behavioral-regret calibration with operation fixed effects.","workshop_core"),
        ClaimDecision("erase_write_magnitude_factorization",h1 and h2,"Requires H1 plus all H2 absolute-gain, equivalence, interaction, retention, and tuning guardrails.","workshop_core"),
        ClaimDecision("joint_diagonalizability_principle",h3,"Requires common-rotation recovery and a residual noncommuting gap recovered by richer control.","theory_branch"),
        ClaimDecision("operation_specific_functional_mediation",h1 and h2 and h4,"Requires H1-H2 plus dose, specificity, transplant, and rescue across paired seeds.","workshop_core_strong"),
        ClaimDecision("semantic_external_validity_anchor",h1 and h2 and h5,"Requires positive held-out-SUPERSEDE direction and completed two-reviewer audit.","workshop_anchor"),
        ClaimDecision("reusable_state_assimilation",h1 and h2 and h6,"Requires multi-update robustness and quality-constrained break-even against external and cached baselines.","post_workshop"),
        ClaimDecision("operation_matched_transformer_boundary",rqt,"Requires pinned official backend, localization parity, own-oracle normalization, and measured cost.","post_workshop"),
    ]
