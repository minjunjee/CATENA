#!/usr/bin/env python3
"""Validate the E26 Final speed gate and select a non-evidence token budget."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import (
    dumps_json_strict,
    read_json_object_strict,
    sha256_canonical_json,
    write_json_strict,
)
from catena.lm.e26_final_resources import (
    E26FinalResourceError,
    policy_from_mapping,
    select_token_budget,
    speed_observation_from_mapping,
)

REQUEST_SCHEMA_VERSION = "catena-e26-final-resource-request-v1"
RECEIPT_SCHEMA_VERSION = "catena-e26-final-resource-selection-v1"
REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "bridge_hours",
        "observations",
    }
)


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise E26FinalResourceError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise E26FinalResourceError(f"{field} keys must be strings")
    return value


def select_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic receipt from systems-only measurements.

    The exact top-level schema intentionally has no field for task loss,
    accuracy, effect size, or any other scientific outcome.
    """

    unknown = sorted(set(payload).difference(REQUEST_FIELDS))
    missing = sorted(REQUEST_FIELDS.difference(payload))
    if unknown or missing:
        raise E26FinalResourceError(
            f"Resource request fields differ; missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise E26FinalResourceError(f"schema_version must be exactly {REQUEST_SCHEMA_VERSION!r}")

    policy_payload = _require_mapping(payload["policy"], "policy")
    policy = policy_from_mapping(policy_payload)
    observation_payloads = payload["observations"]
    if not isinstance(observation_payloads, list):
        raise E26FinalResourceError("observations must be a list")
    observations = tuple(
        speed_observation_from_mapping(_require_mapping(row, f"observations[{index}]"))
        for index, row in enumerate(observation_payloads)
    )
    selection = select_token_budget(
        observations,
        bridge_hours=payload["bridge_hours"],
        policy=policy,
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "run_mode": "NON_EVIDENCE_SPEED_PREFLIGHT",
        "scientific_evidence": False,
        "scientific_e26a_started": False,
        "outcome_inputs_used": False,
        "request_sha256": sha256_canonical_json(dict(payload)),
        "policy": policy.as_dict(),
        "bridge_hours": float(payload["bridge_hours"]),
        "selection": selection.as_dict(),
    }
    receipt["receipt_sha256"] = sha256_canonical_json(receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Strict systems-only request JSON")
    parser.add_argument("--output", help="New receipt path; stdout when omitted")
    args = parser.parse_args(argv)

    try:
        receipt = select_from_payload(read_json_object_strict(args.input))
        if args.output is None:
            print(dumps_json_strict(receipt, indent=2))
        else:
            output = Path(args.output)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Refusing to overwrite resource receipt: {output}")
            write_json_strict(output, receipt)
    except (E26FinalResourceError, FileExistsError, OSError, ValueError) as error:
        print(f"E26 Final resource selection error: {error}", file=sys.stderr)
        return 2

    return 0 if receipt["selection"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
