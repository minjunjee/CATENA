#!/usr/bin/env python3
"""Resolve the locked FineWeb-Edu inventory and optionally download selected shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from catena.core.provenance_v61 import sha256_canonical_json, write_json_strict
from catena.lm.fineweb_source import (
    download_and_verify,
    resolve_inventory,
    snapshot_source_metadata,
    write_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-output", required=True)
    parser.add_argument("--download-root")
    parser.add_argument("--download-receipt")
    parser.add_argument("--metadata-root")
    parser.add_argument("--expansion-additions", type=int, default=0)
    parser.add_argument("--capacity-prior-validation-tokens", type=int)
    parser.add_argument("--capacity-required-validation-tokens", type=int)
    parser.add_argument("--capacity-prior-build")
    args = parser.parse_args()
    inventory = resolve_inventory()
    inventory_path = write_inventory(args.inventory_output, inventory)
    print(f"source inventory: {inventory_path.resolve()}")
    if args.metadata_root is not None:
        metadata = snapshot_source_metadata(args.metadata_root)
        print(f"source metadata: {metadata['metadata_sha256']}")
    if args.download_root is None:
        return 0
    if args.download_receipt is None:
        parser.error("--download-receipt is required with --download-root")
    rows = download_and_verify(
        inventory,
        args.download_root,
        expansion_additions=args.expansion_additions,
    )
    if args.expansion_additions:
        if (
            args.capacity_prior_validation_tokens is None
            or args.capacity_required_validation_tokens is None
            or args.capacity_prior_build is None
        ):
            parser.error(
                "capacity-triggered expansion requires prior/required validation "
                "tokens and prior build path"
            )
        if (
            args.capacity_prior_validation_tokens
            >= args.capacity_required_validation_tokens
        ):
            parser.error("expansion is forbidden when the prior capacity gate passed")
    selected_indices = inventory.selected_indices(args.expansion_additions)
    added_indices = inventory.missing_expansion_indices[: args.expansion_additions]
    payload = {
        "schema_version": "catena-e26-fineweb-download-v1",
        "scientific_evidence": False,
        "inventory_sha256": inventory.as_dict()["inventory_sha256"],
        "dataset_id": inventory.dataset_id,
        "subset": inventory.subset,
        "revision": inventory.revision,
        "license": inventory.license,
        "selection_policy": "INITIAL_PLUS_PREFIX_OF_MISSING_8_GRID_V1",
        "initial_indices": list(inventory.initial_indices),
        "expansion_grid_indices": list(inventory.expansion_indices),
        "expansion_additions": args.expansion_additions,
        "added_indices": list(added_indices),
        "selected_indices": list(selected_indices),
        "capacity_amendment": (
            {
                "trigger": "REGISTERED_MINIMUM_TOKEN_CAPACITY_ONLY",
                "outcome_based_selection": False,
                "prior_validation_tokens": args.capacity_prior_validation_tokens,
                "required_validation_tokens": args.capacity_required_validation_tokens,
                "prior_build_path": str(Path(args.capacity_prior_build).resolve()),
                "prior_build_disposition": "FAILED_CAPACITY_IMMUTABLE",
            }
            if args.expansion_additions
            else None
        ),
        "shards": list(rows),
        "all_verified": True,
    }
    payload["receipt_sha256"] = sha256_canonical_json(payload)
    output = Path(args.download_receipt)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite download receipt: {output}")
    write_json_strict(output, payload)
    print(f"download receipt: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
