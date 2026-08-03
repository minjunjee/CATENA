#!/usr/bin/env python3
"""Render or apply the prospective E26 Final gate-only official-source patch.

This tool is intentionally pinned to one upstream commit and one exact
``lit_gpt/gdn2.py`` byte sequence.  It does not edit a kernel implementation:
the only source substitution is the two official gate-construction statements
in ``GatedDeltaNet2.forward``.  Runtime selection remains explicit through
``self.e26_gate_policy``; a missing or unknown value fails closed.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

PINNED_OFFICIAL_COMMIT = "95709fc250357c2dd109361c353192f2aa5913f9"
PINNED_GDN2_SHA256 = "5d93765adcb4e9bf755e7d4160a01d4e2ee8438ec55759d17903223dd18b0324"
TARGET_RELATIVE_PATH = Path("lit_gpt/gdn2.py")
POLICY_ATTRIBUTE = "e26_gate_policy"
ALLOWED_POLICIES = ("dual_gdn2", "projected_tied_gdn2")

_ORIGINAL_GATE_BLOCK = (
    b"        b = self.b_proj(hidden_states).sigmoid()\n"
    b"        w = self.w_proj(hidden_states).sigmoid()\n"
)

_PATCHED_GATE_BLOCK = (
    b"        b_logits = self.b_proj(hidden_states)\n"
    b"        w_logits = self.w_proj(hidden_states)\n"
    b"        if b_logits.shape != w_logits.shape:\n"
    b"            raise ValueError(\n"
    b'                "E26 Final requires b_proj and w_proj logits with identical shapes; "\n'
    b'                f"got {tuple(b_logits.shape)} and {tuple(w_logits.shape)}"\n'
    b"            )\n"
    b'        gate_policy = getattr(self, "e26_gate_policy", None)\n'
    b'        if gate_policy == "dual_gdn2":\n'
    b"            b = b_logits.sigmoid()\n"
    b"            w = w_logits.sigmoid()\n"
    b'        elif gate_policy == "projected_tied_gdn2":\n'
    b"            tied_gate = ((b_logits + w_logits) / 2.0).sigmoid()\n"
    b"            b = tied_gate\n"
    b"            w = tied_gate\n"
    b"        else:\n"
    b"            raise ValueError(\n"
    b'                "E26 Final requires an explicit e26_gate_policy equal to "\n'
    b'                "\'dual_gdn2\' or \'projected_tied_gdn2\'"\n'
    b"            )\n"
)

_PATCH_SENTINEL = b'gate_policy = getattr(self, "e26_gate_policy", None)'
PatchMode = Literal["render", "apply"]


class E26FinalOfficialPatchError(RuntimeError):
    """Raised when the pinned prospective patch contract is not satisfied."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repo_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise E26FinalOfficialPatchError(
            f"Unable to validate official Git checkout at {repo_root}"
        ) from exc
    return completed.stdout.strip()


def _new_output_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.exists() or candidate.is_symlink():
        raise E26FinalOfficialPatchError(f"Refusing to overwrite output: {candidate}")
    parent = candidate.parent.resolve(strict=True)
    return parent / candidate.name


def _validate_and_patch(repo_root: Path) -> tuple[Path, bytes, bytes]:
    repo = repo_root.expanduser().resolve(strict=True)
    if _git(repo, "rev-parse", "HEAD") != PINNED_OFFICIAL_COMMIT:
        raise E26FinalOfficialPatchError(
            "Official checkout HEAD does not match the pinned E26 Final commit"
        )
    if _git(repo, "ls-files", "--error-unmatch", TARGET_RELATIVE_PATH.as_posix()) != str(
        TARGET_RELATIVE_PATH
    ):
        raise E26FinalOfficialPatchError("Pinned official gate source is not tracked")

    target = (repo / TARGET_RELATIVE_PATH).resolve(strict=True)
    if repo not in target.parents:
        raise E26FinalOfficialPatchError("Official gate source resolves outside the checkout")
    original = target.read_bytes()
    if _PATCH_SENTINEL in original:
        raise E26FinalOfficialPatchError("Official gate source is already E26 Final patched")
    observed_sha = _sha256(original)
    if observed_sha != PINNED_GDN2_SHA256:
        raise E26FinalOfficialPatchError(
            "Official gate source bytes do not match the pinned E26 Final source: "
            f"expected {PINNED_GDN2_SHA256}, got {observed_sha}"
        )
    if original.count(_ORIGINAL_GATE_BLOCK) != 1:
        raise E26FinalOfficialPatchError(
            "Pinned two-line gate-construction anchor is not unique"
        )

    patched = original.replace(_ORIGINAL_GATE_BLOCK, _PATCHED_GATE_BLOCK, 1)
    if patched.replace(_PATCHED_GATE_BLOCK, _ORIGINAL_GATE_BLOCK, 1) != original:
        raise E26FinalOfficialPatchError("Gate-only inverse patch validation failed")
    return target, original, patched


def _unified_diff(original: bytes, patched: bytes) -> bytes:
    before = original.decode("utf-8").splitlines(keepends=True)
    after = patched.decode("utf-8").splitlines(keepends=True)
    rendered = "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=f"a/{TARGET_RELATIVE_PATH.as_posix()}",
            tofile=f"b/{TARGET_RELATIVE_PATH.as_posix()}",
        )
    )
    if not rendered:
        raise E26FinalOfficialPatchError("Prospective official patch is unexpectedly empty")
    return rendered.encode("utf-8")


def apply_e26_final_official_patch(
    *,
    repo_root: Path,
    patch_output: Path,
    receipt_output: Path,
    mode: PatchMode,
) -> dict[str, Any]:
    """Validate pinned bytes, emit a patch receipt, and optionally apply it."""

    if mode not in {"render", "apply"}:
        raise E26FinalOfficialPatchError(f"Unknown patch mode: {mode!r}")
    patch_path = _new_output_path(patch_output)
    receipt_path = _new_output_path(receipt_output)
    if patch_path == receipt_path:
        raise E26FinalOfficialPatchError("Patch and receipt outputs must be different files")

    target, original, patched = _validate_and_patch(repo_root)
    patch_bytes = _unified_diff(original, patched)
    payload: dict[str, Any] = {
        "schema_version": "catena-e26-final-official-gate-patch-v1",
        "status": "APPLIED" if mode == "apply" else "RENDERED_NOT_APPLIED",
        "official_commit": PINNED_OFFICIAL_COMMIT,
        "target_relative_path": TARGET_RELATIVE_PATH.as_posix(),
        "base_file_sha256": _sha256(original),
        "patched_file_sha256": _sha256(patched),
        "unified_diff_sha256": _sha256(patch_bytes),
        "unified_diff_path": str(patch_path),
        "policy_attribute": POLICY_ATTRIBUTE,
        "allowed_policy_values": list(ALLOWED_POLICIES),
        "explicit_policy_required": True,
        "projection_heads_preserved": ["b_proj", "w_proj"],
        "kernel_calls_modified": False,
        "allow_neg_eigval_contract": "ENFORCED_FALSE_OUTSIDE_THIS_GATE_ONLY_PATCH",
    }
    receipt_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

    # Both outputs are opened exclusively. The official file is written only
    # after every validation and only under an explicit apply mode.
    with patch_path.open("xb") as stream:
        stream.write(patch_bytes)
    with receipt_path.open("xb") as stream:
        stream.write(receipt_bytes)
    if mode == "apply":
        target.write_bytes(patched)
        if _sha256(target.read_bytes()) != payload["patched_file_sha256"]:
            raise E26FinalOfficialPatchError("Patched official source failed post-write hashing")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the hash-locked E26 Final gate-only official GDN-2 patch"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    selected_mode: PatchMode = "apply" if args.apply else "render"
    payload = apply_e26_final_official_patch(
        repo_root=args.repo_root,
        patch_output=args.patch_output,
        receipt_output=args.receipt_output,
        mode=selected_mode,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
