"""Pinned FineWeb-Edu source inventory and download verification.

This module contains no import-time dependency on Hugging Face tooling.  The
optional package is imported only by the explicit data-preparation command.
Scientific training consumes the resulting immutable manifests, never a moving
Hub branch or an implicit streaming dataset.
"""

from __future__ import annotations

import importlib
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from catena.core.provenance_v61 import sha256_canonical_json, sha256_file, write_json_strict

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET = "sample-10BT"
DATASET_PATH = "sample/10BT"
LOCKED_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
EXPECTED_LICENSE = "odc-by"
METADATA_USER_AGENT = (
    "Mozilla/5.0 (compatible; CATENA-E26-Data-Lock/1.0; "
    "+https://github.com/minjunjee/CATENA)"
)
METADATA_REQUEST_HEADERS = {
    "Accept": "text/plain,text/markdown,text/html;q=0.9,*/*;q=0.1",
    "Accept-Encoding": "identity",
    "User-Agent": METADATA_USER_AGENT,
}


class FineWebSourceError(RuntimeError):
    """Raised when the pinned external source differs from its prospective lock."""


@dataclass(frozen=True, slots=True)
class FineWebShard:
    path: str
    size: int
    git_oid: str
    lfs_sha256: str
    xet_hash: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FineWebInventory:
    dataset_id: str
    subset: str
    revision: str
    license: str
    all_shards: tuple[FineWebShard, ...]
    initial_indices: tuple[int, ...]
    expansion_indices: tuple[int, ...]

    @property
    def initial_shards(self) -> tuple[FineWebShard, ...]:
        return tuple(self.all_shards[index] for index in self.initial_indices)

    @property
    def expansion_shards(self) -> tuple[FineWebShard, ...]:
        return tuple(self.all_shards[index] for index in self.expansion_indices)

    @property
    def missing_expansion_indices(self) -> tuple[int, ...]:
        initial = set(self.initial_indices)
        return tuple(index for index in self.expansion_indices if index not in initial)

    def selected_indices(self, expansion_additions: int = 0) -> tuple[int, ...]:
        if (
            isinstance(expansion_additions, bool)
            or expansion_additions < 0
            or expansion_additions > len(self.missing_expansion_indices)
        ):
            raise ValueError(
                "expansion_additions must be within the missing expansion-grid prefix"
            )
        return tuple(
            sorted(
                (
                    *self.initial_indices,
                    *self.missing_expansion_indices[:expansion_additions],
                )
            )
        )

    def payload_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": "catena-e26-fineweb-source-v1",
            "manifest_type": "E26_PINNED_FINEWEB_INVENTORY",
            "scientific_evidence": False,
            "dataset_id": self.dataset_id,
            "subset": self.subset,
            "revision": self.revision,
            "license": self.license,
            "selection_algorithm": "EVEN_GRID_FLOOR_X_PLUS_HALF_V1",
            "all_shards": [item.as_dict() for item in self.all_shards],
            "initial_indices": list(self.initial_indices),
            "expansion_indices": list(self.expansion_indices),
            "initial_bytes": sum(item.size for item in self.initial_shards),
            "expansion_bytes": sum(item.size for item in self.expansion_shards),
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.payload_without_hash()
        payload["inventory_sha256"] = sha256_canonical_json(payload)
        return payload


def even_grid_indices(item_count: int, grid_count: int) -> tuple[int, ...]:
    """Return the preregistered evenly spaced indices without language-specific round()."""

    if item_count < 1:
        raise ValueError("item_count must be positive")
    if grid_count < 1 or grid_count > item_count:
        raise ValueError("grid_count must be in [1,item_count]")
    if grid_count == 1:
        return (0,)
    denominator = 2 * (grid_count - 1)
    indices = tuple(
        (2 * index * (item_count - 1) + (grid_count - 1)) // denominator
        for index in range(grid_count)
    )
    if len(set(indices)) != grid_count:
        raise FineWebSourceError("Even-grid selection produced duplicate indices")
    return indices


def _value(value: Any, *names: str) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _shard_from_entry(entry: Any) -> FineWebShard:
    path = _value(entry, "path", "rfilename")
    size = _value(entry, "size")
    git_oid = _value(entry, "blob_id", "oid")
    lfs = _value(entry, "lfs")
    lfs_sha = _value(lfs, "sha256", "oid")
    xet_hash = _value(entry, "xet_hash", "xetHash")
    if not isinstance(path, str) or not path.endswith(".parquet"):
        raise FineWebSourceError(f"Unexpected source entry path: {path!r}")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise FineWebSourceError(f"Missing source byte size for {path}")
    if not isinstance(git_oid, str) or len(git_oid) != 40:
        raise FineWebSourceError(f"Missing Git blob oid for {path}")
    if not isinstance(lfs_sha, str) or len(lfs_sha) != 64:
        raise FineWebSourceError(f"Missing LFS SHA-256 for {path}")
    if xet_hash is not None and not isinstance(xet_hash, str):
        raise FineWebSourceError(f"Invalid Xet hash for {path}")
    return FineWebShard(path, size, git_oid, lfs_sha, xet_hash)


def resolve_inventory(*, api: Any | None = None) -> FineWebInventory:
    """Resolve and verify the prospectively pinned source without downloading shards."""

    if api is None:
        try:
            hub = importlib.import_module("huggingface_hub")
        except ModuleNotFoundError as error:
            raise FineWebSourceError(
                "Source resolution requires pinned huggingface_hub==1.26.0"
            ) from error
        api = hub.HfApi()
    info = api.dataset_info(DATASET_ID, revision=LOCKED_REVISION)
    revision = _value(info, "sha")
    if revision != LOCKED_REVISION:
        raise FineWebSourceError(
            f"Pinned revision mismatch: expected {LOCKED_REVISION}, observed {revision}"
        )
    card_data = _value(info, "card_data", "cardData")
    license_value = _value(card_data, "license")
    if license_value != EXPECTED_LICENSE:
        raise FineWebSourceError(
            f"Dataset license mismatch: expected {EXPECTED_LICENSE}, observed {license_value}"
        )
    entries = api.list_repo_tree(
        DATASET_ID,
        path_in_repo=DATASET_PATH,
        recursive=False,
        expand=True,
        revision=LOCKED_REVISION,
        repo_type="dataset",
    )
    shards = tuple(
        sorted(
            (
                _shard_from_entry(item)
                for item in entries
                if str(_value(item, "path")).endswith(".parquet")
            ),
            key=lambda item: item.path,
        )
    )
    if len(shards) != 14:
        raise FineWebSourceError(
            f"Locked sample/10BT inventory requires 14 shards, got {len(shards)}"
        )
    return FineWebInventory(
        dataset_id=DATASET_ID,
        subset=DATASET_SUBSET,
        revision=LOCKED_REVISION,
        license=EXPECTED_LICENSE,
        all_shards=shards,
        initial_indices=even_grid_indices(len(shards), 4),
        expansion_indices=even_grid_indices(len(shards), 8),
    )


def write_inventory(path: str | Path, inventory: FineWebInventory) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite source inventory: {destination}")
    write_json_strict(destination, inventory.as_dict())
    return destination


def download_and_verify(
    inventory: FineWebInventory,
    destination_root: str | Path,
    *,
    expansion_additions: int = 0,
    downloader: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    """Download only the locked shards and verify bytes against their Git-LFS SHA-256."""

    if downloader is None:
        try:
            hub = importlib.import_module("huggingface_hub")
        except ModuleNotFoundError as error:
            raise FineWebSourceError(
                "Source download requires pinned huggingface_hub==1.26.0"
            ) from error
        downloader = hub.hf_hub_download
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    selected_indices = inventory.selected_indices(expansion_additions)
    selected = tuple(inventory.all_shards[index] for index in selected_indices)
    receipts: list[dict[str, Any]] = []
    for shard in selected:
        resolved = Path(
            downloader(
                repo_id=inventory.dataset_id,
                filename=shard.path,
                repo_type="dataset",
                revision=inventory.revision,
                local_dir=str(root),
            )
        ).resolve(strict=True)
        if resolved.stat().st_size != shard.size:
            raise FineWebSourceError(f"Downloaded byte count mismatch for {shard.path}")
        digest = sha256_file(resolved)
        if digest != shard.lfs_sha256:
            raise FineWebSourceError(f"Downloaded SHA-256 mismatch for {shard.path}")
        receipts.append(
            {
                **shard.as_dict(),
                "local_path": str(resolved),
                "local_sha256": digest,
                "verified": True,
            }
        )
    return tuple(receipts)


def snapshot_source_metadata(
    destination_root: str | Path,
    *,
    opener: Any | None = None,
) -> dict[str, Any]:
    """Snapshot the pinned dataset card and official ODC-By page."""

    root = Path(destination_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite source metadata snapshot: {root}")
    root.mkdir(parents=True)
    if opener is None:
        opener = urllib.request.urlopen
    urls = {
        "fineweb_edu_dataset_card.md": (
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/raw/"
            f"{LOCKED_REVISION}/README.md"
        ),
        "odc_by_1_0_license.html": "https://opendatacommons.org/licenses/by/1-0/",
    }
    records: list[dict[str, Any]] = []
    for name, url in urls.items():
        request = urllib.request.Request(
            url,
            headers=METADATA_REQUEST_HEADERS,
            method="GET",
        )
        with opener(request, timeout=60) as response:
            content = response.read()
            response_url = (
                str(response.geturl()) if hasattr(response, "geturl") else url
            )
            status_value = getattr(response, "status", None)
            if status_value is None and hasattr(response, "getcode"):
                status_value = response.getcode()
            status_code = int(status_value) if status_value is not None else 200
            headers = getattr(response, "headers", {})
            response_headers = {
                key.lower(): value
                for key in (
                    "Content-Type",
                    "Content-Length",
                    "ETag",
                    "Last-Modified",
                )
                if (value := headers.get(key)) is not None
            }
        if status_code != 200:
            raise FineWebSourceError(
                f"Metadata snapshot expected HTTP 200 for {url}, got {status_code}"
            )
        path = root / name
        path.write_bytes(content)
        records.append(
            {
                "path": name,
                "url": url,
                "request": {
                    "method": "GET",
                    "headers": dict(sorted(METADATA_REQUEST_HEADERS.items())),
                },
                "response": {
                    "final_url": response_url,
                    "status_code": status_code,
                    "headers": dict(sorted(response_headers.items())),
                },
                "bytes": len(content),
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": "catena-e26-source-metadata-v1",
        "revision": LOCKED_REVISION,
        "expected_license": EXPECTED_LICENSE,
        "files": records,
    }
    payload["metadata_sha256"] = sha256_canonical_json(payload)
    write_json_strict(root / "metadata_receipt.json", payload)
    return payload
