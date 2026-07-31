import hashlib
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from catena.lm.fineweb_source import (
    METADATA_USER_AGENT,
    FineWebInventory,
    FineWebShard,
    download_and_verify,
    even_grid_indices,
    resolve_inventory,
    snapshot_source_metadata,
)


class _FakeApi:
    def dataset_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
        assert repo_id == "HuggingFaceFW/fineweb-edu"
        return SimpleNamespace(sha=revision, card_data={"license": "odc-by"})

    def list_repo_tree(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
        assert kwargs["path_in_repo"] == "sample/10BT"
        return [
            {
                "path": f"sample/10BT/{index:03d}_00000.parquet",
                "size": index + 1,
                "oid": f"{index:040x}",
                "lfs": {"oid": f"{index + 1:064x}"},
                "xetHash": f"{index + 2:064x}",
            }
            for index in reversed(range(14))
        ]


def test_locked_source_inventory_uses_prospective_even_grid() -> None:
    inventory = resolve_inventory(api=_FakeApi())
    assert inventory.initial_indices == (0, 4, 9, 13)
    assert inventory.expansion_indices == (0, 2, 4, 6, 7, 9, 11, 13)
    assert inventory.missing_expansion_indices == (2, 6, 7, 11)
    assert inventory.selected_indices(1) == (0, 2, 4, 9, 13)
    assert [item.path for item in inventory.all_shards] == sorted(
        item.path for item in inventory.all_shards
    )
    assert even_grid_indices(14, 4) == inventory.initial_indices


def test_download_verifies_actual_bytes_against_lfs_sha(tmp_path: Path) -> None:
    payloads = [b"a", b"bb", b"ccc", b"dddd"]
    shards = tuple(
        FineWebShard(
            path=f"sample/10BT/{index:03d}_00000.parquet",
            size=len(payload),
            git_oid=f"{index:040x}",
            lfs_sha256=hashlib.sha256(payload).hexdigest(),
            xet_hash=None,
        )
        for index, payload in enumerate(payloads)
    )
    inventory = FineWebInventory(
        dataset_id="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        license="odc-by",
        all_shards=shards,
        initial_indices=(0, 1, 2, 3),
        expansion_indices=(0, 1, 2, 3),
    )

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        index = int(Path(filename).name.split("_")[0])
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[index])
        return str(path)

    rows = download_and_verify(inventory, tmp_path / "download", downloader=downloader)
    assert len(rows) == 4
    assert all(row["verified"] for row in rows)


def test_metadata_snapshot_sends_identity_request_and_records_response(
    tmp_path: Path,
) -> None:
    requests: list[urllib.request.Request] = []

    class Response:
        status = 200
        headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": "6",
            "ETag": '"locked"',
        }

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b"locked"

        def geturl(self) -> str:
            return "https://resolved.example/locked"

    def opener(request: urllib.request.Request, *, timeout: int) -> Response:
        assert timeout == 60
        requests.append(request)
        return Response()

    receipt = snapshot_source_metadata(tmp_path / "metadata", opener=opener)
    assert len(requests) == 2
    assert all(request.get_header("User-agent") == METADATA_USER_AGENT for request in requests)
    assert all(request.get_header("Accept-encoding") == "identity" for request in requests)
    assert all(row["response"]["status_code"] == 200 for row in receipt["files"])
    assert all(
        row["response"]["final_url"] == "https://resolved.example/locked"
        for row in receipt["files"]
    )
