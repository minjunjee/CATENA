from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from catena.core.config import load_config
from catena.data.sequence_control_lattice import (
    SequenceDemandFamily,
    base_sequence_control_digest,
    generate_sequence_control_lattice_batch,
    sequence_control_lattice_model_input,
)
from catena.models.sequence_control_lattice import (
    MatchedSequenceControlLattice,
    SequenceControlFreedom,
    sequence_lattice_parameter_count,
)
from catena.training.sequence_control_lattice import state_dict_sha256
from experiments import e18a_sequence_control_lattice as e18a
from experiments import e18b_sequence_control_lattice_aggregate as e18b

REPO_ROOT = Path(__file__).resolve().parents[1]
E18A_CONFIG = REPO_ROOT / "configs/e18a_sequence_control_lattice.yaml"
E18B_CONFIG = (
    REPO_ROOT / "configs/e18b_sequence_control_lattice_aggregate.yaml"
)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_all_freedoms_share_parameter_surface_and_paired_initialization() -> None:
    counts: set[int] = set()
    hashes: set[str] = set()
    for freedom in SequenceControlFreedom:
        torch.manual_seed(101)
        model = MatchedSequenceControlLattice(
            freedom=freedom,
            num_entities=8,
            value_dim=8,
            embedding_dim=16,
            hidden_dim=32,
        )
        counts.add(sequence_lattice_parameter_count(model))
        hashes.add(state_dict_sha256(model.state_dict()))
    assert len(counts) == 1
    assert len(hashes) == 1


def test_distractor_path_is_model_visible_and_base_is_gap_paired() -> None:
    for family in SequenceDemandFamily:
        common = {
            "family": family,
            "batch_size": 3,
            "num_entities": 8,
            "value_dim": 8,
            "updates": 2,
            "seed": 700 + list(SequenceDemandFamily).index(family),
            "device": torch.device("cpu"),
        }
        no_gap = generate_sequence_control_lattice_batch(
            **common,
            gap_events=0,
        )
        gap4 = generate_sequence_control_lattice_batch(
            **common,
            gap_events=4,
        )
        gap8 = generate_sequence_control_lattice_batch(
            **common,
            gap_events=8,
        )
        assert base_sequence_control_digest(no_gap) == base_sequence_control_digest(
            gap4
        )
        assert base_sequence_control_digest(gap4) == base_sequence_control_digest(
            gap8
        )
        assert gap4.update_mask[0].nonzero().flatten().tolist() == [0, 5]
        assert not hasattr(gap4.inputs, "update_mask")
        assert torch.equal(
            gap4.inputs.erase_entity_ids[:, 1:5],
            gap8.inputs.erase_entity_ids[:, 1:5],
        )
        assert torch.equal(
            gap4.inputs.write_entity_ids[:, 1:5],
            gap8.inputs.write_entity_ids[:, 1:5],
        )
        assert torch.equal(
            gap4.inputs.candidate_values[:, 1:5],
            gap8.inputs.candidate_values[:, 1:5],
        )
        assert torch.equal(
            gap4.inputs.demand_features[:, 1:5],
            gap8.inputs.demand_features[:, 1:5],
        )

    torch.manual_seed(808)
    model = MatchedSequenceControlLattice(
        freedom=SequenceControlFreedom.STATE_AWARE,
        num_entities=8,
        value_dim=8,
        embedding_dim=16,
        hidden_dim=32,
    ).eval()
    batch = generate_sequence_control_lattice_batch(
        family=SequenceDemandFamily.MAGNITUDE,
        batch_size=3,
        num_entities=8,
        value_dim=8,
        updates=2,
        gap_events=4,
        seed=909,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        normal = model(sequence_control_lattice_model_input(batch)).state
        activated = model(
            sequence_control_lattice_model_input(
                batch,
                activate_distractor_verified=True,
            )
        ).state
    assert float((normal - activated).abs().max()) > 1e-8


def test_registered_synthetic_aggregate_opens_every_adjacent_gate() -> None:
    config = load_config(E18B_CONFIG)
    rows = e18b._synthetic_dry_rows(config)
    assert e18b.paired_grid_contract(rows=rows, config=config)
    contrasts, paired, active = e18b.aggregate_contrasts(
        rows=rows,
        config=config,
    )
    assert len(contrasts) == 4
    assert len(paired) == 20
    assert len(active) == 100
    assert all(value["passed"] for value in contrasts.values())
    assert all(value["stress_sign_flip_p"] == 0.03125 for value in contrasts.values())


def test_e18a_cpu_dry_run_writes_active_path_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e18a",
            "--config",
            str(E18A_CONFIG),
            "--device",
            "cpu",
            "--artifact-root",
            str(tmp_path),
            "--dry-run",
        ],
    )
    e18a.main()
    latest = _read_json(tmp_path / e18a.EXPERIMENT_ID / "latest.json")
    run_dir = Path(str(latest["run_dir"]))
    report = _read_json(run_dir / "report.json")
    rows = [
        json.loads(line)
        for line in (
            run_dir / "sequence_control_lattice_metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert report["status"] == "DRY_RUN"
    assert report["rows"] == report["expected_rows"] == 8
    assert report["distractor_path_contract"]["passed"] is True
    assert len(rows) == 8
    assert sum(
        "distractor_activation_retention_harm" in row for row in rows
    ) == 4


def test_e18b_cpu_dry_run_exercises_registered_gate_logic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e18b",
            "--config",
            str(E18B_CONFIG),
            "--device",
            "cpu",
            "--artifact-root",
            str(tmp_path),
            "--dry-run",
        ],
    )
    e18b.main()
    latest = _read_json(tmp_path / e18b.EXPERIMENT_ID / "latest.json")
    report = _read_json(Path(str(latest["run_dir"])) / "report.json")
    assert report["status"] == "DRY_RUN"
    assert report["claim_gate"]["supported"] is False
    assert all(report["claim_gate"]["conditions"].values())
    assert report["summary"]["metric_rows"] == 1200
