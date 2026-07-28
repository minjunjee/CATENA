from catena.core.schema import CandidateMode
from catena.data.geometry_sweep import generate_geometry_grid
from catena.training.schedules_v61 import (
    balanced_geometry_schedule,
    schedule_cell_counts,
    schedule_sha256,
)


def _episodes():
    return generate_geometry_grid(
        seed=100,
        candidate_mode=CandidateMode.ORACLE,
        grid={
            "key_dim": [8],
            "value_dim": [8],
            "num_associations": [4, 6],
            "key_correlations": [0.0, 0.2],
            "old_scales": [0.8, 1.2],
            "new_scales": [1.0],
            "old_new_cosines": [0.0],
        },
        count_per_cell=3,
    )


def test_balanced_schedule_covers_every_geometry_operation_cell():
    episodes = _episodes()
    cells = schedule_cell_counts(episodes)
    schedule = balanced_geometry_schedule(
        episodes,
        steps=len(cells) + 3,
        seed=17,
    )
    scheduled = schedule_cell_counts(schedule)
    assert set(scheduled) == set(cells)
    assert max(scheduled.values()) - min(scheduled.values()) <= 1


def test_balanced_schedule_is_reproducible_for_matched_conditions():
    episodes = _episodes()
    first = balanced_geometry_schedule(episodes, steps=40, seed=19)
    second = balanced_geometry_schedule(episodes, steps=40, seed=19)
    assert schedule_sha256(first) == schedule_sha256(second)
