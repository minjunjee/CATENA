from catena.core.schema import DemandOrientation
from catena.data.orientation import build_subspace_demand
from catena.theory.subspace_geometry import orientation_regrets


def test_axis_aligned_projector_is_diagonal_representable() -> None:
    demand = build_subspace_demand(
        dim=16,
        active_dim=4,
        orientation=DemandOrientation.AXIS_CONTIGUOUS,
        seed=1,
    )
    result = orientation_regrets(demand.projector)
    assert result.diagonal_regret < 1e-7


def test_rotated_projector_is_not_diagonal_in_general() -> None:
    demand = build_subspace_demand(
        dim=16,
        active_dim=4,
        orientation=DemandOrientation.ROTATED,
        seed=2,
    )
    result = orientation_regrets(demand.projector)
    assert result.diagonal_regret > 1e-3
    assert result.matrix_regret == 0.0
