import torch

from catena.models.interventions import Intervention, InterventionKind, apply_intervention
from catena.models.memory import GateOutput


def test_graded_clamps() -> None:
    gates = GateOutput(erase=torch.tensor(0.8), write=torch.tensor(0.6))
    erased = apply_intervention(
        gates, Intervention(InterventionKind.ERASE_CLAMP, dose=0.25)
    )
    written = apply_intervention(
        gates, Intervention(InterventionKind.WRITE_CLAMP, dose=0.5)
    )
    assert torch.isclose(erased.erase, torch.tensor(0.2))
    assert torch.isclose(erased.write, torch.tensor(0.6))
    assert torch.isclose(written.erase, torch.tensor(0.8))
    assert torch.isclose(written.write, torch.tensor(0.3))
