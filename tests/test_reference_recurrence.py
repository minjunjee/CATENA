import torch

from catena.models.reference_recurrence import (
    gdn2_reference_update,
    kda_tied_reference_update,
)


def test_gdn2_tied_equals_kda_reference() -> None:
    generator = torch.Generator().manual_seed(5)
    dim = 8
    state = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    key = torch.randn(dim, generator=generator, dtype=torch.float64)
    value = torch.randn(dim, generator=generator, dtype=torch.float64)
    decay = torch.sigmoid(torch.randn(dim, generator=generator, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn((), generator=generator, dtype=torch.float64))
    full = gdn2_reference_update(
        state,
        key,
        value,
        decay,
        beta.expand_as(key),
        beta.expand_as(value),
    )
    tied = kda_tied_reference_update(state, key, value, decay, beta)
    assert torch.equal(full, tied)
