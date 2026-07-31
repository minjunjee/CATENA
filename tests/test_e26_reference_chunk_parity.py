import pytest
import torch

from catena.lm import ModelConfig
from catena.lm.model import CatenaLM


@pytest.mark.parametrize("variant", ["dual_delta_lm", "projected_tied_delta_lm"])
@pytest.mark.parametrize("sequence", [1, 7, 8, 9, 31])
def test_reference_full_chunk_and_state_carry_parity(variant: str, sequence: int) -> None:
    torch.manual_seed(9)
    config = ModelConfig.tiny_reference(variant)
    model = CatenaLM(config).eval()
    inputs = torch.randint(0, config.vocab_size, (2, sequence))
    with torch.no_grad():
        full = model(inputs, chunked_reference=False)
        chunked = model(inputs, chunked_reference=True)
    torch.testing.assert_close(full.logits, chunked.logits, rtol=1e-5, atol=1e-6)
    for left, right in zip(
        full.runtime_state.recurrent,
        chunked.runtime_state.recurrent,
        strict=True,
    ):
        torch.testing.assert_close(left.matrix, right.matrix, rtol=1e-5, atol=1e-6)


def test_runtime_clone_has_no_storage_alias() -> None:
    model = CatenaLM(ModelConfig.tiny_reference())
    inputs = torch.randint(0, model.config.vocab_size, (1, 5))
    state = model(inputs).runtime_state
    clone = state.clone()
    assert not (set(state.storage_ptrs()) & set(clone.storage_ptrs()))
