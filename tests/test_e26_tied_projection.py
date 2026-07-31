import copy

import torch

from catena.lm import ModelConfig, build_paired_models


def _equalize_gate_halves(model) -> None:
    with torch.no_grad():
        for block in model.blocks:
            if not block.is_recurrent:
                continue
            head = block.mixer.gate_head
            half = head.weight.shape[0] // 2
            head.weight[half:].copy_(head.weight[:half])
            head.bias[half:].copy_(head.bias[:half])


def test_projection_matches_dual_when_raw_logits_are_equal() -> None:
    tied, dual = build_paired_models(ModelConfig.tiny_reference(), seed=4)
    tied = copy.deepcopy(tied).eval()
    dual = copy.deepcopy(dual).eval()
    _equalize_gate_halves(tied)
    _equalize_gate_halves(dual)
    inputs = torch.randint(0, tied.config.vocab_size, (2, 21))
    with torch.no_grad():
        tied_output = tied(inputs)
        dual_output = dual(inputs)
    torch.testing.assert_close(tied_output.logits, dual_output.logits, rtol=1e-6, atol=1e-6)
    for left, right in zip(
        tied_output.runtime_state.recurrent,
        dual_output.runtime_state.recurrent,
        strict=True,
    ):
        torch.testing.assert_close(left.matrix, right.matrix, rtol=1e-6, atol=1e-6)
