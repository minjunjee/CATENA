from __future__ import annotations

import torch

from catena.lm.config import ModelConfig
from catena.lm.evaluator import evaluate_episode_branched
from catena.lm.model import CatenaLM
from catena.lm.tokenizer import ByteTokenizer
from catena.lm.transactional_stream import Operation, generate_episode


def test_query_permutation_does_not_change_branched_scores() -> None:
    torch.manual_seed(26026)
    config = ModelConfig(
        vocab_size=259,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ffn_multiplier=2.0,
        recurrent_layers=(0,),
        local_attention_layers=(1,),
        local_attention_window=16,
        context_length=512,
    )
    model = CatenaLM(config).eval()
    tokenizer = ByteTokenizer()
    episode = generate_episode(
        seed=17,
        split="validation",
        domain="access_control",
        operation=Operation.SUPERSEDE,
        index=1,
        distractor_units=0,
    )
    natural = evaluate_episode_branched(model, tokenizer, episode)
    reversed_order = evaluate_episode_branched(
        model,
        tokenizer,
        episode,
        query_order=tuple(reversed(range(4))),
    )
    assert [row.predicted_index for row in natural] == [
        row.predicted_index for row in reversed_order
    ]
    assert [[score.log_probability for score in row.candidate_scores] for row in natural] == [
        [score.log_probability for score in row.candidate_scores] for row in reversed_order
    ]
