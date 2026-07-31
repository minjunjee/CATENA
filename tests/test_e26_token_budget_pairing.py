from catena.lm import ModelConfig, build_paired_models
from catena.lm.tokenizer import ByteTokenizer
from catena.lm.trainer import (
    compare_optimizer_signatures,
    cycle_tensor_batches,
    train_reference_steps,
)


def test_paired_reference_steps_use_same_tokens_and_optimizer_shapes() -> None:
    tokenizer = ByteTokenizer()
    tied, dual = build_paired_models(ModelConfig.tiny_reference(), seed=33)
    texts = ["alpha update", "beta retention", "gamma stale"]
    sequences = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        sequences.append((ids * 4)[:24])
    tied_metrics, tied_opt = train_reference_steps(
        tied, cycle_tensor_batches(sequences, batch_size=2, device="cpu"), steps=2
    )
    dual_metrics, dual_opt = train_reference_steps(
        dual, cycle_tensor_batches(sequences, batch_size=2, device="cpu"), steps=2
    )
    assert tied_metrics[-1].tokens_seen == dual_metrics[-1].tokens_seen
    assert compare_optimizer_signatures(tied_opt, dual_opt).matched
