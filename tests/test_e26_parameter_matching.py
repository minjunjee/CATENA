from catena.lm import ModelConfig, build_paired_models
from catena.lm.model import assert_matched_models


def test_paired_models_have_identical_registered_surface() -> None:
    tied, dual = build_paired_models(ModelConfig.tiny_reference(), seed=123)
    assert_matched_models(tied, dual)
    assert tied.parameter_count() == dual.parameter_count()
    assert tied.parameter_signature() == dual.parameter_signature()
