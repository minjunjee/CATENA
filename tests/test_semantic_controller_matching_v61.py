import torch

from catena.models.semantic_controllers_v61 import (
    MatchedSemanticControllerV61,
    SemanticRoute,
    assert_matched_semantic_pair,
)


def test_semantic_pair_has_identical_parameters_initialization_and_compute():
    torch.manual_seed(91)
    factorized = MatchedSemanticControllerV61(112, 64, SemanticRoute.FACTORIZED)
    shared = MatchedSemanticControllerV61(112, 64, SemanticRoute.SHARED)
    shared.load_state_dict(factorized.state_dict())
    assert_matched_semantic_pair(factorized, shared)
    assert torch.equal(factorized.routing_matrix, torch.eye(2))
    assert torch.equal(shared.routing_matrix, torch.full((2, 2), 0.5))


def test_shared_route_is_rank_one_and_factorized_route_is_rank_two():
    factorized = MatchedSemanticControllerV61(7, 5, SemanticRoute.FACTORIZED)
    shared = MatchedSemanticControllerV61(7, 5, SemanticRoute.SHARED)
    assert torch.linalg.matrix_rank(factorized.routing_matrix).item() == 2
    assert torch.linalg.matrix_rank(shared.routing_matrix).item() == 1


def test_controller_forward_schema_and_finite_guard():
    model = MatchedSemanticControllerV61(7, 5, SemanticRoute.FACTORIZED)
    gates = model(torch.zeros(3, 7))
    assert gates.erase.shape == (3,)
    assert gates.write.shape == (3,)
    assert torch.all((gates.erase > 0.0) & (gates.erase < 1.0))
    assert torch.all((gates.write > 0.0) & (gates.write < 1.0))

    bad = torch.zeros(3, 7)
    bad[0, 0] = torch.nan
    try:
        model(bad)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("non-finite model input was accepted")
