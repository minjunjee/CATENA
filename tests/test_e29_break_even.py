from catena.lm.systems_boundary import (
    PolicyMeasurement,
    break_even_queries,
    quality_constrained_pareto,
)


def test_break_even_formula() -> None:
    recurrent = PolicyMeasurement("recurrent", 60.0, 0.0, 8.0, 10)
    external = PolicyMeasurement("external", 0.0, 0.0, 30.0, 20)
    result = break_even_queries(recurrent, external)
    assert result.query_count is not None
    assert abs(result.query_count - 60.0 / 22.0) < 1e-12


def test_failed_quality_policy_is_not_on_frontier() -> None:
    good = PolicyMeasurement("good", 10.0, 0.0, 2.0, 10, quality_pass=True)
    bad = PolicyMeasurement("bad", 0.0, 0.0, 1.0, 1, quality_pass=False)
    assert [
        item.policy for item in quality_constrained_pareto([good, bad], queries_per_update=4)
    ] == ["good"]
