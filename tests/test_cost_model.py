from catena.systems.cost_model import CostModel


def test_break_even_exists() -> None:
    model = CostModel(
        assimilation_update_ms=5.0,
        assimilation_query_ms=0.5,
        external_retrieve_ms=2.0,
        external_query_ms=0.5,
        full_refresh_ms=20.0,
    )
    assert model.break_even_queries() == 3
