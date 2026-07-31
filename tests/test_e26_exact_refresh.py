from catena.lm.transactional_stream import Operation, QueryType, generate_episode


def test_exact_refresh_contains_current_state_and_not_future_query() -> None:
    episode = generate_episode(
        seed=19,
        split="main_test",
        domain="workflow",
        operation=Operation.SUPERSEDE,
        index=5,
    )
    target = episode.protected_fields["target_entity"]
    assert target in episode.exact_refresh_text
    assert str(episode.protected_fields["new_version"]) in episode.exact_refresh_text
    for query in episode.queries:
        assert query.prompt not in episode.exact_refresh_text
    assert {q.query_type for q in episode.queries} == {item.value for item in QueryType}
