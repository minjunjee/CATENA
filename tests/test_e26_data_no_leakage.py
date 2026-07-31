import pytest

from catena.lm.transactional_stream import Operation, generate_episode, validate_episode


@pytest.mark.parametrize("operation", list(Operation))
def test_generated_episode_has_no_visible_control_or_answer_leakage(operation: Operation) -> None:
    episode = generate_episode(
        seed=11,
        split="main_test",
        domain="access_control",
        operation=operation,
        index=3,
        distractor_units=2,
    )
    assert validate_episode(episode) == []
    assert not any(episode.visible_input_audit.values())
    for query in episode.queries:
        assert query.prompt not in episode.branch_prefix_text
