from catena.lm.transactional_stream import Operation, audit_split_disjointness, generate_grid


def test_split_namespaces_are_disjoint() -> None:
    episodes = list(
        generate_grid(
            seed=17,
            splits=["train", "validation", "calibration", "main_test", "heldout_domain"],
            domains=["access_control", "api_configuration"],
            operations=list(Operation),
            items_per_cell=2,
        )
    )
    audit = audit_split_disjointness(episodes)
    assert audit["disjoint"] is True
    assert audit["duplicates"] == []
    assert audit["validation_errors"] == {}
