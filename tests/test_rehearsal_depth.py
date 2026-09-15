import pytest

from vc_clone_graph.rehearsal_depth import effective_question_limit


def test_depth_limits_are_bounded_by_server_ceiling() -> None:
    assert effective_question_limit("quick", 8) == 3
    assert effective_question_limit("standard", 8) == 5
    assert effective_question_limit("deep", 8) == 8
    assert effective_question_limit("deep", 6) == 6


@pytest.mark.parametrize("ceiling", [0, 9])
def test_server_ceiling_must_remain_within_runtime_bounds(ceiling: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 8"):
        effective_question_limit("standard", ceiling)
