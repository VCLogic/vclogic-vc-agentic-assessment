from vc_clone_graph.panel_question_canary import CanaryCandidate, select_panel_question_canaries


def test_selects_two_per_label_per_vc_deterministically() -> None:
    rows = []
    for vc in ("alpha", "beta"):
        for label in ("In", "Out"):
            for index, count in enumerate((2, 8, 5)):
                rows.append(CanaryCandidate(
                    vc_slug=vc, registry_vc_slug=vc, episode_slug=f"{index}-{vc}-{label.lower()}",
                    actual_decision=label, insertion_count=count,
                    pitch_path=f"/{index}/pitch.txt", audit_path=f"/{index}/audit.json",
                    pitch_sha256=str(index) * 64, canonical_artifact_root=f"/{index}/canonical",
                ))
    selected = select_panel_question_canaries(rows)
    assert len(selected) == 8
    for vc in ("alpha", "beta"):
        for label in ("In", "Out"):
            subset = [row for row in selected if row.vc_slug == vc and row.actual_decision == label]
            assert [row.insertion_count for row in subset] == [8, 5]


def test_rejects_stratum_with_fewer_than_two_eligible_cases() -> None:
    row = CanaryCandidate(
        vc_slug="alpha", registry_vc_slug="alpha", episode_slug="one",
        actual_decision="In", insertion_count=1, pitch_path="p", audit_path="a",
        pitch_sha256="a" * 64, canonical_artifact_root="c",
    )
    try:
        select_panel_question_canaries([row])
    except ValueError as exc:
        assert "alpha/In" in str(exc)
    else:
        raise AssertionError("incomplete stratum must fail")
