from __future__ import annotations

from vc_clone_graph.schemas_v5 import InvestigationV5
from vc_clone_graph.v5_rationale_classifier import association_feature_map
from tests.test_schemas_v5 import payload


def test_association_features_are_continuous_and_do_not_activate_labels() -> None:
    raw = payload()
    raw["episode_level_associations"] = raw["rationales"][0]["associated_rationales"]
    investigation = InvestigationV5.model_validate(raw)
    features = association_feature_map(
        investigation,
        families={"market_size_assessment": "market_opportunity"},
    )
    assert features["assoc__market_size_assessment__max_probability"] == 0.75
    assert features["assoc__market_size_assessment__max_lift"] == 2.0
    assert features["assoc__market_size_assessment__max_support"] == 4.0
    assert features["assoc__market_opportunity__probability_sum"] == 0.75
    assert features["assoc__distinct_label_count"] == 1.0
    assert not any(name.endswith("__present") for name in features)
