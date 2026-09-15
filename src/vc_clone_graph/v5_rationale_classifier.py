"""Continuous v5 association features for rationale decision models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .schemas_v5 import InvestigationV5


def association_feature_map(
    investigation: InvestigationV5, *, families: Mapping[str, str]
) -> dict[str, float]:
    rows = investigation.episode_level_associations
    result: dict[str, float] = {
        "assoc__distinct_label_count": float(len({row.taxonomy_label for row in rows})),
        "assoc__rule_count": float(len(rows)),
        "assoc__claim_count": float(
            sum(bool(row.associated_rationales) for row in investigation.rationales)
        ),
        "assoc__statement_count": float(
            sum(
                bool(row.associated_rationales)
                for row in investigation.material_statement_coverage
            )
        ),
    }
    family_probability: defaultdict[str, float] = defaultdict(float)
    by_label: defaultdict[str, list] = defaultdict(list)
    for row in rows:
        by_label[row.taxonomy_label].append(row)
    for label, label_rows in by_label.items():
        result[f"assoc__{label}__max_probability"] = max(
            row.posterior_probability for row in label_rows
        )
        result[f"assoc__{label}__max_lift"] = max(row.lift for row in label_rows)
        result[f"assoc__{label}__max_support"] = float(
            max(row.support for row in label_rows)
        )
        result[f"assoc__{label}__rule_count"] = float(len(label_rows))
        family_probability[families[label]] += result[
            f"assoc__{label}__max_probability"
        ]
    for family, value in family_probability.items():
        result[f"assoc__{family}__probability_sum"] = value
    return result
