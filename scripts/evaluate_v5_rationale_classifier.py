#!/usr/bin/env python3
"""Compare the primary rationale classifier with v5 association features."""

from __future__ import annotations

import argparse, csv, json
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

from vc_clone_graph.actual_rationale_cases import build_rationale_model_cases
from vc_clone_graph.actual_rationale_evaluation import align_rationale_cases
from vc_clone_graph.actual_rationale_models import nested_per_vc_predictions
from vc_clone_graph.phase1_evaluation import load_phase1_cases
from vc_clone_graph.phase2_calibration_evaluation import (
    MethodPrediction,
    classification_metric_rows,
    load_phase2_records,
    ranking_metric_rows,
)
from vc_clone_graph.schemas_v5 import InvestigationV5
from vc_clone_graph.v5_rationale_classifier import association_feature_map


def method_rows(predictions, name: str):
    return [MethodPrediction(
        vc_slug=row.vc_slug, vc_name=row.vc_name, episode_slug=row.episode_slug,
        group=row.held_group, target=row.target, method=name, score=row.score,
        predicted=row.balanced_decision,
    ) for row in predictions]


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-registry", type=Path, default=Path("evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json"))
    p.add_argument("--references", type=Path, default=Path("evaluation/phase1_ground_truth_rationales"))
    p.add_argument("--taxonomy", type=Path, default=Path("../agentic-vc-clone-framework/taxonomy/codebook_v_final.json"))
    p.add_argument("--v5-root", type=Path, default=Path("outputs/v5-associated-rationales/phase1"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-jobs", type=int, default=6)
    args=p.parse_args(argv)
    phase1,taxonomy=load_phase1_cases(Path.cwd(),args.source_registry,args.references,args.taxonomy)
    records=load_phase2_records(args.source_registry)
    base_cases=align_rationale_cases(
        records, build_rationale_model_cases(phase1,taxonomy)
    )
    enriched=[]
    for case in base_cases:
        path=args.v5_root/"investors"/case.vc_slug/case.episode_slug/"phase1/investigation.json"
        v5=InvestigationV5.model_validate_json(path.read_text())
        assoc=association_feature_map(v5,families={k:v.coarse_parent for k,v in taxonomy.items()})
        enriched.append(replace(
            case,
            actual_features={**case.actual_features,**assoc},
            predicted_features={**case.predicted_features,**assoc},
        ))
    # run_all_rationale_models advances 30,000 before the actual-to-predicted
    # per-VC logistic endpoint. Preserve that exact established baseline seed.
    model_seed = 20260817 + 30_000
    baseline=nested_per_vc_predictions(base_cases,model_family="logistic",source_condition="actual_to_predicted",seed=model_seed,n_jobs=args.n_jobs)
    challenger=nested_per_vc_predictions(enriched,model_family="logistic",source_condition="actual_to_predicted",seed=model_seed,n_jobs=args.n_jobs)
    methods={"baseline_actual_to_predicted":method_rows(baseline,"baseline_actual_to_predicted"),"v5_association_enhanced":method_rows(challenger,"v5_association_enhanced")}
    classification=[row for rows in methods.values() for row in classification_metric_rows(rows)]
    ranking=[row for rows in methods.values() for row in ranking_metric_rows(rows)]
    b=methods["baseline_actual_to_predicted"]; c=methods["v5_association_enhanced"]
    b_only=sum(x.predicted==x.target and y.predicted!=y.target for x,y in zip(b,c,strict=True))
    c_only=sum(y.predicted==y.target and x.predicted!=x.target for x,y in zip(b,c,strict=True))
    pvalue=binomtest(min(b_only,c_only),b_only+c_only,0.5).pvalue if b_only+c_only else 1.0
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/"classification_metrics.csv",classification); write_csv(args.output/"ranking_metrics.csv",ranking)
    pred_rows=[{"vc_slug":x.vc_slug,"episode_slug":x.episode_slug,"target":x.target,"baseline_score":x.score,"baseline_decision":x.predicted,"v5_score":y.score,"v5_decision":y.predicted} for x,y in zip(b,c,strict=True)]
    write_csv(args.output/"predictions.csv",pred_rows)
    macro={r["method"]:r for r in classification if r["scope"]=="macro"}
    rank5={r["method"]:r for r in ranking if r["scope"]=="macro" and r["review_budget"]==5}
    summary={"schema":"v5-rationale-classifier-evaluation-v1","case_count":len(b),"mcnemar_exact_p":pvalue,"baseline_only_correct":b_only,"v5_only_correct":c_only,"classification_macro":macro,"ranking_macro_at_5":rank5}
    (args.output/"metrics.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (args.output/"evaluation.md").write_text(
        "# V5 Rationale Classifier\n\n"
        f"Population: **{len(b)}** cases across six VCs.\n\n"
        "| Method | Balanced accuracy | In precision | In recall | In F1 | AP | ROC AUC |\n|---|---:|---:|---:|---:|---:|---:|\n"+
        "\n".join(f"| {name} | {macro[name]['balanced_accuracy']:.3f} | {macro[name]['in_precision']:.3f} | {macro[name]['in_recall']:.3f} | {macro[name]['in_f1']:.3f} | {rank5[name]['average_precision']:.3f} | {rank5[name]['roc_auc']:.3f} |" for name in methods)+
        f"\n\nExact paired McNemar p={pvalue:.4f}; v5-only correct={c_only}, baseline-only correct={b_only}.\n",
        encoding="utf-8")
    print(json.dumps(summary,indent=2,default=str))
    return 0

if __name__=="__main__": raise SystemExit(main())
