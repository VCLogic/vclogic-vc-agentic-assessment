#!/usr/bin/env python3
"""Offline, append-only evaluation of preserved assessment results and failures."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from prepare_pitchshow_evaluation import read, write
from vc_clone_graph.phase1_evaluation import score_set

QUALIFICATION = "Automated candidate transcript references, not human-validated ground truth. Strict mismatches require review; neutral questions do not imply valenced assessments."


def provider_metadata(artifact):
    """Use the actual execution config, never today's investor template."""
    artifact = Path(artifact)
    candidates = [artifact / 'run-config.json', artifact.parent.parent / 'run-config.json',
                  artifact / 'config.json']
    for path in candidates:
        if path.exists():
            config = read(path)
            provider = config.get('provider', {})
            if isinstance(provider, dict):
                return {'backend': provider.get('kind', 'unknown'),
                        'model': provider.get('model', 'unknown'),
                        'phase1_model': config.get('phase1', {}).get('model') or provider.get('model', 'unknown'),
                        'phase2_model': config.get('phase2', {}).get('model') or provider.get('model', 'unknown')}
    return dict.fromkeys(('backend', 'model', 'phase1_model', 'phase2_model'), 'unknown')


def rationale_stage_errors(pred, ref, controlling_ids, decision_correct):
    """Exact-label diagnostics; overlapping concepts still need semantic review."""
    decisive=[r for r in ref if r.get('decision_link') in {'explicit','supports_in','supports_out'}]
    labels={r.get('taxonomy_label') or r.get('rationale_label') for r in pred}
    missing=set();direction=set();noncontrolling=set();controlling=set()
    for r in decisive:
        label=r.get('rationale_label') or r.get('taxonomy_label')
        matches=[p for p in pred if (p.get('taxonomy_label') or p.get('rationale_label'))==label]
        if label not in labels:missing.add(label);continue
        same=[p for p in matches if p['direction']==r['direction']]
        if not same:direction.add(label);continue
        if any(p.get('rationale_id') in controlling_ids for p in same):controlling.add(label)
        else:noncontrolling.add(label)
    return {'missing_decisive_labels':sorted(missing),'direction_disagreements':sorted(direction),
            'detected_not_controlling':sorted(noncontrolling),
            'detected_controlling_but_decision_wrong':sorted(controlling) if decision_correct is False else [],
            'qualification':'Exact-label review cues, not proof of causal weighting or semantic equivalence.'}


def evaluate_record(row, root):
    result = dict(row)
    result.update(issues=[], reference_quality=QUALIFICATION, label_scores=None,
                  label_direction_scores=None, decision_correct=None)
    artifact = Path(row.get('native_artifact_root', '.'))
    result.update(provider_metadata(artifact))
    # Explicit saved metadata remains authoritative when the config is unavailable.
    for key in ('backend', 'model', 'phase1_model', 'phase2_model'):
        if result[key] == 'unknown' and row.get(key):
            result[key] = row[key]
    if row.get('execution_status') in {'failed', 'running', 'pending'}:
        result['issues'].append('execution_failure' if row['execution_status'] == 'failed' else 'execution_incomplete')
        return result
    summary_path = artifact / 'summary.json'
    if summary_path.exists() and row.get('summary_sha256') and sha256(summary_path.read_bytes()).hexdigest() != row['summary_sha256']:
        result['issues'].append('artifact_integrity_failure')
        return result
    if any(row.get(k) in {'failed', 'error'} for k in ('phase1_status', 'phase2_status')):
        result['execution_status'] = 'failed'
        result['issues'].append('execution_failure')
        return result
    result['execution_status'] = 'completed'
    if 'provisional' in (row.get('phase1_status'), row.get('phase2_status')):
        result['issues'].append('unresolved_convergence')
    predicted, actual = row.get('predicted_decision'), row.get('actual_pitch_decision')
    if predicted in {'In', 'Out'} and actual in {'In', 'Out'}:
        result['decision_correct'] = predicted == actual
        if predicted != actual:
            result['issues'].append('decision_mismatch')
    else:
        result['issues'].append('decision_comparison_unavailable')
    vc, episode, mode = row['vc_slug'], row['episode_slug'], row['input_mode']
    audit = root / 'investors' / vc / 'cases' / episode / mode / 'context-quality.json'
    if audit.exists():
        result['input_context_audit'] = read(audit)
        if result['input_context_audit'].get('findings'):
            result['issues'].append('input_context_review')
    else:
        result['issues'].append('input_context_not_audited')
    refs = [root/'rationale-corpus/live-founder-v2'/vc/f'{episode}.json', root/'rationale-corpus/records'/vc/f'{episode}.json',
            root/'investors'/vc/'rationale-references'/f'{episode}.json']
    ref_path = next((p for p in refs if p.exists()), None)
    prediction_path = Path(row['phase1_path']) if row.get('phase1_path') else artifact/'phase1/investigation.json'
    if ref_path is None or not prediction_path.exists():
        result['issues'].append('rationale_comparison_unavailable')
        return result
    reference = read(ref_path)
    ref = reference.get('rationales')
    pred = read(prediction_path).get('rationales')
    if not isinstance(ref, list) or not isinstance(pred, list):
        result['issues'].append('rationale_comparison_unavailable')
        return result
    result.update(reference_path=str(ref_path), reference_sha256=sha256(ref_path.read_bytes()).hexdigest(),
                  prediction_sha256=sha256(prediction_path.read_bytes()).hexdigest(),
                  reference_kind=reference.get('reference_kind', 'unknown'),
                  reference_human_validated=reference.get('human_validated', False))
    pairs = lambda items: {(r.get('taxonomy_label') or r.get('rationale_label'), r.get('direction')) for r in items}
    pp, rr = pairs(pred), pairs(ref)
    if any(label is None or direction is None for label, direction in pp | rr):
        result['issues'].append('invalid_rationale_attributes')
        return result
    pl, rl = {p[0] for p in pp}, {r[0] for r in rr}
    result.update(label_scores=score_set(pl, rl), label_direction_scores=score_set(pp, rr),
                  matched_labels=sorted(pl & rl), missed_labels=sorted(rl-pl),
                  additional_labels=sorted(pl-rl),
                  missed_label_directions=sorted(rr-pp), additional_label_directions=sorted(pp-rr))
    decision_path=artifact/'phase2/decision.json'
    decision=read(decision_path) if decision_path.exists() else {}
    result['rationale_stage_errors']=rationale_stage_errors(pred,ref,decision.get('controlling_rationale_ids',[]),result['decision_correct'])
    result['reference_scoring_scope']=reference.get('scoring_scope','legacy_unspecified')
    result['reference_qc'] = {
        'neutral_question_labels': sorted({r.get('rationale_label') for r in ref if r.get('direction') == 'neutral' and (r.get('utterance_type') == 'question' or r.get('activation') == 'queried')}),
        'valenced_question_labels': sorted({r.get('rationale_label') for r in ref if r.get('direction') in {'positive', 'negative'} and r.get('utterance_type') == 'question'}),
        'missing_evidence_count': sum(not (r.get('quote') or r.get('evidence')) for r in ref),
        'decision_reference_disagreement': reference.get('actual_decision') in {'In', 'Out'} and reference['actual_decision'] != actual,
        'policy': 'Preserve reference labels and directions; these flags are review cues, not automatic corrections.'}
    if result['reference_qc']['decision_reference_disagreement']:
        result['issues'].append('decision_reference_conflict')
    if pp != rr:
        result['issues'].append('rationale_mismatch')
    if not reference.get('human_validated', False):
        result['issues'].append('candidate_reference_uncertainty')
    return result


def aggregate(rows):
    groups = defaultdict(list)
    keys = ('vc_slug', 'input_mode', 'cohort', 'backend', 'model', 'phase1_model', 'phase2_model', 'reference_scoring_scope')
    for row in rows:
        groups[tuple(row.get(k, 'unknown') for k in keys)].append(row)
    output = []
    for key, cases in sorted(groups.items()):
        group = dict(zip(keys, key))
        decisions = [r['decision_correct'] for r in cases if r['decision_correct'] is not None]
        group.update(case_count=len(cases), decision_count=len(decisions),
                     decision_accuracy=sum(decisions)/len(decisions) if decisions else None,
                     issues=dict(Counter(issue for r in cases for issue in r['issues'])))
        for metric in ('label_scores', 'label_direction_scores'):
            scored = [r[metric] for r in cases if r[metric] is not None]
            tp, fp, fn = (sum(r[k] for r in scored) for k in ('tp', 'fp', 'fn'))
            group[metric] = {'case_count': len(scored), 'tp': tp, 'fp': fp, 'fn': fn,
                            'precision': tp/(tp+fp) if tp+fp else None,
                            'recall': tp/(tp+fn) if tp+fn else None,
                            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
        output.append(group)
    return output


def build_report(root=Path('evaluation/pitchshow'), batch=Path('outputs/pitchshow-evaluation/batch'), rows=None):
    root = Path(root)
    if rows is None:
        catalog = root/'results/catalog.json'
        rows = read(catalog)['results'] if catalog.exists() else []
    rows = list(rows)
    for path in sorted(Path(batch).glob('jobs/*/status.json')):
        from index_pitchshow_results import execution_cohort
        state = read(path)
        cohort = execution_cohort(path)
        existing = next((r for r in rows if r.get('job_id') == state['job_id']), None)
        if state['status'] == 'completed' and existing is None:
            from index_pitchshow_results import result_record
            annotations = root/'investors'/state['vc_slug']/'annotations.json'
            actuals = {r['episode_slug']: r.get('pitch_window_decision') for r in read(annotations)} if annotations.exists() else {}
            record = result_record(Path(state['summary_path']), vc=state['vc_slug'], episode=state['episode_slug'],
                                   mode=state['mode'], cohort=cohort, actual=actuals.get(state['episode_slug']))
            record.update(job_id=state['job_id'], selected_attempt=state['selected_attempt'],
                          summary_sha256=state['summary_sha256'])
            rows.append(record)
        if state['status'] != 'completed':
            if existing is not None:
                rows.remove(existing)
            rows.append({'vc_slug': state['vc_slug'], 'episode_slug': state['episode_slug'],
                         'input_mode': state['mode'], 'cohort': cohort,
                         'job_id': state['job_id'], 'execution_status': state['status'],
                         'backend': state.get('backend', 'unknown'), 'error': state.get('error'),
                         'attempts': state.get('attempts', [])})
    evaluated = [evaluate_record(row, root) for row in rows]
    report = {'schema': 'pitchshow-assessment-evaluation-v1', 'reference_quality': QUALIFICATION,
              'grouping_note': 'Cohorts, backends and model combinations remain separate. Repeated jobs are separate executions, not independent benchmark samples.',
              'results': evaluated, 'groups': aggregate(evaluated)}
    vcs = sorted({p.name for p in (root/'investors').glob('*') if p.is_dir()} | {r['vc_slug'] for r in evaluated})
    report['vc_coverage'] = [{'vc_slug': vc, 'execution_records': sum(r['vc_slug'] == vc for r in evaluated),
                              'completed_records': sum(r['vc_slug'] == vc and r.get('execution_status') == 'completed' for r in evaluated)}
                             for vc in vcs] if (root/'investors').exists() else []
    identifier = sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16]
    destination = root/'results/automated-evaluation'/f'{identifier}.json'
    write(destination, report)
    write(root/'results/automated-evaluation/latest.json', {'report_path': str(destination), 'case_count': len(evaluated)})
    return destination, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('evaluation/pitchshow'))
    parser.add_argument('--batch', type=Path, default=Path('outputs/pitchshow-evaluation/batch'))
    args = parser.parse_args()
    destination, report = build_report(args.root, args.batch)
    print(json.dumps({'report': str(destination), 'cases': len(report['results']), 'groups': len(report['groups'])}))


if __name__ == '__main__':
    main()
