import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_pitchshow_results import evaluate_record, aggregate, build_report


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path):
    artifact = tmp_path/'attempt-1/runs/ep'
    prediction = save(artifact/'phase1/investigation.json', {'rationales': [
        {'taxonomy_label': 'market_timing', 'direction': 'positive'},
        {'taxonomy_label': 'market_timing', 'direction': 'positive'}]})
    ref = save(tmp_path/'rationale-corpus/records/vc/ep.json', {'human_validated': False, 'rationales': [
        {'rationale_label': 'market_timing', 'direction': 'neutral', 'utterance_type': 'question', 'quote': 'Why now?'}]})
    save(tmp_path/'attempt-1/run-config.json', {'provider': {'kind': 'openrouter', 'model': 'model-a'}})
    return {'vc_slug': 'vc', 'episode_slug': 'ep', 'input_mode': 'founders-only',
            'cohort': 'current', 'native_artifact_root': str(artifact), 'phase1_path': str(prediction),
            'predicted_decision': 'In', 'actual_pitch_decision': 'Out', 'phase1_status': 'provisional'}, ref


def test_strict_metrics_neutral_question_and_immutable_reference(tmp_path):
    row, ref = fixture(tmp_path)
    before = ref.read_bytes()
    result = evaluate_record(row, tmp_path)
    assert result['label_scores']['f1'] == 1
    assert result['label_direction_scores']['f1'] == 0
    assert result['label_scores']['tp'] == 1
    assert result['decision_correct'] is False
    assert {'unresolved_convergence', 'decision_mismatch', 'rationale_mismatch'} <= set(result['issues'])
    assert result['reference_qc']['neutral_question_labels'] == ['market_timing']
    assert ref.read_bytes() == before


def test_failures_never_scored_even_with_stale_prediction(tmp_path):
    row, _ = fixture(tmp_path)
    row['execution_status'] = 'failed'
    result = evaluate_record(row, tmp_path)
    assert result['label_scores'] is None and result['decision_correct'] is None
    assert result['issues'] == ['execution_failure']


def test_group_separation_and_legacy_without_provider(tmp_path):
    row, _ = fixture(tmp_path)
    a = evaluate_record(row, tmp_path)
    b = dict(a, backend='codex')
    c = dict(a, model='model-b')
    d = dict(a, input_mode='founders-context-restored')
    e = evaluate_record(dict(row, native_artifact_root=str(tmp_path/'legacy'),
                             predicted_decision=None, cohort='legacy'), tmp_path)
    assert e['backend'] == 'unknown' and e['decision_correct'] is None
    assert len(aggregate([a,b,c,d,e])) == 5


def test_reports_preserve_history_and_failed_jobs(tmp_path):
    row, _ = fixture(tmp_path)
    batch = tmp_path/'batch'
    save(batch/'jobs/failed/status.json', {'vc_slug':'vc', 'episode_slug':'ep', 'mode':'founders-only',
                                         'job_id':'failed', 'status':'failed', 'error':'timeout'})
    first, report = build_report(tmp_path, batch, [row])
    before = first.read_bytes()
    assert len(report['results']) == 2
    second, _ = build_report(tmp_path, batch, [dict(row, predicted_decision='Out')])
    assert first != second and first.read_bytes() == before
    assert sum(g['decision_count'] for g in report['groups']) == 1


def test_failed_status_supersedes_old_catalog_success(tmp_path):
    row, _ = fixture(tmp_path)
    row['job_id'] = 'same'
    batch = tmp_path/'batch'
    save(batch/'jobs/same/status.json', {'vc_slug':'vc', 'episode_slug':'ep', 'mode':'founders-only',
                                       'job_id':'same', 'status':'failed'})
    _, report = build_report(tmp_path, batch, [row])
    assert len(report['results']) == 1
    assert report['results'][0]['decision_correct'] is None


def test_failed_phase_is_not_scored_as_legacy_success(tmp_path):
    row, _ = fixture(tmp_path)
    row['phase1_status'] = 'failed'
    result = evaluate_record(row, tmp_path)
    assert result['label_scores'] is None
    assert result['execution_status'] == 'failed'


def test_uncataloged_validation_failure_keeps_its_cohort(tmp_path):
    batch=tmp_path/'batch'
    save(batch/'jobs/f/status.json',{'vc_slug':'vc','episode_slug':'ep','mode':'founders-only','job_id':'f','status':'failed'})
    save(batch/'jobs/f/experiment.json',{'experiment':'untouched-validation','evaluation_role':'validation'})
    _,report=build_report(tmp_path,batch,[])
    assert report['results'][0]['cohort']=='untouched-validation'
