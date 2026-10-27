import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_pitchshow_batch as batch


def test_alias_mapping_evidence_is_bound_to_source(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'slug': 'one', 'transcript': 'Founder: We are Acme AI.'}))
    mapping = {'episodes': {'one': {'aliases': ['Acme AI'], 'source_path': 'source.json',
        'source_sha256': batch.digest(source), 'evidence': [{'alias': 'Acme AI',
        'field': 'transcript', 'quote': 'We are Acme AI.', 'matched_name': 'Acme AI'}]}}}
    assert batch.validate_aliases(tmp_path, mapping, {'one'}) == {'one': ['Acme AI']}
    mapping['episodes']['one']['evidence'][0]['quote'] = 'A different company'
    with pytest.raises(ValueError): batch.validate_aliases(tmp_path, mapping, {'one'})
    with pytest.raises(ValueError): batch.validate_aliases(tmp_path, mapping, {'missing'})


def config(tmp_path, output):
    c = SimpleNamespace(run=SimpleNamespace(episode_slug='one', output_root=str(output)))
    c.resolve_path = lambda p: Path(p)
    return c


def test_resume_skips_only_verified_completed_results(tmp_path, monkeypatch):
    calls = []
    def prepare(job, attempt):
        return config(tmp_path, tmp_path / f'attempt-{attempt}')
    def run(c):
        calls.append(c)
        out = Path(c.run.output_root) / 'one';out.mkdir(parents=True)
        (out/'summary.json').write_text(json.dumps({'decision': {'decision': 'In'}}))
    monkeypatch.setattr(batch, 'command_run', run)
    monkeypatch.setattr(batch, 'command_verify', lambda c: None)
    job={'job_id':'abc', 'vc_slug':'vc', 'episode_slug':'one','mode':'founders-only'}
    state=tmp_path/'state.json'
    first=batch.run_job(job, state, prepare)
    assert first['status']=='completed' and len(calls)==1
    assert batch.run_job(job,state,prepare)['status']=='completed' and len(calls)==1
    def corrupt(c): raise ValueError('corrupt artifact')
    monkeypatch.setattr(batch,'command_verify',corrupt)
    assert batch.run_job(job,state,prepare)['status']=='failed'
    assert len(calls)==1


def test_failure_retry_uses_fresh_attempt_and_preserves_history(tmp_path, monkeypatch):
    numbers=[]
    def prepare(job,attempt):
        numbers.append(attempt)
        return config(tmp_path,tmp_path/f'attempt-{attempt}')
    def fail(c): raise RuntimeError('provider unavailable')
    monkeypatch.setattr(batch,'command_run',fail)
    monkeypatch.setattr(batch,'command_verify',lambda c: None)
    job={'job_id':'abc','vc_slug':'vc','episode_slug':'one','mode':'founders-only'}
    path=tmp_path/'state.json'
    assert batch.run_job(job,path,prepare)['status']=='failed'
    assert batch.run_job(job,path,prepare)['status']=='failed'
    assert numbers==[1,2]
    assert len(json.loads(path.read_text())['attempts'])==2


def test_dry_run_never_packages_or_executes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(batch,'plan',lambda *a: [{'job_id':'a','mode':'founders-only'}])
    def forbidden(*a,**k): raise AssertionError('dry run attempted execution')
    monkeypatch.setattr(batch,'prepare_attempt',forbidden)
    monkeypatch.setattr(batch,'command_run',forbidden)
    assert batch.main([])==0


def test_no_decision_is_a_failure_even_if_engine_returns(tmp_path,monkeypatch):
    c=config(tmp_path,tmp_path/'attempt-1')
    out=Path(c.run.output_root)/'one';out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps({'decision': None}))
    monkeypatch.setattr(batch,'command_run',lambda c: None)
    monkeypatch.setattr(batch,'command_verify',lambda c: None)
    result=batch.run_job({'job_id':'a'},tmp_path/'status.json',lambda j,n:c)
    assert result['status']=='failed'


def test_asset_changes_fail_before_packaging(tmp_path,monkeypatch):
    monkeypatch.setattr(batch,'asset_signature',lambda *a:'new')
    monkeypatch.setattr(batch,'package_case',lambda *a,**k:pytest.fail('packaged stale job'))
    with pytest.raises(ValueError,match='changed since planning'):
        batch.prepare_attempt(tmp_path,tmp_path,{'vc_slug':'vc','asset_signature':'old'},1)


def test_codex_override_changes_job_identity_before_execution(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(batch,'plan',lambda *a:[{'job_id':'original','mode':'founders-only'}])
    assert batch.main(['--backend','codex'])==0
    plan=json.loads(next((tmp_path/'outputs/pitchshow-evaluation/batch').glob('plan-*.json')).read_text())
    job=plan['jobs'][0]
    assert job['backend']=='codex' and job['job_id']!='original'


def test_components_failure_never_fails_a_completed_job(tmp_path, monkeypatch):
    def run(c):
        out = Path(c.run.output_root) / 'one';out.mkdir(parents=True)
        (out/'summary.json').write_text(json.dumps({'decision': {'decision': 'Out'}}))
    def broken(c): raise RuntimeError('no key')
    monkeypatch.setattr(batch, 'command_run', run)
    monkeypatch.setattr(batch, 'command_verify', lambda c: None)
    monkeypatch.setattr(batch, 'run_default_components', broken)
    job={'job_id':'abc', 'vc_slug':'vc', 'episode_slug':'one','mode':'founders-only'}
    result=batch.run_job(job, tmp_path/'state.json', lambda j,n: config(tmp_path, tmp_path/'out'))
    assert result['status']=='completed'
