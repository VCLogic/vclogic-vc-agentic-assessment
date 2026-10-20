import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from index_pitchshow_results import result_record


def test_common_result_preserves_phase1_only_without_inventing_prediction(tmp_path):
    p=tmp_path/'summary.json'
    p.write_text(json.dumps({'episode_slug':'ep','contract_version':'v4','phase1_status':'accepted',
                            'phase2_status':'not_run','decision':{},'usage':{'cost_usd':1.2}}))
    result=result_record(p,vc='vc',episode='ep',mode='founders-and-other-vcs',cohort='legacy-panel',actual='In')
    assert result['predicted_decision'] is None
    assert result['phase2_status']=='not_run'
    assert result['actual_pitch_decision']=='In'
    assert result['schema']=='pitchshow-result-v1'


def test_common_result_keeps_native_paths_and_prediction(tmp_path):
    p=tmp_path/'summary.json'
    p.write_text(json.dumps({'episode_slug':'ep','contract_version':'v4.1','phase1_status':'provisional',
                            'phase2_status':'accepted','decision':{'decision':'Out','investment_likelihood':0.2}}))
    result=result_record(p,vc='vc',episode='ep',mode='founders-only',cohort='current',actual='Out')
    assert result['predicted_decision']=='Out'
    assert result['investment_likelihood']==0.2
    assert result['native_artifact_root']==str(tmp_path.resolve())


def test_regression_cohort_does_not_pool_with_new_cases(tmp_path):
    from index_pitchshow_results import execution_cohort
    status = tmp_path / 'status.json'
    assert execution_cohort(status) == 'current-pitchshow'
    (tmp_path / 'experiment.json').write_text(json.dumps({'experiment':'repair-regression-v1','evaluation_role':'regression'}))
    assert execution_cohort(status) == 'repair-regression-v1'
