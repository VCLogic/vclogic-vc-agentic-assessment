import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_pitchshow_rationales import Rationale, Extraction, validate_reference
from vc_clone_graph.precedent_builder import parse_speaker_turns


def rationale(**kwargs):
    values=dict(rationale_label='competitive_advantage_assessment',direction='negative',salience='primary',
        confidence=0.9,utterance_type='decision_explanation',decision_link='supports_out',
        timing='original_pitch',turn_index=1,quote='There is no moat.',explanation='The investor rejects defensibility.')
    values.update(kwargs)
    return Rationale(**values)


def test_exact_target_turns_and_separate_later_rationales():
    turns=parse_speaker_turns('Founder: Hello.\nMac: There is no moat.\nPhil: Too expensive.\nMac: I changed my mind.')
    extraction=Extraction(rationales=[rationale(),rationale(timing='later_commentary',turn_index=3,quote='I changed my mind.')],
        reviewed_target_turns=[1,3],notes='Reviewed all target turns.')
    result=validate_reference(extraction,turns,['Mac'],{'competitive_advantage_assessment'})
    assert len(result['rationales'])==1 and len(result['later_rationales'])==1
    assert result['rationales'][0]['evidence_turns'][0]['turn_index']==1
    assert result['rationales'][0]['evidence']==['There is no moat.']


@pytest.mark.parametrize('change',[{'turn_index':2,'quote':'Too expensive.'},{'quote':'No moat whatsoever'}, {'rationale_label':'invented'}])
def test_wrong_speaker_quote_or_taxonomy_fails(change):
    turns=parse_speaker_turns('Founder: Hello.\nMac: There is no moat.\nPhil: Too expensive.')
    extraction=Extraction(rationales=[rationale(**change)],reviewed_target_turns=[1],notes='')
    with pytest.raises(ValueError):validate_reference(extraction,turns,['Mac'],{'competitive_advantage_assessment'})


def test_missing_turn_coverage_fails_and_empty_is_not_negative():
    turns=parse_speaker_turns('Mac: Hello.')
    with pytest.raises(ValueError):validate_reference(Extraction(rationales=[],reviewed_target_turns=[],notes=''),turns,['Mac'],set())
    assert validate_reference(Extraction(rationales=[],reviewed_target_turns=[0],notes='Greeting only.'),turns,['Mac'],set())['rationales']==[]


def test_scoring_collapses_repeated_labels_but_keeps_turn_references():
    from prepare_pitchshow_rationales import scoring_rationales
    turns=parse_speaker_turns('Mac: Is there a moat?\nMac: There is no moat.')
    extraction=Extraction(rationales=[rationale(turn_index=0,quote='Is there a moat?',utterance_type='question',direction='neutral',decision_link='none'),rationale()],reviewed_target_turns=[0,1],notes='')
    groups=validate_reference(extraction,turns,['Mac'],{'competitive_advantage_assessment'})
    rows=scoring_rationales(groups['rationales'])
    assert len(rows)==1 and len(rows[0]['all_mention_ids'])==2
    assert rows[0]['direction']=='negative' and rows[0]['utterance_type']=='decision_reason'
    assert rows[0]['decision_link']=='explicit'


def test_typography_repair_retains_exact_source_and_original_model_quote():
    turns=parse_speaker_turns("Mac: There's no moat.")
    e=Extraction(rationales=[rationale(turn_index=0,quote='there’s no moat.')],reviewed_target_turns=[0],notes='')
    r=validate_reference(e,turns,['Mac'],{'competitive_advantage_assessment'})['rationales'][0]
    assert r['quote']=="There's no moat."
    assert r['model_quote_before_typography_repair']=='there’s no moat.'


def test_unmatched_quote_requires_third_semantic_check_before_expansion(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import prepare_pitchshow_rationales as m
    calls=[]
    target={'vc_slug':'vc','name':'Mac','aliases':['Mac']}
    episode={'slug':'pitch','transcript':'Mac: There is no moat.'}
    path=tmp_path/'investors/vc/source/episodes/pitch.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(episode))
    extraction=Extraction(rationales=[rationale(turn_index=0,quote='No competitive moat.')],reviewed_target_turns=[0],notes='')
    class Provider:
        def __init__(self,*a,**k): self.client=SimpleNamespace(close=lambda:None)
        def generate(self,request):
            calls.append(request.phase)
            parsed={'verdicts':[{'index':0,'supported':True,'reason':'The source explicitly denies a moat.'}]} if request.phase.startswith('rationale_evidence_check') else extraction.model_dump()
            return SimpleNamespace(parsed=parsed,usage=SimpleNamespace(model_dump=lambda:{}),elapsed_seconds=0,content=json.dumps(parsed))
    monkeypatch.setattr(m,'OpenRouterProvider',Provider)
    taxonomy=[{'label':'competitive_advantage_assessment','definition':'Defensibility','coarse_parent':'competition'}]
    result=m.process(tmp_path,episode,target,{'pitch_window_decision':'Out','later_decision':'Unknown'},taxonomy,'hash','model')
    assert len(calls)==3
    assert result['rationales'][0]['quote']=='There is no moat.'
    assert len(result['expanded_quote_evidence'])==1
    again=m.process(tmp_path,episode,target,{'pitch_window_decision':'Out','later_decision':'Unknown'},taxonomy,'hash','model',offline=True)
    assert len(calls)==3 and again['fingerprint']==result['fingerprint']


def test_credit_failure_stops_further_requests(tmp_path,monkeypatch):
    import json
    from threading import Event
    from types import SimpleNamespace
    import prepare_pitchshow_rationales as m
    calls=[]
    class CreditError(Exception):status_code=402
    class Provider:
        def __init__(self,*a,**k):self.client=SimpleNamespace(close=lambda:None)
        def generate(self,request):calls.append(request);raise CreditError('insufficient credits')
    monkeypatch.setattr(m,'OpenRouterProvider',Provider)
    monkeypatch.setattr(m,'CREDIT_BLOCKED',Event())
    e={'slug':'pitch','transcript':'Mac: No moat.'};t={'vc_slug':'vc','name':'Mac','aliases':['Mac']}
    annotation={'pitch_window_decision':'Out','later_decision':'Unknown'}
    for _ in range(2):
        with pytest.raises(RuntimeError,match='credit budget blocked'):
            m.process(tmp_path,e,t,annotation,[],'hash','model')
    assert len(calls)==1


def test_codex_reuses_legacy_stages_and_records_new_backend(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import prepare_pitchshow_rationales as m
    episode={'slug':'pitch','transcript':'Mac: There is no moat.'}
    target={'vc_slug':'vc','name':'Mac','aliases':['Mac']}
    annotation={'pitch_window_decision':'Out','later_decision':'Unknown'}
    taxonomy=[{'label':'competitive_advantage_assessment'}]
    source=tmp_path/'investors/vc/source/episodes/pitch.json';source.parent.mkdir(parents=True);source.write_text(json.dumps(episode))
    extraction=Extraction(rationales=[rationale(turn_index=0)],reviewed_target_turns=[0],notes='')
    phases=[]
    class Legacy:
        def __init__(self,*a,**k):self.client=SimpleNamespace(close=lambda:None)
        def generate(self,request):
            if request.phase=='rationale_verify':raise RuntimeError('credit exhausted')
            return SimpleNamespace(parsed=extraction.model_dump(),usage=SimpleNamespace(model_dump=lambda:{}),elapsed_seconds=0,content='{}')
    monkeypatch.setattr(m,'OpenRouterProvider',Legacy)
    with pytest.raises(RuntimeError,match='credit exhausted'):
        m.process(tmp_path,episode,target,annotation,taxonomy,'hash','legacy')
    class Codex(Legacy):
        def generate(self,request):
            phases.append(request.phase)
            return SimpleNamespace(parsed=extraction.model_dump(),usage=SimpleNamespace(model_dump=lambda:{}),elapsed_seconds=0,content='{}',raw_metadata={'backend':'codex','model':'gpt-5.6-sol'})
    monkeypatch.setattr(m,'CodexProvider',Codex,raising=False)
    result=m.process(tmp_path,episode,target,annotation,taxonomy,'hash','legacy',backend='codex')
    assert phases==['rationale_verify']
    assert [stage['backend'] for stage in result['generation_stages']]==['openrouter','codex']
    assert result['generation_stages'][-1]['model']=='gpt-5.6-sol'
    receipt=json.loads((tmp_path/'investors/vc/rationale-references/receipts/pitch.json').read_text())
    assert receipt['calls'][-1]['backend']=='codex'
    assert receipt['calls'][-1]['model']=='gpt-5.6-sol'
    again=m.process(tmp_path,episode,target,annotation,taxonomy,'hash','legacy',offline=True,backend='codex')
    assert again==result and phases==['rationale_verify']


def test_completed_legacy_reference_is_unchanged_on_codex_resume(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import prepare_pitchshow_rationales as m
    episode={'slug':'pitch','transcript':'Mac: There is no moat.'}
    target={'vc_slug':'vc','name':'Mac','aliases':['Mac']}
    annotation={'pitch_window_decision':'Out','later_decision':'Unknown'}
    taxonomy=[{'label':'competitive_advantage_assessment'}]
    source=tmp_path/'investors/vc/source/episodes/pitch.json';source.parent.mkdir(parents=True);source.write_text(json.dumps(episode))
    extraction=Extraction(rationales=[rationale(turn_index=0)],reviewed_target_turns=[0],notes='')
    class Legacy:
        def __init__(self,*a,**k):self.client=SimpleNamespace(close=lambda:None)
        def generate(self,request):
            return SimpleNamespace(parsed=extraction.model_dump(),usage=SimpleNamespace(model_dump=lambda:{}),elapsed_seconds=0,content='{}')
    monkeypatch.setattr(m,'OpenRouterProvider',Legacy)
    result=m.process(tmp_path,episode,target,annotation,taxonomy,'hash','legacy')
    path=tmp_path/'investors/vc/rationale-references/pitch.json';before=path.read_bytes()
    def forbidden(*a,**k):raise AssertionError('Completed reference must not construct a provider')
    monkeypatch.setattr(m,'OpenRouterProvider',forbidden);monkeypatch.setattr(m,'CodexProvider',forbidden)
    assert m.process(tmp_path,episode,target,annotation,taxonomy,'hash','legacy',backend='codex')==result
    assert path.read_bytes()==before
