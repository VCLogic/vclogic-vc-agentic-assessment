import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('consistency',Path(__file__).resolve().parents[1]/'scripts/investor_consistency_review.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

def packet():
 return {'phase2':{'decision':'In','investment_likelihood':.6,'decision_confidence':.7},'pitch_evidence':[{'evidence_id':'P-001','text':'pitch'}],'wiki_evidence':[{'evidence_id':'W-a','text':'rule'}],'historical_evidence':[{'evidence_id':'H-a','text':'history'}]}

def response():
 return dict(action='RETAIN_PHASE2',decision=None,investment_likelihood=None,decision_confidence=None,justification='No supported contradiction.',revision_basis=None,checks=[dict(check=k,status='consistent',finding='Supported.',evidence_ids=['P-001','W-a']) for k in ['investor_constraints','historical_comparisons','evidence_weighting']],evidence_gaps=[])

def test_retain_inherits_phase2():
 assert m.resolve_review(response(),packet())==packet()['phase2']

def test_retain_cannot_silently_change_scores():
 r=response();r['decision_confidence']=.9
 with pytest.raises(ValueError):m.resolve_review(r,packet())

def test_revision_needs_actual_change_and_supported_defect():
 r=response();r.update(action='REVISE_PHASE2',decision='Out',investment_likelihood=.3,decision_confidence=.65,revision_basis='Explicit constraint contradicts original decision.');r['checks'][0]['status']='inconsistent'
 assert m.resolve_review(r,packet())['decision']=='Out'
 r['checks'][0]['evidence_ids']=['invented']
 with pytest.raises(ValueError):m.resolve_review(r,packet())

def test_all_three_checks_required():
 r=response();r['checks'].pop()
 with pytest.raises(ValueError):m.resolve_review(r,packet())

def test_packet_excludes_outcome_metadata(tmp_path):
 import json
 p=tmp_path/'run';(p/'phase1').mkdir(parents=True);(p/'phase2').mkdir()
 (p/'phase1/investigation.json').write_text(json.dumps({'episode_slug':'1-demo','rationales':[]}))
 (p/'phase2/decision.json').write_text(json.dumps({'episode_slug':'1-demo',**packet()['phase2']}))
 (p/'summary.json').write_text(json.dumps({'actual_decision':'SECRET_OUTCOME','stratum':'false_in'}))
 pitch=tmp_path/'pitch.txt';pitch.write_text('Founder pitch.')
 result,_=m.build_packet(p,pitch,'Investor')
 text=m.build_prompt(result)
 assert 'SECRET_OUTCOME' not in text and 'false_in' not in text
 assert result['phase2']['decision']=='In'

def test_native_artifacts_preserve_upstream_and_resume_without_call(tmp_path):
 import json
 from vc_clone_graph.providers.base import GenerationResult,Usage
 run=tmp_path/'run';(run/'phase1').mkdir(parents=True);(run/'phase2').mkdir()
 (run/'phase1/investigation.json').write_text(json.dumps({'episode_slug':'1-demo','rationales':[]}))
 decision=run/'phase2/decision.json';decision.write_text(json.dumps({'episode_slug':'1-demo',**packet()['phase2']}));before=decision.read_bytes()
 pitch=tmp_path/'pitch.txt';pitch.write_text('Founder pitch.')
 class Fake:
  calls=0
  def generate(self,request):
   self.calls+=1;r=response()
   for check in r['checks']:check['evidence_ids']=['P-001']
   return GenerationResult(parsed=r,content=json.dumps(r),usage=Usage(cost_usd=.01),elapsed_seconds=.1,raw_metadata={})
 provider=Fake()
 first=m.run_review(run,pitch,'Investor',execute=True,provider=provider)
 assert first['status']=='completed' and first['decision']['decision_source']=='phase2'
 assert first['decision']['decision_confidence']==.7
 assert decision.read_bytes()==before
 assert (run/'phase3/decision.sha256').read_text().strip()==m.sha(run/'phase3/decision.json')
 assert m.run_review(run,pitch,'Investor',execute=True,provider=provider)==first
 assert provider.calls==1

def test_different_saved_excerpts_for_same_evidence_id_are_preserved(tmp_path):
 import json
 run=tmp_path/'run';(run/'phase1/turn-01').mkdir(parents=True);(run/'phase2/turn-01').mkdir(parents=True)
 (run/'phase1/investigation.json').write_text('{}');(run/'phase2/decision.json').write_text('{}')
 for phase,text in [('phase1','first excerpt'),('phase2','second excerpt')]:
  (run/phase/'turn-01/wiki-reads.json').write_text(json.dumps([{'evidence_id':'W-a','text':text}]))
 pitch=tmp_path/'pitch';pitch.write_text('pitch')
 p,_=m.build_packet(run,pitch,'Investor')
 assert {e['text'] for e in p['wiki_evidence']}=={'first excerpt','second excerpt'}
 assert all(e['artifact_path'] for e in p['wiki_evidence'])

def test_retain_exact_echo_is_normalized_but_changes_rejected():
 r=response();r.update(packet()['phase2'])
 assert m.resolve_review(r,packet())==packet()['phase2']
 normalized=m.normalize_review(r,packet())
 assert normalized['decision'] is None and normalized['investment_likelihood'] is None
 r['investment_likelihood']=.61
 with pytest.raises(ValueError):m.resolve_review(r,packet())
