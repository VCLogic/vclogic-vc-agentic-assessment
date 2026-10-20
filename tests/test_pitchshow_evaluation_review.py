import importlib.util
import sys
from pathlib import Path

spec=importlib.util.spec_from_file_location('review',Path(__file__).resolve().parents[1]/'scripts/pitchshow_evaluation_review.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def test_wrong_speaker_and_implicit_leakage_fail_closed():
    episode={'slug':'example','panel':[{'name':'Mac'}], 'transcript':'Founder: We have ten customers.\nFounder: Thanks for joining our round!\nMac: I am out.\nHost: Mac invested later.'}
    review=m.EpisodeReview(episode_kind='pitch',founder_speakers=['Founder'],pitch_start_turn=0,pitch_end_turn=2,excluded_turns=[],boundary_reason='before verdict',decisions=[m.Decision(vc_slug='mac',pitch_window_decision='Out',initial_response='Out',decision_context='later_diligence',pitch_evidence=[m.Evidence(turn_index=2,quote='I am out.')],later_decision='In',later_evidence=[m.Evidence(turn_index=3,quote='Mac invested later.')],condition=None,reason='later reversal')])
    verify=m.Verification(boundary_supported=True,remove_lines=[1],safe_after_removal=True,reason='remove reaction',decisions=[m.Verdict(vc_slug='mac',pitch_decision_supported=True,later_decision_supported=True,pitch_evidence=[m.Evidence(turn_index=2,quote='I am out.')],later_evidence=[m.Evidence(turn_index=3,quote='Mac invested later.')],condition=None,reason='exact')])
    pitch,segments,turns=m.prepare_candidate(episode,review)
    result=m.finalize(episode,[{'vc_slug':'mac','aliases':['Mac']}],review,verify,pitch,segments,turns)
    assert result['pitch']=='Founder: We have ten customers.\n'
    assert result['annotations'][0]['pitch_window_decision']=='Out'
    assert result['annotations'][0]['later_decision']=='In'
    assert result['annotations'][0]['evaluation_eligible']
    result=m.finalize(episode,[{'vc_slug':'mac','aliases':['Someone else']}],review,verify,pitch,segments,turns)
    assert not result['annotations'][0]['evaluation_eligible']
    assert result['annotations'][0]['pitch_window_decision']=='Unobserved'
    verify.boundary_supported=False
    result=m.finalize(episode,[{'vc_slug':'mac','aliases':['Mac']}],review,verify,pitch,segments,turns)
    assert not result['pitch']
    assert not result['annotations'][0]['evaluation_eligible']


def test_schemas_supported_by_actual_provider():
    from vc_clone_graph.providers.base import GenerationRequest
    for schema in (m.EpisodeReview,m.Verification):
        GenerationRequest(phase='test',prompt='test',schema=schema.model_json_schema())


def test_panel_member_cannot_be_declared_founder():
    import pytest
    episode={'slug':'example','panel':[{'name':'Mac Conwell'}],'transcript':'Mac: I love this company.\nFounder: Thanks!'}
    review=m.EpisodeReview(episode_kind='pitch',founder_speakers=['Mac'],pitch_start_turn=0,pitch_end_turn=1,excluded_turns=[],boundary_reason='bad identity',decisions=[])
    with pytest.raises(ValueError,match='overlaps'):
        m.prepare_candidate(episode,review)


def test_factual_uses_of_investor_names_are_preserved_for_semantic_review():
    episode={'slug':'example','panel':[{'name':'Mac Conwell'},{'name':'Elizabeth Yin'}], 'transcript':'Founder: Our software runs on the Mac. Purely Elizabeth is a competitor.\nMac: I am out.'}
    review=m.EpisodeReview(episode_kind='pitch',founder_speakers=['Founder'],pitch_start_turn=0,pitch_end_turn=1,excluded_turns=[],boundary_reason='before verdict',decisions=[])
    pitch,segments,turns=m.prepare_candidate(episode,review)
    assert 'on the Mac' in pitch
    assert 'Purely Elizabeth' in pitch
