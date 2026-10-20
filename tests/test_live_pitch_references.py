import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_rebuild_uses_mentions_and_keeps_post_exit_separate():
    from review_live_pitch_references import corrected_reference
    def mention(turn, direction):
        return dict(rationale_label='market_size_assessment',direction=direction,
                    turn_index=turn,quote='quote',timing='original_pitch',
                    utterance_type='decision_explanation',decision_link='supports_out',
                    salience='primary',source_rationale_id=f'R{turn}',
                    evidence_turns=[{'turn_index':turn,'quote':'quote'}])
    original={'rationale_mentions':[mention(10,'positive'),mention(20,'negative')],
              'rationales':[], 'later_rationales':[], 'excluded_rationales':[]}
    revised=corrected_reference(original,15)
    assert revised['rationales'][0]['direction']=='positive'
    assert revised['later_rationales'][0]['turn_index']==20
    assert original['rationale_mentions'][1]['timing']=='original_pitch'


def test_stage_diagnostics_separate_detection_direction_and_weight():
    from evaluate_pitchshow_results import rationale_stage_errors
    ref=[dict(rationale_label='a',direction='negative',decision_link='explicit'),
         dict(rationale_label='b',direction='negative',decision_link='explicit'),
         dict(rationale_label='c',direction='negative',decision_link='explicit')]
    pred=[dict(taxonomy_label='b',direction='positive',rationale_id='R1'),
          dict(taxonomy_label='c',direction='negative',rationale_id='R2')]
    result=rationale_stage_errors(pred,ref,[],False)
    assert result['missing_decisive_labels']==['a']
    assert result['direction_disagreements']==['b']
    assert result['detected_not_controlling']==['c']
