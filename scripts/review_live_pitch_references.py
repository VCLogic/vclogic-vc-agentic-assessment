"""Publish source-bound live-founder scoring overrides without changing originals."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
from prepare_pitchshow_rationales import scoring_rationales

# Reviewed against original indexed dialogue; exclusive first post-founder turn.
# Ambiguous farewell banter is excluded conservatively and documented in provenance.
BOUNDARIES = {'136':158,'129':145,'189':274,'187':157,'119':175,'4':128,
              '5':152,'163':155,'128':224,'117':155,'41':174,'43':113,
              '191':224,'39':112,'126':298,'186':213,'87':195,'71':116,'9':112,
              '130':168,'134':186,'51':117,'45':122}


def corrected_reference(original, end):
    result=deepcopy(original)
    mentions=result.get('rationale_mentions')
    if mentions is None:
        raise ValueError('Cannot correct aggregated references without individual mentions')
    live=[];later=result.get('later_rationales',[])
    for row in mentions:
        if row['turn_index']>=end:
            row['timing']='later_commentary';later.append(row)
        else:live.append(row)
    result['rationale_mentions']=live
    result['rationales']=scoring_rationales(live)
    result['later_rationales']=later
    result['scoring_scope']='live_founder_present_v2'
    result['human_validated']=False
    return result


def main():
    root=Path('evaluation/pitchshow');out=root/'rationale-corpus/live-founder-v2';report=[]
    for path in sorted((root/'investors').glob('*/rationale-references/*.json')):
        ep=path.stem;number=ep.split('-')[0]
        if number not in BOUNDARIES:continue
        source=path.parent.parent/'source/episodes'/path.name
        original=json.loads(path.read_text());end=BOUNDARIES[number]
        result=corrected_reference(original,end)
        result['timing_review']={'reviewer':'codex_source_inspection','human_validated':False,
            'first_post_founder_turn':end,'original_path':str(path),
            'original_sha256':sha256(path.read_bytes()).hexdigest(),
            'source_path':str(source),'source_sha256':sha256(source.read_bytes()).hexdigest(),
            'reason':'Reviewed farewell/exit and following panel or host transition. Includes live decision discussion and reversals; excludes investor-only debrief. Borderline farewell banter conservatively ends at reviewed boundary.'}
        dest=out/path.parent.parent.name/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
        payload=json.dumps(result,indent=2)+'\n'
        if dest.exists() and dest.read_text()!=payload:raise ValueError(f'Immutable override changed: {dest}')
        dest.write_text(payload)
        report.append({'vc_slug':path.parent.parent.name,'episode_slug':ep,
                       'moved_mentions':len(original['rationale_mentions'])-len(result['rationale_mentions']),
                       'path':str(dest),'first_post_founder_turn':end})
    (out/'review-report.json').write_text(json.dumps({'policy':'live_founder_present_v2','human_validated':False,'cases':report},indent=2)+'\n')
    print(f'Published {len(report)} references; moved {sum(r["moved_mentions"] for r in report)} mentions; originals preserved.')

if __name__=='__main__':main()
