#!/usr/bin/env python3
"""Preserve the selected original LangGraph cohorts and verify every copied byte."""
from pathlib import Path
from hashlib import sha256
import json,shutil,re

def main():
    source=Path('/home/dpasch01/vc-digital-twins/langgraph-vc-clone-framework')
    dest=Path('outputs/legacy-langgraph/workspace')
    roots=['outputs/canonical-v4-v41-portfolio-2026-08-15','outputs/panel-question-phase1-canary-2026-08-20','evaluation/canonical_runs_v4_v41_portfolio_2026-08-15.json','evaluation/labels','evaluation/phase1_ground_truth_rationales','reports/evaluation/canonical-v4-v41-portfolio-2026-08-15','reports/evaluation/phase1-rationale-canonical-v4-v41-2026-08-16','inputs/data/pitch_variants/panel-question-augmentation-v1','inputs/taxonomy','configs']
    vcs=['charles-hudson-precursor-ventures','cyan-banister-long-journey-ventures','elizabeth-yin-hustle-fund','jesse-middleton-flybridge','jillian-manus-structure-capital','phil-nadel']
    for vc in vcs:
     roots += [f'inputs/data/investors/{vc}/pitches',f'inputs/data/investors/{vc}/audits',f'inputs/data/investors/{vc}/manifests',f'inputs/wiki/{vc}']
    rows=[];missing=[]
    for relative in roots:
     path=source/relative
     if not path.exists():missing.append(relative);continue
     files=sorted(path.rglob('*')) if path.is_dir() else [path]
     for file in files:
      if not file.is_file():continue
      if file.is_symlink():raise ValueError(f'unexpected symlink {file}')
      rel=file.relative_to(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
      digest=sha256(file.read_bytes()).hexdigest()
      if target.exists():
       if sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError(f'existing archive differs {target}')
      else:shutil.copy2(file,target)
      assert sha256(target.read_bytes()).hexdigest()==digest
      rows.append({'path':str(rel),'sha256':digest,'bytes':file.stat().st_size})
     print(relative,flush=True)
    manifest={'schema':'legacy-langgraph-import-v1','source_root':str(source),'archive_root':str(dest),'files':rows,'missing_optional_roots':missing,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'all_copied_files_hash_verified':True}
    recovered=[]
    canonical=source/'outputs/canonical-v4-v41-portfolio-2026-08-15/investors'
    for provenance_path in canonical.glob('*/*/input-provenance.json'):
     provenance=json.loads(provenance_path.read_text()); pitch=Path(provenance['pitch_path'])
     if pitch.exists() and sha256(pitch.read_bytes()).hexdigest()==provenance['pitch_sha256']:continue
     prompt_path=provenance_path.parent/'phase1/turn-01/investigation-model-response.json'
     prompt=json.loads(prompt_path.read_text())['prompt']; found=False
     for match in re.finditer(r'"pitch"\s*:\s*',prompt):
      value,_=json.JSONDecoder().raw_decode(prompt[match.end():])
      for text in (value,value+'\n',value.strip()+'\n'):
       if sha256(text.encode()).hexdigest()!=provenance['pitch_sha256']:continue
       target=Path('outputs/legacy-langgraph/recovered-inputs')/provenance_path.parent.parent.name/(provenance_path.parent.name+'.txt')
       target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
       recovered.append({'path':str(target),'sha256':provenance['pitch_sha256'],'prompt_source':str(prompt_path.relative_to(source)), 'prompt_source_sha256':sha256(prompt_path.read_bytes()).hexdigest()})
       found=True;break
      if found:break
     if not found:raise ValueError(f'cannot recover original pitch {provenance_path}')
    manifest['recovered_original_inputs']=recovered
    p=Path('evaluation/pitchshow/legacy-import-manifest.json');p.write_text(json.dumps(manifest,indent=2)+'\n')
    print({k:v for k,v in manifest.items() if k!='files'},flush=True)

if __name__ == '__main__':
    main()
