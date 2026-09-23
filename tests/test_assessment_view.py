import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vc_clone_graph import rehearsal_cli as cli


def test_assessment_parser_defaults_to_readable_output():
    args = cli._parser().parse_args(['assessment', '--config', 'config.toml', '--session', 'demo'])
    assert args.format == 'markdown'


def test_assessment_reads_verified_baseline_without_runtime(tmp_path, monkeypatch, capsys):
    metadata = {'canonical_run_path': str(tmp_path / 'run'), 'episode_slug': 'pitch', 'canonical_pitch_sha256': 'digest'}
    class Store:
        def verify(self): return {}
        def read_json(self, name):
            return metadata if name == 'session-config.json' else {'vc_slug': 'mac'}
    monkeypatch.setattr(cli, 'locate_session', lambda *args: tmp_path)
    monkeypatch.setattr(cli.RehearsalArtifactStore, 'open', lambda *args: Store())
    investigation = {'rationales': [{'rationale_id': 'R1', 'taxonomy_label': 'founder_market_fit', 'direction': 'positive', 'confidence': .9, 'salience': 'primary', 'justification': 'Relevant experience.', 'pitch_evidence_ids': ['P-1']}]}
    decision = {'decision': 'In', 'decision_confidence': .68, 'controlling_rationale_ids': ['R1'], 'decision_justification': 'Strong fit.', 'diligence_questions': ['Verify revenue.']}
    def load(**kwargs):
        assert kwargs['expected_run_vc_slug'] == 'mac'
        assert kwargs['expected_pitch_sha256'] == 'digest'
        return SimpleNamespace(investigation=SimpleNamespace(model_dump=lambda **kw: investigation), decision=SimpleNamespace(model_dump=lambda **kw: decision), phase1_status='accepted', phase2_status='accepted')
    monkeypatch.setattr(cli, 'load_canonical_baseline_from_run', load)
    config = SimpleNamespace(workspace=tmp_path, rehearsal=SimpleNamespace(output_root='outputs'), resolve_path=lambda x: tmp_path/x)
    cli.command_assessment(config, 'demo', 'markdown')
    output = capsys.readouterr().out
    for text in ['founder_market_fit', 'Relevant experience.', 'Strong fit.', 'Verify revenue.', 'P-1', '68%']:
        assert text in output
    cli.command_assessment(config, 'demo', 'json')
    assert json.loads(capsys.readouterr().out)['decision'] == decision
    metadata.pop('canonical_run_path')
    with pytest.raises(ValueError, match='canonical assessment'):
        cli.command_assessment(config, 'demo', 'markdown')
