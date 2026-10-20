"""Publication must keep outcome evidence outside model-visible pitches."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "pitchshow_publish_test", SCRIPTS / "prepare_pitchshow_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.remove(str(SCRIPTS))


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "evaluation/pitchshow"
    vc = "charles-hudson-precursor-ventures"
    folder = root / "investors" / vc
    slug = "example-outcome-revealing-title"
    episode = {
        "slug": slug,
        "url": "https://www.thepitch.show/example",
        "transcript": "Founder: We have ten customers.\nCharles: I am out.",
    }
    source = folder / "source/episodes" / f"{slug}.json"
    module.write(source, episode)
    module.write(folder / "source/decisions.json", [{"episode_slug": slug, "reported_decision": "In"}])
    module.write(folder / "source/collection.json", {"complete": True, "missing_transcripts": []})
    result = {
        "annotations": [{"vc_slug": vc, "pitch_window_decision": "Out", "later_decision": "Unknown",
                         "evaluation_eligible": True, "review_status": "machine_reviewed_and_verified"}],
        "pitch_safe": True,
        "pitch": "Founder: We have ten customers.\n",
        "review": {"episode_kind": "pitch","pitch_start_turn": 0, "pitch_end_turn": 1, "boundary_reason": "Before first decision"},
        "segments": [], "removed_candidate_lines": [], "verification": {"safe_after_removal": True},
    }
    receipt = root / "reviews" / f"{slug}.json"
    module.write(receipt, result)
    return root, vc, folder, slug, episode, source, result, receipt


def publish(dataset):
    root, vc, folder, slug, episode, source, result, _ = dataset
    summary = module.publish(root, {slug: episode}, {}, {slug: [(folder, source)]}, {slug: result})
    row = json.loads((folder / "annotations.json").read_text())[0]
    return summary, row


def test_raw_evidence_stays_separate_and_audit_receipt_resolves(dataset):
    root, vc, folder, slug, episode, source, result, receipt = dataset
    original = source.read_bytes()
    summary, row = publish(dataset)
    pitch = folder / row["pitch_path"]
    assert pitch.read_text() == result["pitch"]
    assert slug not in pitch.name
    assert "I am out" not in pitch.read_text()
    assert (folder / "transcripts" / f"{slug}.txt").read_text() == episode["transcript"]
    assert source.read_bytes() == original
    assert row["pitch_window_decision"] == "Out"
    assert row["reported_profile_investment"] == "In"
    assert summary["investors"][vc]["eligible_pitches"] == 1
    audit_path = folder / "audits" / f"{row['neutral_pitch_id']}.json"
    audit = json.loads(audit_path.read_text())
    assert (audit_path.parent / audit["review_receipt"]).resolve() == receipt
    assert receipt.is_file()


@pytest.mark.parametrize("prior", ["In", "Unobserved"])
def test_prior_label_conflict_quarantines_previously_published_pitch(dataset, prior):
    _, vc, folder, slug, _, _, _, _ = dataset
    _, previous = publish(dataset)
    old_pitch = folder / previous["pitch_path"]
    module.write(Path("evaluation/labels/charles_pitch_window_decisions.json"),
                 [{"episode_slug": slug, "pitch_window_decision": prior}])
    summary, row = publish(dataset)
    assert row["prior_audit_conflict"] is True
    assert row["evaluation_eligible"] is False
    assert row["review_status"] == "conflicts_with_prior_audit"
    assert "pitch_path" not in row
    assert not old_pitch.exists()
    assert summary["investors"][vc]["eligible_pitches"] == 0
    assert summary["investors"][vc]["prior_audit_conflicts"] == 1


def test_removes_stale_generated_files_and_preserves_current_files(dataset):
    _, _, folder, _, _, _, _, _ = dataset
    _, before = publish(dataset)
    for kind, suffix in (("pitches", ".txt"), ("audits", ".json")):
        (folder / kind / f"pitch-obsolete{suffix}").write_text("obsolete")
        (folder / kind / "README.md").write_text("keep")
    _, after = publish(dataset)
    assert after["neutral_pitch_id"] == before["neutral_pitch_id"]
    assert (folder / after["pitch_path"]).is_file()
    for kind, suffix in (("pitches", ".txt"), ("audits", ".json")):
        assert not (folder / kind / f"pitch-obsolete{suffix}").exists()
        assert (folder / kind / "README.md").read_text() == "keep"


def test_unsafe_pitch_cannot_be_counted_or_published(dataset):
    _, vc, folder, _, _, _, result, _ = dataset
    _, prior = publish(dataset)
    result["pitch_safe"] = False
    summary, row = publish(dataset)
    assert row["evaluation_eligible"] is False
    assert row["pitch_review_status"] == "not_published"
    assert row["review_status"] == "decision_verified_pitch_not_safe"
    assert not (folder / prior["pitch_path"]).exists()
    assert summary["investors"][vc]["eligible_pitches"] == 0
