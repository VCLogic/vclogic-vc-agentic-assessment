from hashlib import sha256
import json
from pathlib import Path

import pytest

import vc_clone_graph.importer as importer_module
from vc_clone_graph.firewall import InputBoundaryError, verify_package
from vc_clone_graph.importer import import_assets
from vc_clone_graph.manifest_builder import build_episode_manifest
from vc_clone_graph.precedent_builder import build_precedent_corpus
from vc_clone_graph.precedents import PrecedentCorpus
from vc_clone_graph.portfolio_memory import (
    DisclosureEvidence,
    PortfolioDisclosure,
    PortfolioMemoryCorpus,
    PortfolioMemoryIndex,
    make_disclosure_id,
)


VC = "elizabeth-yin-hustle-fund"
EPISODE = "135-thoras-ai-the-twin-effect"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def make_package(tmp_path: Path) -> Path:
    root = tmp_path / "inputs"
    pitch = root / f"data/investors/{VC}/pitches/{EPISODE}.txt"
    audit = root / f"data/investors/{VC}/audits/{EPISODE}.json"
    registry = root / f"investors/{VC}.toml"
    wiki = root / f"wiki/{VC}/persona.md"
    taxonomy = root / "taxonomy/codebook_v_final.json"
    for path in (pitch, audit, registry, wiki, taxonomy):
        path.parent.mkdir(parents=True, exist_ok=True)
    pitch.write_text("Founder: We have five paid pilots.\n", encoding="utf-8")
    registry.write_text(
        f'vc_slug = "{VC}"\nwiki_path = "wiki/{VC}"\n', encoding="utf-8"
    )
    wiki.write_text("# Persona\nEvidence-led pre-seed investor.\n", encoding="utf-8")
    taxonomy.write_text('{"rationales": []}\n', encoding="utf-8")
    audit.write_text(
        json.dumps(
            {
                "schema": "pitch-audit-v1",
                "status": "approved",
                "episode_slug": EPISODE,
                "vc_slug": VC,
                "pitch_path": f"data/investors/{VC}/pitches/{EPISODE}.txt",
                "pitch_sha256": digest(pitch),
                "target_company_aliases": ["Thoras.ai", "Thoras AI"],
                "leakage_checklist": {"actual_label_in_package": False},
            }
        ),
        encoding="utf-8",
    )
    files = []
    for path in (pitch, audit, registry, wiki, taxonomy):
        files.append(
            {
                "destination": path.relative_to(root).as_posix(),
                "sha256": digest(path),
            }
        )
    manifest = root / f"data/investors/{VC}/source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "langgraph-vc-clone-source-manifest-v1",
                "vc_slug": VC,
                "episode_slug": EPISODE,
                "files": sorted(files, key=lambda row: row["destination"]),
            }
        ),
        encoding="utf-8",
    )
    return root


def add_precedent_corpus(root: Path, *, persist_index: bool = True) -> Path:
    source = root.parent / "precedent-source"
    source.mkdir(exist_ok=True)
    slug = "18-rowvigor"
    (source / f"{slug}.json").write_text(
        json.dumps({"transcript": "Narrator: Historical garden tools pitch."}),
        encoding="utf-8",
    )
    ledger = root.parent / "precedent-ledger.json"
    ledger.write_text(
        json.dumps(
            [
                {
                    "episode_slug": slug,
                    "pitch_window_decision": "Unobserved",
                    "decision_context": "unclear",
                    "audit_notes": "No observed decision.",
                }
            ]
        ),
        encoding="utf-8",
    )
    corpus = root / f"data/investors/{VC}/precedents"
    build_precedent_corpus(source, ledger, corpus, ("Elizabeth",))
    if persist_index:
        PrecedentCorpus.build(corpus).save(corpus / "precedents-index.json")
    return corpus


def bind_corpus(root: Path, corpus: Path) -> None:
    manifest = root / f"data/investors/{VC}/source-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"].extend(
        {
            "destination": path.relative_to(root).as_posix(),
            "sha256": digest(path),
        }
        for path in sorted(corpus.rglob("*"))
        if path.is_file()
    )
    data["files"].sort(key=lambda row: row["destination"])
    manifest.write_text(json.dumps(data), encoding="utf-8")


class FakeEmbedder:
    metadata = {"backend": "fake", "model": "fake", "revision": "one"}

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts):
        return [[1.0, 0.0] for _ in texts]


def add_portfolio_memory(root: Path) -> Path:
    destination = root / f"data/investors/{VC}/portfolio-memory"
    destination.mkdir(parents=True)
    slug = "18-rowvigor"
    disclosure = PortfolioDisclosure(
        disclosure_id=make_disclosure_id(VC, slug, 4, "LetsMeet"),
        vc_slug=VC,
        company_name="LetsMeet",
        aliases=("LetsMeet",),
        descriptor="group scheduling",
        relationship="investment",
        observed_overlap="possible",
        observed_consequence="unclear",
        source_episode_slug=slug,
        source_episode_number=18,
        evidence=(DisclosureEvidence(
            source_sha256="a" * 64, turn_index=4, speaker="Elizabeth",
            text="I invested in LetsMeet.",
        ),),
        confidence=0.95,
        validation_status="automated_candidate",
    )
    corpus = PortfolioMemoryCorpus(VC, (disclosure,))
    events = destination / "disclosure-events.jsonl"
    corpus.save_events(events)
    index = PortfolioMemoryIndex.build(corpus, FakeEmbedder())
    index_file = destination / "embedding-index.json"
    index.save(index_file, events)
    (destination / "company-index.json").write_text(json.dumps([
        row.model_dump(mode="json") for row in index.for_target("999-all").entities
    ]))
    (destination / "extraction-audit.jsonl").write_text(
        json.dumps({"status": "accepted", "disclosure_id": disclosure.disclosure_id}) + "\n"
    )
    payload = {
        "schema": "portfolio-memory-corpus-manifest-v1",
        "vc_slug": VC,
        "accepted_count": 1,
        "named_count": 1,
        "anonymous_count": 0,
        "rejected_count": 0,
        "candidate_unextracted_count": 0,
        "files": {
            path.name: digest(path)
            for path in destination.iterdir()
            if path.name != "corpus-manifest.json"
        },
        "embedding": index.metadata,
    }
    (destination / "corpus-manifest.json").write_text(json.dumps(payload))
    return destination


def rebind_manifest_file(root: Path, path: Path) -> None:
    manifest = root / f"data/investors/{VC}/source-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    destination = path.relative_to(root).as_posix()
    for row in data["files"]:
        if row["destination"] == destination:
            row["sha256"] = digest(path)
            break
    else:
        raise AssertionError(f"manifest row not found: {destination}")
    manifest.write_text(json.dumps(data), encoding="utf-8")


def test_verifies_hash_bound_inference_package(tmp_path: Path) -> None:
    verified = verify_package(make_package(tmp_path), VC, EPISODE)
    assert verified.pitch.name == f"{EPISODE}.txt"
    assert verified.wiki.name == VC
    assert verified.target_company_aliases == ("Thoras.ai", "Thoras AI")
    assert verified.precedents is None
    assert verified.precedent_manifest is None
    assert verified.portfolio_memory is None


def test_verifies_manifest_bound_portfolio_memory(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    portfolio = add_portfolio_memory(root)
    bind_corpus(root, portfolio)
    verified = verify_package(root, VC, EPISODE)
    assert verified.portfolio_memory == portfolio
    assert verified.portfolio_manifest == portfolio / "corpus-manifest.json"


@pytest.mark.parametrize("tamper", ["events", "index", "manifest", "extra"])
def test_rejects_invalid_portfolio_memory(tmp_path: Path, tamper: str) -> None:
    root = make_package(tmp_path)
    portfolio = add_portfolio_memory(root)
    bind_corpus(root, portfolio)
    target = {
        "events": portfolio / "disclosure-events.jsonl",
        "index": portfolio / "embedding-index.json",
        "manifest": portfolio / "corpus-manifest.json",
        "extra": portfolio / "unexpected.json",
    }[tamper]
    target.write_bytes(target.read_bytes() + b" " if target.exists() else b"{}")
    with pytest.raises(InputBoundaryError, match="portfolio|manifest hash"):
        verify_package(root, VC, EPISODE)


@pytest.mark.parametrize("persist_index", [False, True])
def test_verifies_manifest_bound_precedent_corpus(
    tmp_path: Path, persist_index: bool
) -> None:
    root = make_package(tmp_path)
    corpus = add_precedent_corpus(root, persist_index=persist_index)
    bind_corpus(root, corpus)

    verified = verify_package(root, VC, EPISODE)

    assert verified.precedents == corpus
    assert verified.precedent_manifest == corpus / "corpus-manifest.json"


@pytest.mark.parametrize(
    "tamper",
    ["missing-chunks", "record", "source", "manifest", "index"],
)
def test_rejects_incomplete_or_tampered_precedent_corpus(
    tmp_path: Path, tamper: str
) -> None:
    root = make_package(tmp_path)
    corpus = add_precedent_corpus(root)
    bind_corpus(root, corpus)
    targets = {
        "missing-chunks": corpus / "chunks.jsonl",
        "record": corpus / "records/18-rowvigor.json",
        "source": corpus / "sources/18-rowvigor.json",
        "manifest": corpus / "corpus-manifest.json",
        "index": corpus / "precedents-index.json",
    }
    target = targets[tamper]
    if tamper == "missing-chunks":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(InputBoundaryError, match="precedent|manifest hash"):
        verify_package(root, VC, EPISODE)


def test_requires_every_precedent_file_to_be_manifest_bound(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    corpus = add_precedent_corpus(root)
    bind_corpus(root, corpus)
    manifest = root / f"data/investors/{VC}/source-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"] = [
        row for row in data["files"] if not row["destination"].endswith("chunks.jsonl")
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(InputBoundaryError, match="manifest-bound"):
        verify_package(root, VC, EPISODE)


@pytest.mark.parametrize(
    "relative",
    ["reports/leak.json", "records/nested/leak.json", "unexpected.json"],
)
def test_rejects_unexpected_paths_inside_precedent_corpus(
    tmp_path: Path, relative: str
) -> None:
    root = make_package(tmp_path)
    corpus = add_precedent_corpus(root)
    bind_corpus(root, corpus)
    unexpected = corpus / relative
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("{}", encoding="utf-8")

    with pytest.raises(InputBoundaryError, match="forbidden|precedent"):
        verify_package(root, VC, EPISODE)


def test_rejects_symlink_inside_precedent_corpus(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    corpus = add_precedent_corpus(root)
    bind_corpus(root, corpus)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (corpus / "records/link.json").symlink_to(outside)

    with pytest.raises(InputBoundaryError, match="symlink|precedent"):
        verify_package(root, VC, EPISODE)


@pytest.mark.parametrize("mutation", ["arbitrary", "missing", "extra", "tampered"])
def test_rejects_semantically_invalid_chunks_even_when_rebound(
    tmp_path: Path, mutation: str
) -> None:
    root = make_package(tmp_path)
    corpus = add_precedent_corpus(root, persist_index=False)
    bind_corpus(root, corpus)
    chunks = corpus / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks.read_text().splitlines() if line]
    if mutation == "arbitrary":
        chunks.write_text('{"arbitrary":true}\n', encoding="utf-8")
    elif mutation == "missing":
        chunks.write_text("", encoding="utf-8")
    elif mutation == "extra":
        chunks.write_text(
            "".join(json.dumps(row) + "\n" for row in [*rows, rows[0]]),
            encoding="utf-8",
        )
    else:
        rows[0]["text"] = "tampered historical text"
        chunks.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    rebind_manifest_file(root, chunks)

    with pytest.raises(InputBoundaryError, match="precedent chunk"):
        verify_package(root, VC, EPISODE)


def test_rejects_symlink_ancestor_of_precedent_corpus(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    corpus = add_precedent_corpus(root, persist_index=False)
    bind_corpus(root, corpus)
    investor = root / f"data/investors/{VC}"
    outside = tmp_path / "outside-investor"
    investor.rename(outside)
    investor.symlink_to(outside, target_is_directory=True)

    with pytest.raises(InputBoundaryError, match="symlink|escape"):
        verify_package(root, VC, EPISODE)


def test_verifies_selected_manifest_bound_taxonomy(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    taxonomy = root / "taxonomy/codebook_v2.json"
    taxonomy.write_text('[{"label":"check_size_optionality_assessment"}]\n')
    manifest = root / f"data/investors/{VC}/source-manifest.json"
    manifest_data = json.loads(manifest.read_text())
    manifest_data["files"].append(
        {
            "destination": taxonomy.relative_to(root).as_posix(),
            "sha256": digest(taxonomy),
        }
    )
    manifest.write_text(json.dumps(manifest_data))

    verified = verify_package(
        root,
        VC,
        EPISODE,
        taxonomy_path="taxonomy/codebook_v2.json",
    )

    assert verified.taxonomy == taxonomy


def test_rejects_selected_taxonomy_missing_from_manifest(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    taxonomy = root / "taxonomy/codebook_v2.json"
    taxonomy.write_text('[{"label":"check_size_optionality_assessment"}]\n')

    with pytest.raises(InputBoundaryError, match="required files"):
        verify_package(
            root,
            VC,
            EPISODE,
            taxonomy_path="taxonomy/codebook_v2.json",
        )


@pytest.mark.parametrize(
    ("aliases", "message"),
    [
        (None, "target company aliases"),
        ([], "target company aliases"),
        (["Thoras.ai", "thoras.AI"], "unique"),
        (["Thoras\nAI"], "printable"),
        (["x" * 121], "120"),
    ],
)
def test_rejects_invalid_target_company_aliases(
    tmp_path: Path, aliases: object, message: str
) -> None:
    root = make_package(tmp_path)
    audit = root / f"data/investors/{VC}/audits/{EPISODE}.json"
    data = json.loads(audit.read_text())
    if aliases is None:
        data.pop("target_company_aliases")
    else:
        data["target_company_aliases"] = aliases
    audit.write_text(json.dumps(data))
    manifest = root / f"data/investors/{VC}/source-manifest.json"
    manifest_data = json.loads(manifest.read_text())
    for row in manifest_data["files"]:
        if row["destination"].endswith(f"audits/{EPISODE}.json"):
            row["sha256"] = digest(audit)
    manifest.write_text(json.dumps(manifest_data))

    with pytest.raises(InputBoundaryError, match=message):
        verify_package(root, VC, EPISODE)


def test_verifies_two_episode_scoped_manifests_in_one_investor_package(
    tmp_path: Path,
) -> None:
    root = make_package(tmp_path)
    legacy = root / f"data/investors/{VC}/source-manifest.json"
    manifests = root / f"data/investors/{VC}/manifests"
    manifests.mkdir()
    legacy.rename(manifests / f"{EPISODE}.json")

    second = "136-second-company"
    first_pitch = root / f"data/investors/{VC}/pitches/{EPISODE}.txt"
    first_audit = root / f"data/investors/{VC}/audits/{EPISODE}.json"
    second_pitch = root / f"data/investors/{VC}/pitches/{second}.txt"
    second_audit = root / f"data/investors/{VC}/audits/{second}.json"
    second_pitch.write_text("Founder: We have ten paid pilots.\n", encoding="utf-8")
    audit_data = json.loads(first_audit.read_text())
    audit_data.update(
        {
            "episode_slug": second,
            "pitch_path": f"data/investors/{VC}/pitches/{second}.txt",
            "pitch_sha256": digest(second_pitch),
        }
    )
    second_audit.write_text(json.dumps(audit_data), encoding="utf-8")
    first_manifest = json.loads((manifests / f"{EPISODE}.json").read_text())
    shared = [
        row
        for row in first_manifest["files"]
        if f"/{EPISODE}." not in row["destination"]
    ]
    first_manifest["episode_slug"] = second
    first_manifest["files"] = shared + [
        {
            "destination": second_pitch.relative_to(root).as_posix(),
            "sha256": digest(second_pitch),
        },
        {
            "destination": second_audit.relative_to(root).as_posix(),
            "sha256": digest(second_audit),
        },
    ]
    (manifests / f"{second}.json").write_text(
        json.dumps(first_manifest), encoding="utf-8"
    )

    first_verified = verify_package(root, VC, EPISODE)
    second_verified = verify_package(root, VC, second)

    assert first_verified.manifest.name == f"{EPISODE}.json"
    assert second_verified.manifest.name == f"{second}.json"


def test_rejects_hidden_label_inside_inference_root(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    (root / "actual_decision.json").write_text('{"label":"In"}')
    with pytest.raises(InputBoundaryError, match="forbidden"):
        verify_package(root, VC, EPISODE)


def test_rejects_complete_transcript_directory(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    episodes = root / "data/episodes"
    episodes.mkdir(parents=True)
    (episodes / f"{EPISODE}.json").write_text("{}")
    with pytest.raises(InputBoundaryError, match="forbidden"):
        verify_package(root, VC, EPISODE)


def test_rejects_modified_pitch(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    pitch = root / f"data/investors/{VC}/pitches/{EPISODE}.txt"
    pitch.write_text(pitch.read_text() + "changed")
    with pytest.raises(InputBoundaryError, match="hash"):
        verify_package(root, VC, EPISODE)


def test_rejects_unapproved_audit(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    audit = root / f"data/investors/{VC}/audits/{EPISODE}.json"
    data = json.loads(audit.read_text())
    data["status"] = "pending"
    audit.write_text(json.dumps(data))
    manifest = root / f"data/investors/{VC}/source-manifest.json"
    manifest_data = json.loads(manifest.read_text())
    for row in manifest_data["files"]:
        if row["destination"].endswith(f"audits/{EPISODE}.json"):
            row["sha256"] = digest(audit)
    manifest.write_text(json.dumps(manifest_data))
    with pytest.raises(InputBoundaryError, match="approved"):
        verify_package(root, VC, EPISODE)


def test_importer_copies_only_manifest_bound_inference_assets(tmp_path: Path) -> None:
    source_inputs = make_package(tmp_path / "source")
    source_root = source_inputs
    forbidden = source_root / "data/episodes"
    forbidden.mkdir(parents=True)
    (forbidden / f"{EPISODE}.json").write_text('{"actual":"In"}')
    destination = tmp_path / "destination"

    imported = import_assets(source_root, destination, VC, EPISODE)

    verify_package(imported, VC, EPISODE)
    assert not (destination / "data/episodes").exists()
    assert (
        destination / f"data/investors/{VC}/manifests/{EPISODE}.json"
    ).is_file()


def test_importer_binds_registry_when_source_manifest_omits_it(tmp_path: Path) -> None:
    source = make_package(tmp_path / "source")
    manifest = source / f"data/investors/{VC}/source-manifest.json"
    data = json.loads(manifest.read_text())
    data["files"] = [
        row for row in data["files"] if row["destination"] != f"investors/{VC}.toml"
    ]
    manifest.write_text(json.dumps(data))

    imported = import_assets(source, tmp_path / "destination", VC, EPISODE)

    verify_package(imported, VC, EPISODE)


def test_importer_copies_complete_manifest_bound_precedent_corpus(
    tmp_path: Path,
) -> None:
    source = make_package(tmp_path / "source")
    corpus = add_precedent_corpus(source)
    bind_corpus(source, corpus)

    imported = import_assets(source, tmp_path / "destination", VC, EPISODE)
    verified = verify_package(imported, VC, EPISODE)

    assert verified.precedents is not None
    assert {
        path.relative_to(imported).as_posix()
        for path in verified.precedents.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(source).as_posix()
        for path in corpus.rglob("*")
        if path.is_file()
    }


def test_importer_copies_complete_manifest_bound_portfolio_memory(tmp_path: Path) -> None:
    source = make_package(tmp_path / "source")
    portfolio = add_portfolio_memory(source)
    bind_corpus(source, portfolio)
    imported_root = import_assets(source, tmp_path / "destination", VC, EPISODE)
    imported = imported_root / f"data/investors/{VC}/portfolio-memory"
    assert {path.name for path in imported.iterdir()} == {
        "company-index.json", "corpus-manifest.json", "disclosure-events.jsonl",
        "embedding-index.json", "extraction-audit.jsonl",
    }
    assert verify_package(imported_root, VC, EPISODE).portfolio_memory == imported


def test_importer_rejects_partially_manifest_bound_precedent_corpus(
    tmp_path: Path,
) -> None:
    source = make_package(tmp_path / "source")
    corpus = add_precedent_corpus(source)
    bind_corpus(source, corpus)
    manifest = source / f"data/investors/{VC}/source-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["files"] = [
        row
        for row in data["files"]
        if not row["destination"].endswith("sources/18-rowvigor.json")
    ]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(InputBoundaryError, match="manifest-bound"):
        import_assets(source, tmp_path / "destination", VC, EPISODE)


def test_importer_rejects_symlinked_source_ancestor_before_destination_creation(
    tmp_path: Path,
) -> None:
    source = make_package(tmp_path / "source")
    investor = source / f"data/investors/{VC}"
    outside = tmp_path / "outside-investor"
    investor.rename(outside)
    investor.symlink_to(outside, target_is_directory=True)
    destination = tmp_path / "destination"

    with pytest.raises(InputBoundaryError, match="symlink|escape"):
        import_assets(source, destination, VC, EPISODE)

    assert not destination.exists()


def test_importer_rejects_source_mutation_after_preflight_without_output(
    tmp_path: Path, monkeypatch
) -> None:
    source = make_package(tmp_path / "source")
    destination = tmp_path / "destination"
    original = importer_module._before_copy_selected

    def mutate(relative: str, path: Path) -> None:
        if relative.endswith(f"pitches/{EPISODE}.txt"):
            path.write_text("Founder: mutated after preflight.\n", encoding="utf-8")
        original(relative, path)

    monkeypatch.setattr(importer_module, "_before_copy_selected", mutate)

    with pytest.raises(InputBoundaryError, match="source hash mismatch"):
        import_assets(source, destination, VC, EPISODE)

    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.staging-*"))


def test_importer_rejects_ancestor_symlink_swap_after_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    source = make_package(tmp_path / "source")
    destination = tmp_path / "destination"
    investors = source / "investors"
    outside = tmp_path / "outside-investors"
    registry_bytes = (investors / f"{VC}.toml").read_bytes()
    swapped = False

    def swap_ancestor(relative: str, _path: Path) -> None:
        nonlocal swapped
        if relative == f"investors/{VC}.toml" and not swapped:
            investors.rename(outside)
            investors.symlink_to(outside, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(importer_module, "_before_copy_selected", swap_ancestor)

    with pytest.raises(InputBoundaryError, match="descriptor|symlink|traversal"):
        import_assets(source, destination, VC, EPISODE)

    assert swapped
    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.staging-*"))
    assert (outside / f"{VC}.toml").read_bytes() == registry_bytes


def test_importer_copy_failure_leaves_no_destination_or_staging(
    tmp_path: Path, monkeypatch
) -> None:
    source = make_package(tmp_path / "source")
    destination = tmp_path / "destination"
    original = importer_module._copy_selected_file
    calls = 0

    def fail_mid_copy(
        source_root_fd: int,
        relative: str,
        target: Path,
        expected: str | None,
    ) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        return original(source_root_fd, relative, target, expected)

    monkeypatch.setattr(importer_module, "_copy_selected_file", fail_mid_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        import_assets(source, destination, VC, EPISODE)

    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.staging-*"))


def test_importer_unapproved_rebound_audit_leaves_no_destination_or_staging(
    tmp_path: Path,
) -> None:
    source = make_package(tmp_path / "source")
    audit = source / f"data/investors/{VC}/audits/{EPISODE}.json"
    data = json.loads(audit.read_text())
    data["status"] = "pending"
    audit.write_text(json.dumps(data), encoding="utf-8")
    rebind_manifest_file(source, audit)
    destination = tmp_path / "destination"

    with pytest.raises(InputBoundaryError, match="approved"):
        import_assets(source, destination, VC, EPISODE)

    assert not destination.exists()
    assert not list(tmp_path.glob(".destination.staging-*"))


def test_manifest_builder_hash_binds_shared_and_episode_assets(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    (root / f"data/investors/{VC}/source-manifest.json").unlink()

    manifest = build_episode_manifest(root, VC, EPISODE)

    verified = verify_package(root, VC, EPISODE)
    rows = json.loads(manifest.read_text())["files"]
    destinations = {row["destination"] for row in rows}
    assert verified.manifest == manifest
    assert f"wiki/{VC}/persona.md" in destinations
    assert f"data/investors/{VC}/pitches/{EPISODE}.txt" in destinations


def test_manifest_builder_hash_binds_every_precedent_file(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    corpus = add_precedent_corpus(root)
    (root / f"data/investors/{VC}/source-manifest.json").unlink()

    manifest = build_episode_manifest(root, VC, EPISODE)

    destinations = {
        row["destination"] for row in json.loads(manifest.read_text())["files"]
    }
    corpus_destinations = {
        path.relative_to(root).as_posix()
        for path in corpus.rglob("*")
        if path.is_file()
    }
    assert corpus_destinations <= destinations
    assert verify_package(root, VC, EPISODE).precedents == corpus


def test_manifest_builder_hash_binds_every_portfolio_memory_file(tmp_path: Path) -> None:
    root = make_package(tmp_path)
    portfolio = add_portfolio_memory(root)
    (root / f"data/investors/{VC}/source-manifest.json").unlink()
    manifest = build_episode_manifest(root, VC, EPISODE)
    destinations = {
        row["destination"] for row in json.loads(manifest.read_text())["files"]
    }
    assert {path.relative_to(root).as_posix() for path in portfolio.iterdir()} <= destinations
    assert verify_package(root, VC, EPISODE).portfolio_memory == portfolio


@pytest.mark.parametrize(
    ("vc_slug", "episode_slug"),
    [
        ("../outside", EPISODE),
        ("/tmp/outside", EPISODE),
        (VC, "../outside"),
        (VC, "/tmp/outside"),
    ],
)
def test_manifest_builder_rejects_unsafe_slugs(
    tmp_path: Path, vc_slug: str, episode_slug: str
) -> None:
    root = make_package(tmp_path)

    with pytest.raises(InputBoundaryError, match="slug"):
        build_episode_manifest(root, vc_slug, episode_slug)


def test_manifest_builder_rejects_symlinked_wiki_without_external_read(
    tmp_path: Path,
) -> None:
    root = make_package(tmp_path)
    wiki = root / f"wiki/{VC}"
    outside = tmp_path / "outside-wiki"
    wiki.rename(outside)
    wiki.symlink_to(outside, target_is_directory=True)

    with pytest.raises(InputBoundaryError, match="symlink|escape"):
        build_episode_manifest(root, VC, EPISODE)

    assert not (root / f"data/investors/{VC}/manifests/{EPISODE}.json").exists()


def test_manifest_builder_rejects_symlinked_root(tmp_path: Path) -> None:
    actual = make_package(tmp_path / "actual")
    linked = tmp_path / "linked-inputs"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(InputBoundaryError, match="symlink"):
        build_episode_manifest(linked, VC, EPISODE)

    assert not (actual / f"data/investors/{VC}/manifests/{EPISODE}.json").exists()


def test_manifest_builder_rejects_symlinked_output_parent_without_external_write(
    tmp_path: Path,
) -> None:
    root = make_package(tmp_path)
    manifests = root / f"data/investors/{VC}/manifests"
    outside = tmp_path / "outside-manifests"
    outside.mkdir()
    manifests.symlink_to(outside, target_is_directory=True)

    with pytest.raises(InputBoundaryError, match="symlink|escape"):
        build_episode_manifest(root, VC, EPISODE)

    assert not (outside / f"{EPISODE}.json").exists()
