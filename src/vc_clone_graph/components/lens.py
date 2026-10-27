"""Investor lens: turn the generic Phase 1 + coverage view into one investor's view.

Phase 1 investigates the pitch, so twins of different investors end up with
nearly the same rationales. The lens reads the saved rationales and asks only
which of them this investor would raise, in which direction and how strongly,
and which investor-specific reasons are missing. Variants add inputs:

- memory: the run's own target-redacted Investment Memory;
- precedents: plus the investor's own remarks in held-out past episodes
  (target turns only, founders' direct replies kept) and their decisions;
- facts: plus the founders' in-room fact sheet for this episode;
- profile: precedents plus the investor's distinctive rationales from the
  annotations of their other episodes;
- contrastive: profile plus the other panelists' distinctive rationales, asking
  what this investor would raise that the others would not;
- fit: precedents plus an explicit investor-fit check.

The output uses the rationale shape the tally already scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from ..firewall import verify_package
from ..precedents import PrecedentCorpus
from ..providers.base import GenerationProvider, GenerationRequest
from ..retrieval import HybridWikiIndex
from .config import LensSettings
from .profile import build_profile
from .coverage import DIRECTIONS, SALIENCES, _normalize
from .run_inputs import RunArtifacts

SCHEMA_VERSION = "investor-lens-v1"
FIT_ASPECTS = ("stage", "check_size", "sector", "geography", "thesis", "value_add", "portfolio_conflict")
FIT_VERDICTS = ("fits", "concern", "unknown")
PRECEDENT_VARIANTS = frozenset({"precedents", "facts", "profile", "contrastive", "fit"})
ALIGN_VARIANTS = frozenset({"unique-wiki", "unique-record"})
MEMORY_SOURCES = ("persona.md", "theses.md", "portfolio_and_constraints.md")
_EXCLUDED_TEXT = "[Source section excluded for held-out evaluation.]"


def _run_package(run: RunArtifacts, workspace: Path):
    config = json.loads((run.run_dir / "run-config.json").read_text(encoding="utf-8"))["run"]
    root = Path(config["input_root"])
    root = root if root.is_absolute() else workspace / root
    package = verify_package(root, run.vc_slug, run.episode_slug, taxonomy_path=config["taxonomy_path"])
    return root, package


def _memory_index(run: RunArtifacts, root: Path, package) -> tuple[HybridWikiIndex, str]:
    """The run's saved wiki index, or a rebuild from the run's own wiki files.

    Run inputs are hard-linked, so a later in-place install of an investor's
    index can leave a run's saved index out of step with its own wiki files.
    The lens needs no embeddings, so the sections are rebuilt from the wiki the
    run read, and accepted only if every excerpt the run selected is recovered.
    """
    try:
        return HybridWikiIndex.load(root / "indexes" / f"{run.vc_slug}.json", package.wiki, None), "saved_index"
    except ValueError as exc:
        if "hash changed" not in str(exc) and "text changed" not in str(exc):
            raise
    rebuilt = HybridWikiIndex.build(package.wiki, None)
    ids = {chunk.chunk_id for chunk in rebuilt.chunks}
    missing = [e.get("evidence_id") for e in run.policy_evidence if e.get("evidence_id") not in ids]
    if missing or not run.policy_evidence:
        raise ValueError(f"rebuilt memory does not contain the run's evidence: {missing or 'no evidence saved'}")
    return rebuilt, "rebuilt_from_run_wiki"


def load_memory(
    run: RunArtifacts, settings: LensSettings, workspace: Path = Path(".")
) -> tuple[list[dict[str, Any]], str]:
    """The run's Investment Memory with the pitch company redacted, as the run saw it."""
    root, package = _run_package(run, workspace)
    canonical, source = _memory_index(run, root, package)
    index, _ = canonical.for_pitch(package.target_company_aliases)

    def rank(chunk) -> tuple[int, str]:
        path = chunk.source_path
        return (MEMORY_SOURCES.index(path) if path in MEMORY_SOURCES else len(MEMORY_SOURCES), path)

    chosen, used = [], 0
    for chunk in sorted(index.chunks, key=rank):
        path = chunk.source_path
        if not (path in MEMORY_SOURCES or path.startswith("evidence/")) or chunk.text.strip() == _EXCLUDED_TEXT:
            continue
        if used + len(chunk.text) > settings.max_memory_chars:
            continue
        used += len(chunk.text)
        chosen.append({"memory_id": chunk.chunk_id, "source": path, "heading": chunk.heading, "text": chunk.text})
    return chosen, source


def load_precedents(run: RunArtifacts, settings: LensSettings, workspace: Path = Path(".")) -> list[dict[str, Any]]:
    """The investor's own remarks in the past episodes most similar to this pitch."""
    root, package = _run_package(run, workspace)
    if package.precedents is None:
        return []
    canonical = PrecedentCorpus.load(root / "indexes" / f"{run.vc_slug}.precedents.json", package.precedents, None)
    corpus = canonical.for_target(run.episode_slug, excluded_aliases=package.target_company_aliases)
    if any(row.episode_slug == run.episode_slug for row in corpus.list_episodes()):
        raise ValueError("target episode remains accessible after precedent filtering")
    corpus.restrict_to_target_turns()
    slugs: list[str] = []
    for hit in corpus.search(run.pitch_text, top_k=40, observed_only=True):
        if hit.episode_slug not in slugs:
            slugs.append(hit.episode_slug)
        if len(slugs) == settings.precedent_episodes:
            break
    rows = []
    for number, slug in enumerate(slugs, start=1):
        read = corpus.open_transcript(slug)
        decision = corpus.open_decision(slug)
        text = "\n".join(f"{t.speaker}: {t.text}" for t in read.turns)
        rows.append({
            "precedent_id": f"E{number}",
            "episode_slug": slug,
            "decision": decision.status,
            "decision_statement": " ".join(span.text for span in decision.evidence)[:600],
            "remarks": text[: settings.max_precedent_chars],
        })
    return rows


def candidates(run: RunArtifacts, coverage: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = [
        {**r, "candidate_id": r["rationale_id"], "origin": "phase1"}
        for r in run.phase1.get("rationales", [])
    ]
    for item in (coverage or {}).get("additions", []):
        rows.append({**item, "candidate_id": item["addition_id"], "origin": "coverage"})
    return rows


def build_prompt(
    run: RunArtifacts,
    settings: LensSettings,
    rows: Sequence[dict[str, Any]],
    memory: Sequence[dict[str, Any]],
    precedents: Sequence[dict[str, Any]] = (),
    facts: Sequence[dict[str, Any]] = (),
    profile: Sequence[dict[str, Any]] = (),
    panel: Sequence[dict[str, Any]] = (),
) -> str:
    parents = {r["label"]: r.get("coarse_parent", "") for r in run.taxonomy}
    candidate_lines = "\n".join(
        f"- {r['candidate_id']} {r['taxonomy_label']} [{parents.get(r['taxonomy_label'], '')}] "
        f"{r['direction']}, {r['salience']}: {str(r.get('justification', ''))[:240]}"
        for r in rows
    ) or "- none"
    label_lines = "\n".join(
        f"- {r['label']} [{r.get('coarse_parent', '')}]: {str(r.get('definition', ''))[:200]}"
        for r in run.taxonomy
    )
    memory_lines = "\n\n".join(
        f"[{m['memory_id']}] {m['heading']} ({m['source']})\n{m['text']}" for m in memory
    ) or "No Investment Memory is available."
    evidence_rule = "pitch_quote (one contiguous excerpt of at most 40 words, copied exactly from the pitch)"
    sections = []
    if precedents:
        evidence_rule += " and/or precedent_ids"
        sections.append("This investor's own remarks in past episodes of the show (other investors' remarks are removed). "
                        "Use them to judge what this investor tends to raise and how they decide:\n\n" + "\n\n".join(
            f"[{p['precedent_id']}] Past decision: {p['decision']}. Their decision statement: {p['decision_statement']}\n"
            f"{p['remarks']}" for p in precedents))
    if profile:
        sections.append(
            f"What sets this investor apart in practice: rationales they raised in their other {profile[0]['episodes']} "
            "episodes of the show much more often than investors in general (share of their episodes vs all investors' "
            "episodes, usual direction, examples). Use this to decide which reasons THIS investor, rather than any "
            "investor, would raise on this pitch:\n" + "\n".join(
                f"[{v['profile_id']}] {v['taxonomy_label']}: {v['investor_rate']:.0%} vs {v['all_investors_rate']:.0%}, "
                f"usually {v['usual_direction']}. " + " | ".join(v["examples"]) for v in profile))
    if panel:
        sections.append(
            "The other investors on this panel hear the same pitch. Their distinctive rationales, from the same "
            "kind of record:\n" + "\n".join(
                f"- {m['investor_name']}: " + ("; ".join(
                    f"{v['taxonomy_label']} ({v['investor_rate']:.0%} vs {v['all_investors_rate']:.0%}, usually "
                    f"{v['usual_direction']})" for v in m["profile"]) or "no record")
                for m in panel))
    if facts:
        evidence_rule += ", or fact_ids"
        sections.append("Further facts the founders stated when answering questions in this episode:\n" + "\n".join(
            f"[{f['fact_id']}] ({f['category']}) {f['fact']}" for f in facts))
    extra = "\n\n".join(sections)
    selective = (
        f"Be selective: keep plus add at most {settings.max_kept} reasons in total, the ones this investor "
        "would actually voice about this pitch, and drop all others. Real investors give a handful of reasons, "
        "and different investors on the same pitch often give different ones."
        if settings.max_kept else ""
    )
    tone = (
        "" if settings.variant == "contrastive" else
        " Generic analysts are too positive; investors are often negative or neutral where the pitch sounds good."
    )
    contrast = (
        "Contrast: the other investors on this panel will raise the reasons any investor would raise. Keep "
        "and add what THIS investor would raise or weigh differently from them, given their record, and drop "
        "reasons this investor would not voice even if the others would."
        if panel else ""
    )
    fit = (
        "fit_check: before the additions, judge this investor's fit with the pitch on each aspect (stage, "
        "check size, sector, geography, thesis, whether they can add value, portfolio conflicts) from the "
        "Investment Memory and the pitch: fits, concern, or unknown, with memory_ids. Every concern must appear "
        "as a kept or added reason in the investor_fit_constraints dimension."
        if settings.variant == "fit" else ""
    )
    steps = "".join(f"\n{n}. {text}" for n, text in enumerate((t for t in (selective, contrast, fit) if t), start=4))
    return f"""You are turning a general analysis of a startup pitch into the view of one specific investor.

Investor: {run.investor_name}

A general analyst, who looked at the pitch from every angle, produced these candidate rationales (id, taxonomy label [dimension], direction, salience: justification):
{candidate_lines}

Task: decide which of these reasons THIS investor would actually raise and weigh when deciding on this pitch, and how.

1. investor_focus: first, in 2-4 sentences, state what this investor would focus on for this pitch and why, based on the evidence below about them.
2. reviews: one entry for every candidate id.
   - action "drop" when this investor would not raise it or would not care, even if it is true of the pitch. Investors decide on a few reasons; keep only those this investor would plausibly voice.
   - direction from this investor's own standards: positive supports investing, negative weighs against, neutral is a material open question.{tone}
   - salience primary only for this investor's likely deciding reasons.
3. additions: at most {settings.max_additions} reasons this investor would raise that the analyst missed. Check investor fit and constraints in particular (stage, check size, sector, geography, thesis, whether they can add value, portfolio conflicts). Each addition needs {"memory_ids or profile_ids" if profile else "memory_ids"} and {evidence_rule}. An empty list is correct when nothing is missing.{steps}

Taxonomy labels for additions:
{label_lines}

Investment Memory (public material about this investor):
{memory_lines}

{extra}

Pitch (founder statements only):
<pitch>
{run.pitch_text}
</pitch>"""


def build_align_prompt(
    run: RunArtifacts,
    rows: Sequence[dict[str, Any]],
    page: Sequence[dict[str, Any]],
    record: Sequence[dict[str, Any]] = (),
) -> str:
    """Alignment agent: the decision agent's view, reviewed against what sets this investor apart."""
    parents = {r["label"]: r.get("coarse_parent", "") for r in run.taxonomy}
    candidate_lines = "\n".join(
        f"- {r['candidate_id']} {r['taxonomy_label']} [{parents.get(r['taxonomy_label'], '')}] "
        f"{r['direction']}, {r['salience']}: {str(r.get('justification', ''))[:200]}"
        for r in rows
    ) or "- none"
    label_lines = "\n".join(f"- {r['label']} [{r.get('coarse_parent', '')}]" for r in run.taxonomy)
    p2 = run.phase2 or {}
    opposing = (p2.get("strongest_opposing_case") or {}).get("argument", "")
    page_lines = "\n".join(f"[{i['page_id']}] ({i['kind'].replace('_', ' ')}) {i['statement']}" for i in page) or "- none"
    record_text = (
        f"\n\nWhat this investor actually raised on the show in their other {record[0]['episodes']} episodes, more "
        "often than investors in general (share of their episodes vs all investors' episodes, usual direction):\n" +
        "\n".join(f"[{v['profile_id']}] {v['taxonomy_label']}: {v['investor_rate']:.0%} vs {v['all_investors_rate']:.0%}, "
                   f"usually {v['usual_direction']}" for v in record)
        if record else ""
    )
    return f"""You are aligning an assessment of a startup pitch with one specific investor.

Investor: {run.investor_name}

A decision agent assessed the pitch as this investor:
- decision: {p2.get('decision')}, likelihood of In: {p2.get('investment_likelihood')}
- controlling rationales: {', '.join(p2.get('controlling_rationale_ids') or []) or 'none'}
- justification: {str(p2.get('decision_justification', ''))[:900]}
- strongest opposing case: {str(opposing)[:500]}

Candidate rationales from the analysis (id, taxonomy label [dimension], direction, salience: justification):
{candidate_lines}

What sets this investor apart from other investors (from their public statements):
{page_lines}{record_text}

Task: bring this assessment closer to how THIS investor, rather than a typical investor, would judge this pitch.
1. investor_focus: 2-4 sentences on what this investor would focus on for this pitch, given what sets them apart.
2. reviews: one entry for every candidate id. action "drop" when this investor would not raise it or would not care; direction from this investor's own standards (positive supports investing, negative weighs against, neutral is a material open question); salience primary only for their likely deciding reasons.
3. additions: at most 6 reasons this investor would raise that the analysis missed, especially investor fit and their hard limits. Each needs profile_ids (page or record ids) and pitch_quote (one contiguous excerpt of at most 40 words, copied exactly from the pitch). An empty list is correct when nothing is missing.
4. likelihood (0 to 1, the probability this investor says In: 0.1 = very likely Out, 0.9 = very likely In) and decision (In or Out, consistent with the likelihood). In means they commit on air to invest or ask to continue talks. Change the decision agent's view only where what sets this investor apart gives a reason to.

Taxonomy labels for additions:
{label_lines}

Pitch (founder statements only):
<pitch>
{run.pitch_text}
</pitch>"""


def output_schema(rows: Sequence[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "investor_focus": {"type": "string"},
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string", "enum": [r["candidate_id"] for r in rows] or ["none"]},
                        "action": {"type": "string", "enum": ["keep", "drop"]},
                        "direction": {"type": "string", "enum": list(DIRECTIONS)},
                        "salience": {"type": "string", "enum": list(SALIENCES)},
                        "reason": {"type": "string"},
                    },
                },
            },
            "fit_check": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "aspect": {"type": "string", "enum": list(FIT_ASPECTS)},
                        "verdict": {"type": "string", "enum": list(FIT_VERDICTS)},
                        "memory_ids": {"type": "array", "items": {"type": "string"}},
                        "note": {"type": "string"},
                    },
                },
            },
            "additions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "taxonomy_label": {"type": "string", "enum": list(labels)},
                        "direction": {"type": "string", "enum": list(DIRECTIONS)},
                        "salience": {"type": "string", "enum": list(SALIENCES)},
                        "confidence": {"type": "number"},
                        "pitch_quote": {"type": "string"},
                        "fact_ids": {"type": "array", "items": {"type": "string"}},
                        "precedent_ids": {"type": "array", "items": {"type": "string"}},
                        "memory_ids": {"type": "array", "items": {"type": "string"}},
                        "profile_ids": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def apply_lens(
    raw: dict[str, Any] | None,
    run: RunArtifacts,
    rows: Sequence[dict[str, Any]],
    settings: LensSettings,
    *,
    memory_ids: set[str],
    precedent_ids: set[str] = frozenset(),
    fact_ids: set[str] = frozenset(),
    profile_ids: set[str] = frozenset(),
) -> dict[str, Any]:
    """Apply reviews and grounded additions; unreviewed candidates are kept unchanged."""
    if not isinstance(raw, dict) or not isinstance(raw.get("reviews"), list):
        raise ValueError("lens output is not a valid structured object")
    by_id = {r["candidate_id"]: r for r in rows}
    reviews: dict[str, dict[str, Any]] = {}
    for item in raw["reviews"]:
        if item.get("candidate_id") in by_id and item["candidate_id"] not in reviews:
            reviews[item["candidate_id"]] = item
    rationales, dropped, unreviewed = [], [], []
    for row in rows:
        review = reviews.get(row["candidate_id"])
        if review is None:
            unreviewed.append(row["candidate_id"])
            review = {"action": "keep"}
        if review.get("action") == "drop":
            dropped.append({"candidate_id": row["candidate_id"], "taxonomy_label": row["taxonomy_label"],
                            "reason": review.get("reason", "")})
            continue
        direction = review.get("direction") if review.get("direction") in DIRECTIONS else row["direction"]
        salience = review.get("salience") if review.get("salience") in SALIENCES else row["salience"]
        rationales.append({
            "rationale_id": row["candidate_id"],
            "origin": row["origin"],
            "taxonomy_label": row["taxonomy_label"],
            "direction": direction,
            "salience": salience,
            "confidence": float(row.get("confidence", 0.5)),
            "changed": {k: [row[k], v] for k, v in (("direction", direction), ("salience", salience)) if row[k] != v},
            "reason": review.get("reason", ""),
        })
    pitch = _normalize(run.pitch_text)
    labels = {r["label"] for r in run.taxonomy}
    present = {r["taxonomy_label"] for r in rationales}
    added, rejected = [], []
    for item in raw.get("additions") or []:
        label = item.get("taxonomy_label")
        quote = _normalize(str(item.get("pitch_quote", "")))
        grounded = (
            (len(quote) >= 8 and quote in pitch)
            or any(i in precedent_ids for i in item.get("precedent_ids") or [])
            or any(i in fact_ids for i in item.get("fact_ids") or [])
        )
        reason = None
        if label not in labels:
            reason = "unknown_label"
        elif label in present:
            reason = "duplicate_label"
        elif item.get("direction") not in DIRECTIONS or item.get("salience") not in SALIENCES:
            reason = "invalid_direction_or_salience"
        elif not any(i in memory_ids for i in item.get("memory_ids") or []) and not any(
            i in profile_ids for i in item.get("profile_ids") or []
        ):
            reason = "no_valid_memory_id"
        elif not grounded:
            reason = "evidence_not_found"
        elif len(added) >= settings.max_additions:
            reason = "over_max_additions"
        if reason:
            rejected.append({**item, "rejection_reason": reason})
            continue
        present.add(label)
        added.append({
            "rationale_id": f"L{len(added) + 1}",
            "origin": "lens",
            "taxonomy_label": label,
            "direction": item["direction"],
            "salience": item["salience"],
            "confidence": min(max(float(item.get("confidence", 0.5)), 0.0), 1.0),
            "pitch_quote": item.get("pitch_quote", ""),
            "memory_ids": [i for i in item.get("memory_ids") or [] if i in memory_ids],
            "precedent_ids": [i for i in item.get("precedent_ids") or [] if i in precedent_ids],
            "fact_ids": [i for i in item.get("fact_ids") or [] if i in fact_ids],
            "profile_ids": [i for i in item.get("profile_ids") or [] if i in profile_ids],
            "reason": item.get("reason", ""),
        })
    final = rationales + added
    if settings.max_kept and len(final) > settings.max_kept:
        ranked = sorted(final, key=lambda r: (r["salience"] != "primary", -r["confidence"]))
        keep = {id(r) for r in ranked[: settings.max_kept]}
        dropped += [{"candidate_id": r["rationale_id"], "taxonomy_label": r["taxonomy_label"],
                     "reason": "over_max_kept"} for r in final if id(r) not in keep]
        final = [r for r in final if id(r) in keep]
    return {
        "investor_focus": raw.get("investor_focus", ""),
        "fit_check": [i for i in raw.get("fit_check") or [] if isinstance(i, dict)],
        "rationales": final,
        "dropped": dropped,
        "unreviewed_candidate_ids": unreviewed,
        "rejected_additions": rejected,
    }


def run_lens(
    run: RunArtifacts,
    settings: LensSettings,
    provider: GenerationProvider,
    *,
    coverage: dict[str, Any] | None,
    facts: dict[str, Any] | None = None,
    panel: Sequence[tuple[str, str]] = (),
    workspace: Path = Path("."),
) -> dict[str, Any]:
    """``panel`` lists (vc_slug, investor_name) of the other panelists, used by the contrastive variant."""
    rows = candidates(run, coverage)
    if settings.variant in ALIGN_VARIANTS:
        return _run_align(run, settings, provider, rows, workspace)
    memory, memory_source = load_memory(run, settings, workspace) if settings.include_memory else ([], "excluded")
    precedents = load_precedents(run, settings, workspace) if settings.variant in PRECEDENT_VARIANTS else []
    profile = (
        build_profile(workspace / settings.annotation_root, run.vc_slug, run.episode_slug)
        if settings.variant in {"profile", "contrastive"} else []
    )
    panel_rows = [
        {"vc_slug": slug, "investor_name": name,
         "profile": build_profile(workspace / settings.annotation_root, slug, run.episode_slug, top=6)}
        for slug, name in panel
    ] if settings.variant == "contrastive" else []
    fact_rows = (facts or {}).get("facts", []) if settings.variant == "facts" else []
    if settings.variant == "facts" and facts is None:
        raise ValueError(f"fact sheet missing for {run.episode_slug}")
    prompt = build_prompt(run, settings, rows, memory, precedents, fact_rows, profile, panel_rows)
    result = provider.generate(GenerationRequest(
        phase="investor_lens",
        prompt=prompt,
        schema=output_schema(rows, [r["label"] for r in run.taxonomy]),
        max_output_tokens=settings.max_output_tokens,
        reasoning_effort=settings.reasoning_effort,
    ))
    applied = apply_lens(
        result.parsed, run, rows, settings,
        memory_ids={m["memory_id"] for m in memory},
        precedent_ids={p["precedent_id"] for p in precedents},
        fact_ids={f["fact_id"] for f in fact_rows},
        profile_ids={v["profile_id"] for v in profile},
    )
    return {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "variant": settings.variant,
        "vc_slug": run.vc_slug,
        "episode_slug": run.episode_slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "input_hashes": run.hashes,
        "coverage_included": coverage is not None,
        "candidate_count": len(rows),
        "memory_source": memory_source,
        "memory_ids": [m["memory_id"] for m in memory],
        "precedent_episodes": [{k: p[k] for k in ("precedent_id", "episode_slug", "decision")} for p in precedents],
        "fact_count": len(fact_rows),
        "profile": [{k: v[k] for k in ("profile_id", "taxonomy_label", "ratio")} for v in profile],
        "panel": [{"vc_slug": m["vc_slug"], "labels": [v["taxonomy_label"] for v in m["profile"]]} for m in panel_rows],
        "prompt_chars": len(prompt),
        **applied,
        "usage": result.usage.model_dump(),
        "provider_metadata": result.raw_metadata,
    }


def _run_align(
    run: RunArtifacts,
    settings: LensSettings,
    provider: GenerationProvider,
    rows: Sequence[dict[str, Any]],
    workspace: Path,
) -> dict[str, Any]:
    from .uniqueness import page_for_pitch

    if run.phase2 is None:
        raise ValueError("the alignment agent needs the decision agent's output")
    _, package = _run_package(run, workspace)
    page_path = workspace / settings.uniqueness_root / run.vc_slug / f"{run.episode_slug}.json"
    stored = json.loads(page_path.read_text(encoding="utf-8"))
    page = page_for_pitch(stored, package.target_company_aliases)
    record = (
        build_profile(workspace / settings.annotation_root, run.vc_slug, run.episode_slug)
        if settings.variant == "unique-record" else []
    )
    prompt = build_align_prompt(run, rows, page, record)
    schema = output_schema(rows, [r["label"] for r in run.taxonomy])
    schema["properties"]["likelihood"] = {"type": "number"}
    schema["properties"]["decision"] = {"type": "string", "enum": ["In", "Out"]}
    result = provider.generate(GenerationRequest(
        phase="investor_alignment",
        prompt=prompt,
        schema=schema,
        max_output_tokens=settings.max_output_tokens,
        reasoning_effort=settings.reasoning_effort,
    ))
    applied = apply_lens(
        result.parsed, run, rows, settings, memory_ids=set(),
        profile_ids={i["page_id"] for i in page} | {v["profile_id"] for v in record},
    )
    raw = result.parsed or {}
    likelihood = raw.get("likelihood")
    if not isinstance(likelihood, (int, float)) or not 0 <= likelihood <= 1 or raw.get("decision") not in ("In", "Out"):
        raise ValueError("alignment output has no valid decision and likelihood")
    return {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "variant": settings.variant,
        "vc_slug": run.vc_slug,
        "episode_slug": run.episode_slug,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "input_hashes": run.hashes,
        "candidate_count": len(rows),
        "page_items": [i["page_id"] for i in page],
        "page_items_removed_for_pitch": len(stored["items"]) - len(page),
        "record": [{k: v[k] for k in ("profile_id", "taxonomy_label", "ratio")} for v in record],
        "decision_agent": {"decision": run.phase2.get("decision"), "likelihood": run.phase2.get("investment_likelihood")},
        "decision": raw["decision"],
        "likelihood": float(likelihood),
        "prompt_chars": len(prompt),
        **applied,
        "usage": result.usage.model_dump(),
        "provider_metadata": result.raw_metadata,
    }
