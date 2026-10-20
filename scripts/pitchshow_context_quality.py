"""Conservative, offline checks for answers whose question was filtered out."""
from __future__ import annotations

import re

# Only neutral topic labels can enter a restored pitch, never question wording,
# quantities, units, investor names, or investment stance.
TOPICS = {
    'burn': r'\bburn(?: rate)?\b',
    'revenue': r'\b(?:revenue|arr|mrr)\b',
    'margin': r'\bmargin\b',
    'valuation': r'\bvaluation\b',
    'customers': r'\bcustomers?\b',
    'fundraising': r'\b(?:raising|raise|fundraising)\b',
    'runway': r'\brunway\b',
    'team': r'\b(?:team|employees|headcount)\b',
}


def audit_context(receipt, kept, aliases):
    """Flag missing preceding source context; restore only reviewed safe topics.

    shortcut: immediate source adjacency is conservative; manual review is needed
    for distant questions, ambiguous referents, and removed unreviewed questions.
    """
    retained = {s['source_turn_index'] for s in kept}
    target = {a.strip().casefold() for a in aliases}
    rendered = [s for s in receipt['segments'] if s['rendered_text']]
    removed = {id(rendered[i]) for i in receipt['removed_candidate_lines']}
    findings = []
    previous = None
    for segment in receipt['segments']:
        if (segment['source_turn_index'] in retained
                and segment['retained_content_class'] == 'explicit_founder'
                and previous is not None
                and previous['source_turn_index'] not in retained
                and previous['speaker'] != segment['speaker']):
            safe = (previous.get('preceded_by_investor_decision') is False
                    and segment.get('preceded_by_investor_decision') is False
                    and previous['retained_content_class'] == 'explicit_investor_question'
                    and id(previous) not in removed
                    and previous['source_turn_index'] + 1 == segment['source_turn_index'])
            topics = [label for label, pattern in TOPICS.items()
                      if safe and re.search(pattern, previous['rendered_text'], re.I)]
            # Multiple topics would imply a relationship that the short answer
            # may not establish. Leave that ambiguity for review.
            topic = topics[0] if len(topics) == 1 else None
            findings.append({
                'answer_source_turn_index': segment['source_turn_index'],
                'context_source_turn_index': previous['source_turn_index'],
                'context_source_line_sha256': previous.get('attributed_source_line_sha256'),
                'context_from_target': previous['speaker'].strip().casefold() in target,
                'classification': 'orphaned_answer_candidate',
                'restorable_topic': topic,
                'status': 'safe_topic_available' if topic else 'manual_review_required',
            })
        previous = segment
    return {'schema': 'pitchshow-context-quality-v1', 'human_reviewed': False,
            'check_scope': 'immediate preceding source turn; heuristic candidates, not proven errors',
            'findings': findings, 'orphaned_answer_candidates': len(findings),
            'safe_topic_count': sum(f['restorable_topic'] is not None for f in findings)}


def restore_context(receipt, kept, aliases):
    audit = audit_context(receipt, kept, aliases)
    topics = {f['answer_source_turn_index']: f['restorable_topic']
              for f in audit['findings'] if f['restorable_topic']}
    lines = []
    for segment in kept:
        if segment['source_turn_index'] in topics:
            lines.append('[Neutral question topic: ' + topics[segment['source_turn_index']] + ']')
        lines.append(segment['rendered_text'])
    return '\n'.join(lines) + '\n', audit
