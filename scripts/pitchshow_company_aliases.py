"""Validate source-bound company aliases for historical exclusions."""
from hashlib import sha256
from pathlib import Path
from prepare_pitchshow_evaluation import read

def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def normalized_name(value):
    return ''.join(c for c in value.casefold() if c.isalnum())


def validate_aliases(root, mapping, required):
    result = {}
    for slug in sorted(required):
        row = mapping.get('episodes', {}).get(slug)
        if not row:
            raise ValueError(f'missing company aliases: {slug}')
        aliases = row['aliases']
        if (not 1 <= len(aliases) <= 8 or any(not isinstance(a, str) or not a.strip()
            or len(a) > 120 or not a.isprintable() for a in aliases)
            or len({a.casefold() for a in aliases}) != len(aliases)):
            raise ValueError(f'invalid company aliases: {slug}')
        source = (root / row['source_path']).resolve()
        if not source.is_relative_to(root.resolve()) or digest(source) != row['source_sha256']:
            raise ValueError(f'changed alias source: {slug}')
        episode = read(source)
        if episode['slug'] != slug:
            raise ValueError('alias source episode mismatch')
        evidence = row['evidence']
        if len(evidence) != len(aliases) or {e['alias'] for e in evidence} != set(aliases):
            raise ValueError(f'incomplete alias evidence: {slug}')
        for item in evidence:
            if (item['field'] not in {'transcript', 'name', 'description'}
                or not item['quote'] or item['quote'] not in episode.get(item['field'], '')
                or not item['matched_name'] or item['matched_name'] not in item['quote']
                or normalized_name(item['matched_name']) != normalized_name(item['alias'])):
                raise ValueError(f'unsupported company alias: {slug}/{item["alias"]}')
        result[slug] = aliases
    return result
