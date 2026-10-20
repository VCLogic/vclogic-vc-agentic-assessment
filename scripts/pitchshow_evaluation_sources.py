"""Acquire official Pitch Show evidence using the sibling onboarding collector."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
from urllib.parse import urljoin, urlsplit

import httpx

MAX_BYTES = 12 * 1024 * 1024


def official(url):
    p = urlsplit(url)
    if (p.scheme != 'https' or p.hostname not in {'www.thepitch.show', 'thepitch.show'}
            or p.port not in {None, 443} or p.username or p.password or p.fragment):
        raise ValueError(f'Not an official HTTPS URL: {url}')
    return url


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


class SharedDownloads:
    """One bounded request per URL, including failures, shared by every investor."""

    def __init__(self, root, *, fallback=None, transport=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fallback = fallback or {}
        self.client = httpx.Client(timeout=30, follow_redirects=False, transport=transport)
        self.lock = threading.Lock()
        self.locks = {}
        self.receipts = {}
        self.errors = {}

    def __call__(self, url):
        official(url)
        with self.lock:
            lock = self.locks.setdefault(url, threading.Lock())
        with lock:
            if url in self.errors:
                raise ValueError(self.errors[url])
            key = hashlib.sha256(url.encode()).hexdigest()
            metadata = self.root / (key + '.json')
            body_path = self.root / (key + '.txt')
            if metadata.exists():
                receipt = json.loads(metadata.read_text())
                official(receipt['final_url'])
                body = body_path.read_bytes()
                if len(body) > MAX_BYTES or hashlib.sha256(body).hexdigest() != receipt['sha256']:
                    raise ValueError('Cached body failed size/hash verification')
            else:
                try:
                    body, final_url = self.download(url)
                    receipt = {'url': url, 'final_url': final_url, 'provenance': 'fresh_download',
                               'retrieved_at': datetime.now(timezone.utc).isoformat()}
                except Exception as exc:
                    if url not in self.fallback:
                        self.errors[url] = f'{type(exc).__name__}: {exc}'
                        raise
                    source, expected = self.fallback[url]
                    body = source.read_bytes()
                    if len(body) > MAX_BYTES or hashlib.sha256(body).hexdigest() != expected:
                        raise ValueError('Inherited source failed size/hash verification') from exc
                    receipt = {'url': url, 'final_url': url, 'provenance': 'inherited_stale',
                               'inherited_from': str(source), 'fresh_download_error': str(exc)}
                receipt.update(path=body_path.name, sha256=hashlib.sha256(body).hexdigest())
                body_path.write_bytes(body)
                write_json(metadata, receipt)
            self.receipts[url] = receipt
            return httpx.Response(200, content=body, request=httpx.Request('GET', receipt['final_url']))

    def download(self, url):
        for _ in range(6):
            official(url)
            with self.client.stream('GET', url) as response:
                if response.is_redirect:
                    url = official(urljoin(url, response.headers['location']))
                    continue
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError('Response exceeds size limit')
                    chunks.append(chunk)
                return b''.join(chunks), str(response.url)
        raise ValueError('Too many redirects')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onboarding-workspace', type=Path, required=True)
    parser.add_argument('--memory-workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('evaluation/pitchshow'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.onboarding_workspace.resolve() / 'src'))
    from vclogic_onboarding.pitch_show import collect

    bundles, fallback = [], {}
    for path in sorted((args.onboarding_workspace / 'bundles').glob('*-2026-10-17-v1')):
        bundle = json.loads((path / 'bundle.json').read_text())
        if not (args.memory_workspace / 'wiki' / bundle['vc_slug']).is_dir():
            continue
        bundles.append((path, bundle))
        root = path / 'source/pitch-show'
        collection = json.loads((root / 'collection.json').read_text())
        for receipt in collection['sources']:
            url = official(receipt['url'])
            # Only episode HTML is eligible for inherited fallback.
            route = urlsplit(url).path.strip('/')
            if '/' not in route and route != 'sitemap.xml':
                source = (root / receipt['path']).resolve()
                if source.is_relative_to(root.resolve()):
                    fallback.setdefault(url, (source, receipt['sha256']))
    if len(bundles) != 19:
        raise ValueError(f'Expected 19 matching bundles, found {len(bundles)}')
    downloads = SharedDownloads(args.output / 'download-cache', fallback=fallback)

    def acquire(item):
        path, bundle = item
        vc = bundle['vc_slug']
        destination = args.output / 'investors' / vc / 'source'
        if (destination / 'collection.json').exists():
            report = json.loads((destination / 'collection.json').read_text())
        else:
            report = collect(bundle['pitch_show']['investor_slug'], destination, fetch=downloads)
            inherited = []
            for receipt in report['sources']:
                download = downloads.receipts.get(receipt.get('url'))
                if download:
                    receipt['download_provenance'] = download
                    if download['provenance'] == 'inherited_stale':
                        inherited.append(receipt['url'])
            report['inherited_stale_urls'] = inherited
            if inherited:
                report['complete'] = False
                report['coverage_notes'].append('Fresh downloads failed; inherited same-day episode HTML used. Crawl freshness is incomplete.')
            write_json(destination / 'sources/receipts.json', report['sources'])
            write_json(destination / 'collection.json', report)
        print(f"{vc}: {report['episode_count']} episodes, {len(report['failures'])} failures, complete={report['complete']}", flush=True)
        return {'vc_slug': vc, 'bundle': str(path), 'source': str(destination),
                **{k: v for k, v in report.items() if k != 'sources'}}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            reports = list(executor.map(acquire, bundles))
        write_json(args.output / 'collection-summary.json', {
            'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'investor_count': len(reports), 'investors': reports,
            'download_failures': downloads.errors,
            'download_cache': str(downloads.root)})
    finally:
        downloads.client.close()


if __name__ == '__main__':
    main()
