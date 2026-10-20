import importlib.util
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location('pitchshow_evaluation_sources', Path(__file__).parents[1] / 'scripts/pitchshow_evaluation_sources.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shared_cache_fetches_once_and_survives_restart(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text='original transcript')
    cache = module.SharedDownloads(tmp_path, transport=httpx.MockTransport(handler))
    url = 'https://www.thepitch.show/episode'
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(cache, [url] * 8))
    assert [r.text for r in results] == ['original transcript'] * 8
    assert calls == [url]
    cache.client.close()
    cache = module.SharedDownloads(tmp_path, transport=httpx.MockTransport(handler))
    assert cache(url).text == 'original transcript'
    assert calls == [url]
    cache.client.close()


@pytest.mark.parametrize('url', ['https://evil.example/page', 'http://www.thepitch.show/page', 'https://www.thepitch.show:444/page', 'https://user@www.thepitch.show/page'])
def test_official_host_validation(tmp_path, url):
    cache = module.SharedDownloads(tmp_path)
    with pytest.raises(ValueError):
        cache(url)
    cache.client.close()


def test_redirect_checked_before_external_request(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={'location': 'https://evil.example/private'})
    cache = module.SharedDownloads(tmp_path, transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match='official'):
        cache('https://www.thepitch.show/episode')
    assert calls == ['https://www.thepitch.show/episode']
    cache.client.close()


def test_download_size_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'MAX_BYTES', 3)
    cache = module.SharedDownloads(tmp_path, transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b'large')))
    with pytest.raises(ValueError, match='size limit'):
        cache('https://www.thepitch.show/episode')
    cache.client.close()


def test_fallback_is_explicit_and_hash_verified(tmp_path):
    old = tmp_path / 'old.html'
    old.write_bytes(b'transcript')
    url = 'https://www.thepitch.show/episode'
    cache = module.SharedDownloads(tmp_path / 'cache', fallback={url: (old, hashlib.sha256(old.read_bytes()).hexdigest())}, transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    assert cache(url).text == 'transcript'
    assert cache.receipts[url]['provenance'] == 'inherited_stale'
    assert '503' in cache.receipts[url]['fresh_download_error']
    cache.client.close()


def test_failed_request_is_not_retried_per_investor(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(503)
    cache = module.SharedDownloads(tmp_path, transport=httpx.MockTransport(handler))
    for _ in range(3):
        with pytest.raises((httpx.HTTPStatusError, ValueError)):
            cache('https://www.thepitch.show/missing')
    assert len(calls) == 1
    cache.client.close()


def test_cache_tampering_is_rejected(tmp_path):
    cache = module.SharedDownloads(tmp_path, transport=httpx.MockTransport(lambda request: httpx.Response(200, text='original')))
    url = 'https://www.thepitch.show/episode'
    cache(url)
    next(tmp_path.glob('*.txt')).write_text('modified')
    with pytest.raises(ValueError, match='verification'):
        cache(url)
    cache.client.close()
