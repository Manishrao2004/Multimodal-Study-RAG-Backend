from types import SimpleNamespace

import pytest

from app.eval import groq_key_proxy as proxy


@pytest.mark.asyncio
async def test_rate_limit_uses_next_key_without_forwarding_errors(monkeypatch):
    seen = []
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, content, headers):
            seen.append(headers["Authorization"])
            if len(seen) == 1:
                return SimpleNamespace(status_code=429, content=b"secret error")
            return SimpleNamespace(status_code=200, content=b'{"choices": []}')
    async def body(): return b'{}'
    monkeypatch.setattr(proxy, "keys", ["first", "second"])
    monkeypatch.setattr(proxy, "next_key", 0)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", Client)
    response = await proxy.chat(SimpleNamespace(body=body))
    assert response.status_code == 200
    assert seen == ["Bearer first", "Bearer second"]
    assert b"secret" not in response.body
