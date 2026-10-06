from __future__ import annotations

import json
import math

import httpx
import pytest
from asgi_lifespan import LifespanManager
from PIL import Image

from immich_ml_adapter.config import Settings
from immich_ml_adapter.main import Runtime, app


def _image() -> bytes:
    from io import BytesIO

    output = BytesIO()
    Image.new("RGB", (17, 11), (20, 80, 140)).save(output, format="PNG")
    return output.getvalue()


def _embedding() -> list[float]:
    value = 1.0 / math.sqrt(768)
    return [value] * 768


@pytest.fixture
async def client():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/embeddinggemma/embeddings":
            assert request.headers["authorization"] == "Bearer test-key"
            body = json.loads(request.content)
            assert body["model"] == "embeddinggemma-2"
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "embeddinggemma-2",
                    "data": [{"object": "embedding", "index": 0, "embedding": _embedding()}],
                },
            )
        if request.url.path == "/predict":
            fields = request.content.decode("latin-1")
            if "entries" not in fields:
                return httpx.Response(400, json={"error": "missing entries"})
            return httpx.Response(
                200, json={"ocr": {"text": ["猫"]}, "imageHeight": 11, "imageWidth": 17}
            )
        if request.url.path == "/ping":
            return httpx.Response(200, text="pong")
        if request.url.path == "/health/readiness":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    async with LifespanManager(app):
        original = app.state.runtime
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app.state.runtime = Runtime(
            settings=Settings(
                litellm_api_key="test-key",
                max_image_bytes=1024 * 1024,
            ),
            client=mock_client,
            owns_client=False,
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as test_client:
                yield test_client, requests
        finally:
            await mock_client.aclose()
            await original.client.aclose()


@pytest.mark.anyio
async def test_ping(client):
    http, _ = client
    response = await http.get("/ping")
    assert response.status_code == 200
    assert response.text == "pong"


@pytest.mark.anyio
async def test_textual_alias_uses_litellm_and_returns_stringified_vector(client):
    http, requests = client
    response = await http.post(
        "/predict",
        data={
            "entries": json.dumps(
                {"clip": {"textual": {"modelName": "ViT-B-16-SigLIP2__webli", "options": {}}}}
            ),
            "text": "猫がソファにいる写真",
        },
    )
    assert response.status_code == 200
    vector = json.loads(response.json()["clip"])
    assert len(vector) == 768
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=0.01)
    assert requests[-1].url.path == "/v1/embeddinggemma/embeddings"
    assert json.loads(requests[-1].content)["input_type"] == "query"


@pytest.mark.anyio
async def test_visual_alias_preserves_image_dimensions(client):
    http, requests = client
    response = await http.post(
        "/predict",
        data={
            "entries": json.dumps({"clip": {"visual": {"modelName": "ViT-B-16-SigLIP2__webli"}}})
        },
        files={"image": ("asset.png", _image(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["imageHeight"] == 11
    assert response.json()["imageWidth"] == 17
    assert len(json.loads(response.json()["clip"])) == 768
    body = json.loads(requests[-1].content)
    assert body["input_type"] == "document"
    assert body["input"][0]["content"][0]["type"] == "image_url"


@pytest.mark.anyio
async def test_ocr_is_proxied_to_stock_ml(client):
    http, requests = client
    response = await http.post(
        "/predict",
        data={"entries": json.dumps({"ocr": {"detection": {"modelName": "PP-OCRv5_mobile"}}})},
        files={"image": ("asset.png", _image(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["ocr"]["text"] == ["猫"]
    assert requests[-1].url.path == "/predict"


@pytest.mark.anyio
async def test_non_alias_clip_is_proxied_to_stock_ml(client):
    http, requests = client
    response = await http.post(
        "/predict",
        data={
            "entries": json.dumps({"clip": {"textual": {"modelName": "ViT-B-32__openai"}}}),
            "text": "猫",
        },
    )
    assert response.status_code == 200
    assert response.json()["ocr"]["text"] == ["猫"]
    assert requests[-1].url.path == "/predict"


@pytest.mark.anyio
async def test_mixed_embedding_and_ocr_entries_are_merged(client):
    http, _ = client
    response = await http.post(
        "/predict",
        data={
            "entries": json.dumps(
                {
                    "clip": {"textual": {"modelName": "ViT-B-16-SigLIP2__webli"}},
                    "ocr": {"detection": {"modelName": "PP-OCRv5_mobile"}},
                }
            ),
            "text": "猫",
        },
    )
    assert response.status_code == 200
    assert len(json.loads(response.json()["clip"])) == 768
    assert response.json()["ocr"]["text"] == ["猫"]
