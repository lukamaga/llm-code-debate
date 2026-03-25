"""
Tests for OllamaClient and MultiModelClient.

All HTTP calls are mocked via httpx mock responses.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm import LLMRequest, LLMResponse
from src.llm.ollama_client import OllamaClient, MultiModelClient


# =============================================================================
# Helpers
# =============================================================================

def _mock_httpx_response(status_code=200, json_data=None):
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# =============================================================================
# TestOllamaClient
# =============================================================================

class TestOllamaClient:
    @pytest.fixture
    def client(self):
        c = OllamaClient(base_url="http://localhost:11434", model="test-model")
        c._client = AsyncMock(spec=httpx.AsyncClient)
        return c

    async def test_generate_success(self, client):
        client._client.post = AsyncMock(return_value=_mock_httpx_response(
            200, {"response": "hello world", "eval_count": 42, "done_reason": "stop"}
        ))
        request = LLMRequest(prompt="say hello")
        result = await client.generate(request)

        assert isinstance(result, LLMResponse)
        assert result.content == "hello world"
        assert result.model == "test-model"
        assert result.tokens_used == 42

    async def test_generate_includes_system_prompt(self, client):
        client._client.post = AsyncMock(return_value=_mock_httpx_response(
            200, {"response": "ok", "eval_count": 10}
        ))
        request = LLMRequest(prompt="hello", system_prompt="You are helpful")
        await client.generate(request)

        call_args = client._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["system"] == "You are helpful"

    async def test_generate_http_error(self, client):
        from tenacity import RetryError
        client._client.post = AsyncMock(return_value=_mock_httpx_response(500))
        request = LLMRequest(prompt="fail")
        with pytest.raises((RuntimeError, RetryError)):
            await client.generate(request)

    async def test_generate_connection_error(self, client):
        from tenacity import RetryError
        client._client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        request = LLMRequest(prompt="fail")
        with pytest.raises((RuntimeError, RetryError)):
            await client.generate(request)

    async def test_is_available_true(self, client):
        client._client.get = AsyncMock(return_value=_mock_httpx_response(200))
        assert await client.is_available() is True

    async def test_is_available_false(self, client):
        client._client.get = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        assert await client.is_available() is False

    async def test_list_models(self, client):
        client._client.get = AsyncMock(return_value=_mock_httpx_response(
            200, {"models": [{"name": "model-a"}, {"name": "model-b"}]}
        ))
        models = await client.list_models()
        assert models == ["model-a", "model-b"]

    async def test_list_models_error(self, client):
        client._client.get = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        models = await client.list_models()
        assert models == []

    async def test_pull_model_success(self, client):
        client._client.post = AsyncMock(return_value=_mock_httpx_response(200))
        assert await client.pull_model("some-model") is True

    async def test_pull_model_failure(self, client):
        client._client.post = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        assert await client.pull_model("some-model") is False

    async def test_close(self, client):
        mock_http = client._client
        await client.close()
        mock_http.aclose.assert_called_once()
        assert client._client is None


# =============================================================================
# TestMultiModelClient
# =============================================================================

class TestMultiModelClient:
    def test_get_client_creates_once(self):
        mmc = MultiModelClient(base_url="http://localhost:11434")
        c1 = mmc.get_client("model-a")
        c2 = mmc.get_client("model-a")
        assert c1 is c2

    def test_get_client_different_models(self):
        mmc = MultiModelClient(base_url="http://localhost:11434")
        c1 = mmc.get_client("model-a")
        c2 = mmc.get_client("model-b")
        assert c1 is not c2

    async def test_generate_delegates(self):
        mmc = MultiModelClient()
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.generate = AsyncMock(return_value=LLMResponse(
            content="hello", model="model-a", tokens_used=10
        ))
        mmc._clients["model-a"] = mock_client

        request = LLMRequest(prompt="test")
        result = await mmc.generate("model-a", request)
        assert result.content == "hello"
        mock_client.generate.assert_called_once_with(request)

    async def test_close_all(self):
        mmc = MultiModelClient()
        mock_a = AsyncMock(spec=OllamaClient)
        mock_b = AsyncMock(spec=OllamaClient)
        mmc._clients = {"a": mock_a, "b": mock_b}

        await mmc.close_all()
        mock_a.close.assert_called_once()
        mock_b.close.assert_called_once()
        assert mmc._clients == {}
