from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseLLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        model: str = "qwen2.5-coder:7b",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(self, request: LLMRequest) -> LLMResponse:
        client = await self._get_client()
        
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "num_ctx": 32768,
            },
        }
        
        if request.system_prompt:
            payload["system"] = request.system_prompt
        
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences
        
        start_time = time.time()
        
        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            generation_time = time.time() - start_time
            finish_reason = data.get("done_reason", "stop")
            tokens_used = data.get("eval_count", 0)

            if finish_reason == "length":
                logger.warning(
                    "LLM output TRUNCATED at num_predict limit "
                    "(model=%s, tokens_used=%d, num_predict=%d, gen_time=%.1fs). "
                    "Increase max_tokens or shorten prompt.",
                    self.model, tokens_used, request.max_tokens, generation_time,
                )

            return LLMResponse(
                content=data.get("response", ""),
                model=self.model,
                tokens_used=tokens_used,
                generation_time=generation_time,
                finish_reason=finish_reason,
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Ollama connection error: {e}") from e
    
    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        client = await self._get_client()
        
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "num_ctx": 32768,
            },
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        try:
            async with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Ollama connection error: {e}") from e
    
    async def is_available(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/pull",
                json={"name": model_name},
                timeout=httpx.Timeout(600.0),
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def __repr__(self) -> str:
        return f"OllamaClient(model={self.model}, base_url={self.base_url})"


class MultiModelClient:
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._clients: dict[str, OllamaClient] = {}
    
    def get_client(self, model: str) -> OllamaClient:
        if model not in self._clients:
            self._clients[model] = OllamaClient(
                base_url=self.base_url,
                timeout=self.timeout,
                model=model,
            )
        return self._clients[model]
    
    async def generate(
        self, model: str, request: LLMRequest
    ) -> LLMResponse:
        client = self.get_client(model)
        return await client.generate(request)
    
    async def check_all_models(self, models: list[str]) -> dict[str, bool]:
        client = self.get_client(models[0])
        available_models = await client.list_models()
        return {
            model: any(model in am for am in available_models)
            for model in models
        }
    
    async def close_all(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
