from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int = 0
    generation_time: float = 0.0
    finish_reason: str = "stop"
    raw_response: dict[str, Any] | None = None


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.3
    max_tokens: int = 8192
    stop_sequences: list[str] | None = None
    
    def to_messages(self) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})
        return messages


class BaseLLMClient(ABC):
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        pass
    
    @abstractmethod
    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        pass
    
    @abstractmethod
    async def list_models(self) -> list[str]:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass
