"""
Base LLM client interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    tokens_used: int = 0
    generation_time: float = 0.0
    finish_reason: str = "stop"
    raw_response: dict[str, Any] | None = None


@dataclass
class LLMRequest:
    """Request to an LLM."""
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096
    stop_sequences: list[str] | None = None
    
    def to_messages(self) -> list[dict[str, str]]:
        """Convert to message format."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})
        return messages


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Defines the interface that all LLM clients must implement.
    """
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            request: The LLM request containing prompt and parameters.
            
        Returns:
            LLMResponse with the generated content.
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            request: The LLM request containing prompt and parameters.
            
        Yields:
            Chunks of generated text.
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if the LLM is available.
        
        Returns:
            True if the LLM is available, False otherwise.
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> list[str]:
        """
        List available models.
        
        Returns:
            List of available model names.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass
