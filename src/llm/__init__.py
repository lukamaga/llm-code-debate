"""
LLM clients for the Code Debate System.
"""
from .base import BaseLLMClient, LLMRequest, LLMResponse
from .ollama_client import MultiModelClient, OllamaClient

__all__ = [
    "BaseLLMClient",
    "LLMRequest",
    "LLMResponse",
    "MultiModelClient",
    "OllamaClient",
]
