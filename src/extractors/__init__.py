"""
OCR extractor modules using various LLM providers.
"""

from .base_extractor import BaseExtractor
from .ollama_extractor import OllamaExtractor
from .openai_extractor import OpenAIExtractor

__all__ = ['BaseExtractor', 'OllamaExtractor', 'OpenAIExtractor']


def get_extractor(provider: str, model_config: dict) -> BaseExtractor:
    """
    Factory function to get appropriate extractor based on provider.

    Args:
        provider: Provider name ('ollama', 'openai', 'deepseek', 'glm', 'openrouter')
        model_config: Model configuration dictionary

    Returns:
        Appropriate extractor instance
    """
    if provider == "ollama":
        return OllamaExtractor(model_config)
    elif provider in ["openai", "deepseek", "glm", "openrouter"]:
        # All these use OpenAI-compatible API
        return OpenAIExtractor(model_config)
    else:
        raise ValueError(f"Unknown provider: {provider}")
