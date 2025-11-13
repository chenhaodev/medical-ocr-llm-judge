"""
Configuration loader utility for medical OCR evaluation framework.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Loads and manages configuration files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._llm_config: Optional[Dict[str, Any]] = None

    def load_llm_config(self) -> Dict[str, Any]:
        """Load LLM provider configuration."""
        if self._llm_config is None:
            config_path = self.config_dir / "llm_config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                self._llm_config = json.load(f)
        return self._llm_config

    def get_model_config(self, provider: str, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        config = self.load_llm_config()

        if provider not in config['providers']:
            raise ValueError(f"Unknown provider: {provider}")

        provider_config = config['providers'][provider]

        if model not in provider_config['models']:
            raise ValueError(f"Unknown model: {model} for provider: {provider}")

        model_config = provider_config['models'][model].copy()
        model_config['base_url'] = provider_config['base_url']

        # Add API key if specified
        if 'api_key_env' in provider_config:
            api_key = os.getenv(provider_config['api_key_env'])
            if api_key:
                model_config['api_key'] = api_key

        return model_config

