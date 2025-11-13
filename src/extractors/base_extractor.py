"""
Base class for OCR extractors using LLMs.
"""

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union


class BaseExtractor(ABC):
    """Abstract base class for LLM-based OCR extractors."""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize extractor with model configuration.

        Args:
            model_config: Configuration dictionary containing model settings
        """
        self.model_config = model_config
        self.base_url = model_config.get('base_url')
        self.model_name = model_config.get('model_name')
        self.temperature = model_config.get('temperature', 0.1)
        self.max_tokens = model_config.get('max_tokens', 2000)
        self.vision_enabled = model_config.get('vision_enabled', False)
        self.api_key = model_config.get('api_key')

    def encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from response text, handling various formats.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            try:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to extract any JSON object
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    @abstractmethod
    def extract(self, image_path: Union[str, Path], prompt: str) -> Dict[str, Any]:
        """
        Extract information from image using LLM.

        Args:
            image_path: Path to image file
            prompt: Extraction prompt

        Returns:
            Extracted information as dictionary
        """
        pass

    @abstractmethod
    def chat(self, messages: list, **kwargs) -> str:
        """
        Send chat completion request to LLM.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Response text from LLM
        """
        pass
