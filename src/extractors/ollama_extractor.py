"""
Ollama-based OCR extractor for local models.
"""

import requests
from pathlib import Path
from typing import Dict, Any, Union
from .base_extractor import BaseExtractor


class OllamaExtractor(BaseExtractor):
    """OCR extractor using Ollama local models."""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)

        if not self.base_url:
            self.base_url = "http://localhost:11434"

    def chat(self, messages: list, **kwargs) -> str:
        """
        Send chat completion request to Ollama.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (stream, temperature, etc.)

        Returns:
            Response text from model
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": kwargs.get('stream', False),
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens)
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            return result.get('message', {}).get('content', '')

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {str(e)}")

    def extract(self, image_path: Union[str, Path], prompt: str) -> Dict[str, Any]:
        """
        Extract information from image using Ollama vision model.

        Args:
            image_path: Path to image file
            prompt: Extraction prompt

        Returns:
            Extracted information as dictionary
        """
        if not self.vision_enabled:
            raise ValueError(f"Model {self.model_name} does not support vision")

        # Encode image to base64
        image_b64 = self.encode_image(image_path)

        # Prepare messages for vision model
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }
        ]

        # Get response
        response_text = self.chat(messages)

        # Parse JSON response
        result = self.parse_json_response(response_text)

        if result is None:
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response_text
            }

        return result

    def verify_connection(self) -> bool:
        """
        Verify connection to Ollama server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list:
        """
        List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            models = response.json().get('models', [])
            return [model['name'] for model in models]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list Ollama models: {str(e)}")
