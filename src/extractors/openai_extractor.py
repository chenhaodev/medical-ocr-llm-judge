"""
OpenAI-compatible API extractor for GPT-4V and similar models.
"""

import requests
from pathlib import Path
from typing import Dict, Any, Union
from .base_extractor import BaseExtractor


class OpenAIExtractor(BaseExtractor):
    """OCR extractor using OpenAI-compatible APIs."""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)

        if not self.api_key:
            raise ValueError("API key is required for OpenAI extractor")

    def chat(self, messages: list, **kwargs) -> str:
        """
        Send chat completion request to OpenAI-compatible API.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            Response text from model
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens)
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI API request failed: {str(e)}")

    def extract(self, image_path: Union[str, Path], prompt: str) -> Dict[str, Any]:
        """
        Extract information from image using OpenAI vision model.

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

        # Determine image format
        image_path = Path(image_path)
        image_format = image_path.suffix.lower().replace('.', '')
        if image_format == 'jpg':
            image_format = 'jpeg'

        # Prepare messages for vision model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_b64}"
                        }
                    }
                ]
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
