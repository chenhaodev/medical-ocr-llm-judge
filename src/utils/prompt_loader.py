"""
Prompt template loader utility.
"""

from pathlib import Path
from typing import Dict, Optional


class PromptLoader:
    """Loads and manages prompt templates."""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._prompt_cache: Dict[str, str] = {}

    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template by name.

        Args:
            prompt_name: Name of the prompt file (without .txt extension)

        Returns:
            Prompt template as string
        """
        if prompt_name not in self._prompt_cache:
            prompt_path = self.prompts_dir / f"{prompt_name}.txt"

            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

            with open(prompt_path, 'r', encoding='utf-8') as f:
                self._prompt_cache[prompt_name] = f.read()

        return self._prompt_cache[prompt_name]

    def get_ocr_extraction_prompt(self, document_type: str) -> str:
        """
        Get OCR extraction prompt for document type.

        Args:
            document_type: Either 'report' or 'medicine'

        Returns:
            Prompt template string
        """
        if document_type == "report":
            return self.load_prompt("ocr_extraction_report")
        elif document_type == "medicine":
            return self.load_prompt("ocr_extraction_medicine")
        else:
            raise ValueError(f"Unknown document type: {document_type}")

