"""
OCR testing framework for evaluating LLM-based OCR extraction.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from ..extractors import get_extractor
from ..utils import ConfigLoader, PromptLoader


class OCRTester:
    """Test and evaluate OCR extraction quality."""

    def __init__(self,
                 provider: str = "ollama",
                 model: str = "qwen2.5vl",
                 config_dir: str = "config"):
        """
        Initialize OCR tester.

        Args:
            provider: LLM provider name
            model: Model name
            config_dir: Path to configuration directory
        """
        self.provider = provider
        self.model = model

        # Load configurations
        self.config_loader = ConfigLoader(config_dir)
        self.prompt_loader = PromptLoader()

        # Get model configuration
        model_config = self.config_loader.get_model_config(provider, model)
        self.extractor = get_extractor(provider, model_config)

    def test_single_image(self,
                         image_path: Union[str, Path],
                         document_type: str,
                         ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test OCR extraction on a single image.

        Args:
            image_path: Path to image file
            document_type: Type of document ('report' or 'medicine')
            ground_truth: Optional ground truth data for comparison

        Returns:
            Test result dictionary
        """
        image_path = Path(image_path)

        # Load appropriate prompt
        prompt = self.prompt_loader.get_ocr_extraction_prompt(document_type)

        # Extract information
        start_time = datetime.now()
        try:
            extracted_data = self.extractor.extract(image_path, prompt)
            extraction_time = (datetime.now() - start_time).total_seconds()
            extraction_error = None
        except Exception as e:
            extracted_data = {}
            extraction_time = (datetime.now() - start_time).total_seconds()
            extraction_error = str(e)

        # Build result
        result = {
            "image_path": str(image_path),
            "document_type": document_type,
            "model_info": {
                "provider": self.provider,
                "model": self.model
            },
            "extraction_time": extraction_time,
            "extracted_data": extracted_data,
            "extraction_error": extraction_error,
            "timestamp": datetime.now().isoformat()
        }

        # Ground truth support removed (use LLM Judge instead)
        result["metrics"] = None
        result["ground_truth"] = None

        return result
