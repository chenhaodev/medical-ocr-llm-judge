"""
LLM-as-Judge evaluation module for automated OCR quality assessment.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from ..extractors import get_extractor
from ..utils import ConfigLoader, PromptLoader


class LLMJudge:
    """Use LLMs to automatically evaluate OCR extraction quality."""

    def __init__(self,
                 judge_provider: str = "openrouter",
                 judge_model: str = "internvl3-78b",
                 config_dir: str = "config"):
        """
        Initialize LLM Judge.

        Args:
            judge_provider: Provider for judge model
            judge_model: Model to use as judge
            config_dir: Path to configuration directory
        """
        self.judge_provider = judge_provider
        self.judge_model = judge_model

        # Load configurations
        self.config_loader = ConfigLoader(config_dir)
        self.prompt_loader = PromptLoader()

        # Get judge model configuration
        model_config = self.config_loader.get_model_config(judge_provider, judge_model)
        self.judge_extractor = get_extractor(judge_provider, model_config)

    def evaluate_single(self,
                       image_path: Union[str, Path],
                       extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single OCR extraction using LLM as judge.

        Args:
            image_path: Path to original image
            extracted_data: OCR extraction results to evaluate

        Returns:
            Evaluation results with scores and feedback
        """
        image_path = Path(image_path)

        # Load judge prompt
        base_prompt = self.prompt_loader.load_prompt("judge_ocr_quality")

        # Add extracted data to prompt
        extracted_json = json.dumps(extracted_data, ensure_ascii=False, indent=2)
        full_prompt = f"{base_prompt}\n\n**OCR Extraction to Evaluate:**\n```json\n{extracted_json}\n```"

        # Get judge evaluation
        start_time = datetime.now()
        try:
            evaluation_result = self.judge_extractor.extract(image_path, full_prompt)
            evaluation_time = (datetime.now() - start_time).total_seconds()
            evaluation_error = None
        except Exception as e:
            evaluation_result = {}
            evaluation_time = (datetime.now() - start_time).total_seconds()
            evaluation_error = str(e)

        # Build result
        result = {
            "image_path": str(image_path),
            "judge_info": {
                "provider": self.judge_provider,
                "model": self.judge_model
            },
            "evaluation_time": evaluation_time,
            "extracted_data": extracted_data,
            "evaluation": evaluation_result,
            "evaluation_error": evaluation_error,
            "timestamp": datetime.now().isoformat()
        }

        # Extract key metrics if evaluation succeeded
        if not evaluation_error and evaluation_result:
            result["summary_metrics"] = self._extract_summary_metrics(evaluation_result)

        return result

    def _extract_summary_metrics(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract summary metrics from evaluation result.

        Args:
            evaluation: Full evaluation result

        Returns:
            Summary metrics dictionary
        """
        metrics = {
            "overall_score": evaluation.get("overall_score", 0),
            "total_possible": evaluation.get("total_possible", 10),
            "overall_percentage": 0,
            "grade": evaluation.get("grade", "N/A"),
            "usability": evaluation.get("usability", "unknown")
        }

        if metrics["total_possible"] > 0:
            metrics["overall_percentage"] = round(
                (metrics["overall_score"] / metrics["total_possible"]) * 100, 2
            )

        # Extract criteria scores
        criteria_scores = evaluation.get("criteria_scores", {})
        for criterion, details in criteria_scores.items():
            if isinstance(details, dict):
                metrics[f"{criterion}_score"] = details.get("score", 0)
                metrics[f"{criterion}_percentage"] = details.get("percentage", 0)

        # Count issues
        detailed_findings = evaluation.get("detailed_findings", {})
        metrics["error_count"] = len(detailed_findings.get("errors", []))
        metrics["missing_field_count"] = len(detailed_findings.get("missing_fields", []))
        metrics["hallucination_count"] = len(detailed_findings.get("hallucinations", []))
        metrics["correct_extraction_count"] = len(detailed_findings.get("correct_extractions", []))

        return metrics

    def compare_extractions(self,
                          image_path: Union[str, Path],
                          extraction_a: Dict[str, Any],
                          extraction_b: Dict[str, Any],
                          model_a_name: str = "Model A",
                          model_b_name: str = "Model B") -> Dict[str, Any]:
        """
        Compare two OCR extractions using LLM as judge.

        Args:
            image_path: Path to original image
            extraction_a: First extraction to compare
            extraction_b: Second extraction to compare
            model_a_name: Name of first model
            model_b_name: Name of second model

        Returns:
            Comparison results with winner and detailed analysis
        """
        image_path = Path(image_path)

        # Load comparison prompt
        base_prompt = self.prompt_loader.load_prompt("judge_comparison")

        # Add extractions to prompt
        extraction_a_json = json.dumps(extraction_a, ensure_ascii=False, indent=2)
        extraction_b_json = json.dumps(extraction_b, ensure_ascii=False, indent=2)

        full_prompt = f"""{base_prompt}

**{model_a_name} Extraction:**
```json
{extraction_a_json}
```

**{model_b_name} Extraction:**
```json
{extraction_b_json}
```

Please provide a detailed comparison following the specified JSON format.
"""

        # Get judge comparison (text-only, no image needed)
        start_time = datetime.now()
        try:
            # Use chat for text-only comparison (no vision required)
            messages = [{"role": "user", "content": full_prompt}]
            response_text = self.judge_extractor.chat(messages)

            # Parse the response
            comparison_result = self.judge_extractor.parse_json_response(response_text)
            if not comparison_result:
                comparison_result = {"raw_response": response_text}

            comparison_time = (datetime.now() - start_time).total_seconds()
            comparison_error = None
        except Exception as e:
            comparison_result = {}
            comparison_time = (datetime.now() - start_time).total_seconds()
            comparison_error = str(e)

        # Build result
        result = {
            "image_path": str(image_path),
            "judge_info": {
                "provider": self.judge_provider,
                "model": self.judge_model
            },
            "comparison_time": comparison_time,
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "extraction_a": extraction_a,
            "extraction_b": extraction_b,
            "comparison": comparison_result,
            "comparison_error": comparison_error,
            "timestamp": datetime.now().isoformat()
        }

        return result

