"""
Basic import tests for the medical OCR evaluation framework.
"""

def test_imports():
    """Test that all main modules can be imported."""
    from src.testers import OCRTester, LLMJudge
    from src.extractors import get_extractor
    from src.utils import ConfigLoader, PromptLoader

    assert OCRTester is not None
    assert LLMJudge is not None
    assert get_extractor is not None
    assert ConfigLoader is not None
    assert PromptLoader is not None
    print("✓ All imports successful")


def test_config_loading():
    """Test configuration loading."""
    from src.utils import ConfigLoader

    loader = ConfigLoader()
    config = loader.load_llm_config()

    assert "providers" in config
    assert "ollama" in config["providers"]
    assert "openai" in config["providers"]
    assert "glm" in config["providers"]
    assert "openrouter" in config["providers"]
    assert "deepseek" in config["providers"]
    print("✓ Config loading successful")


def test_prompt_loading():
    """Test prompt loading."""
    from src.utils import PromptLoader

    loader = PromptLoader()

    # Test report extraction prompt
    report_prompt = loader.get_ocr_extraction_prompt("report")
    assert len(report_prompt) > 0
    assert "patient_info" in report_prompt.lower()

    # Test medicine extraction prompt
    medicine_prompt = loader.get_ocr_extraction_prompt("medicine")
    assert len(medicine_prompt) > 0
    assert "medicine" in medicine_prompt.lower()

    print("✓ Prompt loading successful")


if __name__ == "__main__":
    test_imports()
    test_config_loading()
    test_prompt_loading()
    print("\n✅ All tests passed!")
