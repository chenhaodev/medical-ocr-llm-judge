# Medical OCR Evaluation with LLM Judge

Automated OCR quality evaluation for Chinese medical documents using LLM-as-Judge.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API keys (required)
export OPENROUTER_API_KEY="..."  # For InternVL baseline
export DEEPSEEK_API_KEY="..."    # For DeepSeek judge

# Test (defaults to qwen2.5vl)
python test.py "data/reports/使用的报告单/1-23/1.jpg"

# Test with different models
python test.py "data/reports/使用的报告单/1-23/1.jpg" --model gpt-4o
python test.py "data/reports/使用的报告单/1-23/1.jpg" --model glm-4v-plus
python test.py "data/reports/使用的报告单/1-23/1.jpg" --model minicpm-v4.5
```

## Supported DUTs (Device Under Test)

| Model | Provider | API Key Required |
|-------|----------|-----------------|
| qwen2.5vl (default) | Ollama (local) | None |
| minicpm-v4.5 | Ollama (local) | None |
| glm-4v-plus | GLM API | GLM_API_KEY |
| gpt-4o | OpenAI API | OPENAI_API_KEY |

## Judge System

- **InternVL3-78b**: Generates reference baseline extraction
- **DeepSeek**: Compares DUT vs baseline, provides analysis

## What's Extracted

Medical reports contain more than just values - they contain diagnostic context:

1. **Structured Data**: Patient info, test values, reference ranges
2. **Diagnostic Symbols**: ↑, ↓, *, H, L, HH, LL (abnormal markers)
3. **Comments**: Doctor notes, lab comments, warnings
4. **Clinical Context**: Interpretations, quality control notes, footnotes

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Accuracy | 30% | Correct values + ALL symbols preserved |
| Completeness | 25% | All fields + markers + comments extracted |
| Structure | 20% | Logical JSON organization |
| Field Correctness | 25% | Proper field assignment + flag association |

## API Keys

- OPENROUTER_API_KEY: https://openrouter.ai/keys
- DEEPSEEK_API_KEY: https://platform.deepseek.com/api_keys
- GLM_API_KEY: https://open.bigmodel.cn
- OPENAI_API_KEY: https://platform.openai.com/api-keys

## Configuration

Edit `config/llm_config.json` to modify model settings.

## Documentation

See `docs/README_FULL.md` for detailed documentation.

## License

For research and educational use.
