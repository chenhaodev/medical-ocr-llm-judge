# Medical OCR Evaluation with LLM-as-Judge

Complete documentation for automated OCR quality evaluation of Chinese medical documents.

## Overview

This framework evaluates vision LLM performance on medical OCR tasks using:
- **InternVL3-78b**: Reference baseline extraction
- **DeepSeek**: Compares DUT extraction vs baseline

### Workflow

```
Image → DUT extracts → InternVL extracts (baseline) → DeepSeek compares → Score + Analysis
```

**DUT (Device Under Test)** = Model being evaluated
**Judge System** = InternVL (baseline) + DeepSeek (comparison)

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### API Keys

```bash
# Required
export OPENROUTER_API_KEY="..."  # For InternVL baseline
export DEEPSEEK_API_KEY="..."    # For DeepSeek judge

# Optional (only if testing these models)
export GLM_API_KEY="..."         # For glm-4v-plus
export OPENAI_API_KEY="..."      # For gpt-4o
```

Get keys:
- OpenRouter: https://openrouter.ai/keys
- DeepSeek: https://platform.deepseek.com/api_keys
- GLM: https://open.bigmodel.cn
- OpenAI: https://platform.openai.com/api-keys

### Basic Usage

```bash
# Default model (qwen2.5vl)
python test.py "data/reports/使用的报告单/1-23/1.jpg"

# Test specific model
python test.py "data/reports/使用的报告单/1-23/1.jpg" --model gpt-4o
python test.py "data/reports/使用的报告单/1-23/1.jpg" --model glm-4v-plus
python test.py "data/reports/使用的报告单/1-23/1.jpg" --model minicpm-v4.5
```

---

## Supported Models

### DUTs (Models Being Evaluated)

| Model | Provider | Local/API | API Key Required |
|-------|----------|-----------|------------------|
| qwen2.5vl | Ollama | Local | None |
| minicpm-v4.5 | Ollama | Local | None |
| glm-4v-plus | GLM API | API | GLM_API_KEY |
| gpt-4o | OpenAI API | API | OPENAI_API_KEY |

### Judge System (Fixed)

- **InternVL3-78b** (OpenRouter): Reference baseline extraction
- **DeepSeek** (DeepSeek API): JSON comparison and analysis

---

## What's Extracted

Medical reports contain diagnostic information beyond just values:

### 1. Structured Data
- Patient information (name, ID, age, gender, department, physician)
- Test items (name, code, measured value, unit, reference range)
- Report metadata (timestamps, report number, hospital)

### 2. Diagnostic Symbols (Critical!)
- **↑, ↓** - Abnormally high/low
- **↑↑, ↓↓, HH, LL** - Severely abnormal (critical values)
- **\*, H, L** - Abnormal markers
- **+, -** - Positive/negative

### 3. Comments & Annotations
- Doctor comments and recommendations
- Lab comments (e.g., "标本轻度溶血")
- Quality control notes
- Special instructions
- Warnings (critical value notifications)
- Footnotes and symbol legends

### 4. Clinical Context
- Result interpretations (e.g., "偏高", "严重偏低")
- Clinical significance notes
- Trend indicators (升高/降低)
- Testing methods and instruments

---

## Evaluation Criteria

| Criterion | Weight | What's Evaluated |
|-----------|--------|------------------|
| **Accuracy** | 30% | Character correctness + ALL symbols preserved exactly |
| **Completeness** | 25% | All fields + diagnostic markers + comments + annotations |
| **Structure** | 20% | JSON organization + logical grouping |
| **Field Correctness** | 25% | Proper assignment + abnormal flags linked to values |

### Grading Scale

- **A+ (9.5-10)**: Excellent, production-ready
- **A (9.0-9.4)**: Very good
- **B+ (8.5-8.9)**: Good
- **B (8.0-8.4)**: Above average
- **C+ (7.5-7.9)**: Average
- **C (7.0-7.4)**: Below average
- **D (6.0-6.9)**: Poor
- **F (0-5.9)**: Failing

---

## API Usage

### Python API

```python
from src.testers import OCRTester, LLMJudge

# Extract with DUT
dut = OCRTester(provider="ollama", model="qwen2.5vl")
result = dut.test_single_image("data/reports/使用的报告单/1-23/1.jpg", "report")

# Extract with InternVL (baseline)
baseline = OCRTester(provider="openrouter", model="internvl3-78b")
ref_result = baseline.test_single_image("data/reports/使用的报告单/1-23/1.jpg", "report")

# Compare with DeepSeek
judge = LLMJudge(judge_provider="deepseek", judge_model="deepseek-chat")
evaluation = judge.compare_extractions(
    image_path="data/reports/使用的报告单/1-23/1.jpg",
    extraction_a=result['extracted_data'],
    extraction_b=ref_result['extracted_data'],
    model_a_name="qwen2.5vl",
    model_b_name="InternVL3-78b"
)

# View results
comp = evaluation['comparison']
print(f"Score: {comp['model_a']['total_score']}/10")
print(f"Strengths: {comp['model_a']['strengths']}")
print(f"Weaknesses: {comp['model_a']['weaknesses']}")
```

### Custom Model Configuration

Edit `config/llm_config.json` to add new models:

```json
{
  "providers": {
    "your_provider": {
      "base_url": "https://api.example.com/v1",
      "api_key_env": "YOUR_API_KEY",
      "models": {
        "your-model": {
          "model_name": "your-model-name",
          "temperature": 0.1,
          "max_tokens": 2000,
          "vision_enabled": true
        }
      }
    }
  }
}
```

Then add to test.py DUTS dictionary:

```python
DUTS = {
    # ... existing models
    "your-model": {"provider": "your_provider", "model": "your-model"}
}
```

---

## Output Format

### Comparison Result

```json
{
  "model_a": {
    "total_score": 7.5,
    "strengths": [
      "Accurate patient information extraction",
      "Good handling of Chinese medical terms"
    ],
    "weaknesses": [
      "Missing diagnostic symbols (↑, ↓)",
      "Incomplete comment extraction"
    ]
  },
  "model_b": {
    "total_score": 8.5
  },
  "key_differences": [
    "Model A missed 3 abnormal markers",
    "Model A did not extract lab comments"
  ],
  "conclusion": "Model A needs improvement in diagnostic symbol extraction",
  "confidence": "high"
}
```

---

## Dataset

### Medical Reports (101 images)
- **Location**: `data/reports/使用的报告单/1-23/`, `data/reports/使用的报告单/24-40/`, etc.
- **Type**: Blood test reports (血常规检验报告单)
- **Content**: Patient info, test values, reference ranges, diagnostic markers

### Medicine Packaging (261 images, 98 categories)
- **Location**: `data/medicine/药品-按药名分原图/*/不带背景/*.jpg`
- **Type**: Medicine box photos
- **Content**: Medicine name, specifications, manufacturer, instructions

---

## Cost Estimation

Per evaluation:
- **InternVL3-78b** (baseline extraction): ~$0.005-0.01
- **DeepSeek** (comparison): ~$0.001-0.002
- **Total**: ~$0.006-0.012 per image

For 100 images: ~$0.60-1.20

Local models (qwen2.5vl, minicpm-v4.5) are free but slower.

---

## Prompts

Prompts are in `prompts/` directory:

- **ocr_extraction_report.txt**: Medical report extraction prompt
- **ocr_extraction_medicine.txt**: Medicine packaging extraction prompt
- **judge_ocr_quality.txt**: Single evaluation prompt (not used in simplified workflow)
- **judge_comparison.txt**: Model comparison prompt

---

## Troubleshooting

### Error: "OPENROUTER_API_KEY not found"
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### Error: "Model deepseek-chat does not support vision"
This is expected - DeepSeek only compares JSON text, not images. The code handles this correctly.

### Error: "Connection to Ollama failed"
```bash
# Start Ollama
ollama serve

# Pull models
ollama pull qwen2.5vl
ollama pull openbmb/minicpm-v4.5
```

### Low scores on all models
- Check if diagnostic symbols are being extracted (↑, ↓, *, H, L)
- Verify comments/annotations are included
- Review `evaluation['comparison']` for specific issues

---

## Best Practices

1. **Run on diverse images**: Test on different hospitals, formats, quality levels
2. **Check diagnostic completeness**: Verify symbols and comments are extracted
3. **Compare multiple DUTs**: Test qwen2.5vl, minicpm-v4.5, glm-4v-plus, gpt-4o
4. **Review detailed findings**: Don't just look at scores - read the analysis
5. **Iterate prompts**: If all models fail on specific fields, update extraction prompts

---

## Architecture

```
.
├── src/                 # Source code
│   ├── extractors/      # LLM API wrappers
│   ├── testers/         # Evaluation logic
│   └── utils/           # Configuration and prompts
├── data/                # Dataset (gitignored)
│   ├── reports/         # Medical reports (101 images)
│   └── medicine/        # Medicine packaging (261 images)
├── config/              # Model configurations
│   └── llm_config.json
├── prompts/             # Extraction and judge prompts
│   ├── ocr_extraction_report.txt
│   ├── ocr_extraction_medicine.txt
│   ├── judge_ocr_quality.txt
│   └── judge_comparison.txt
├── tests/               # Unit tests
│   └── test_imports.py
├── results/             # Output directory (gitignored)
├── docs/                # Documentation
│   └── README_FULL.md
├── test.py              # Main entry point (1-line command)
└── requirements.txt
```

---

## License

For research and educational use. Ensure compliance with healthcare data privacy regulations (HIPAA, GDPR, etc.) when using on real patient data.

---

## Contributing

To add a new model:

1. Add configuration to `config/llm_config.json`
2. Set API key environment variable (if needed)
3. Add to `DUTS` dict in `test.py`
4. Test: `python test.py <image> --model <your-model>`

---

## References

- InternVL3: https://internvl.github.io/
- DeepSeek: https://www.deepseek.com/
- Qwen2.5-VL: https://github.com/QwenLM/Qwen2-VL
- MiniCPM-V: https://github.com/OpenBMB/MiniCPM-V
- GLM-4V: https://open.bigmodel.cn/
