#!/usr/bin/env python
"""
1-line OCR evaluation: python scripts/test.py <image> [--model <model>]

Supported DUTs:
  - qwen2.5vl (default, Ollama local)
  - minicpm-v4.5 (Ollama local)
  - glm-4v-plus (GLM API, needs GLM_API_KEY)
  - gpt-4o (OpenAI API, needs OPENAI_API_KEY)

Judge: InternVL3-78b (baseline) + DeepSeek (comparison)
"""

import sys
import os
import json
from pathlib import Path
from src.testers import OCRTester, LLMJudge

# DUT configurations
DUTS = {
    "qwen2.5vl": {"provider": "ollama", "model": "qwen2.5vl"},
    "minicpm-v4.5": {"provider": "ollama", "model": "minicpm-v4.5"},
    "glm-4v-plus": {"provider": "glm", "model": "glm-4v-plus"},
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"}
}

def main():
    # Parse arguments
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        print("\nExamples:")
        print('  python scripts/test.py "data/reports/使用的报告单/1-23/10.jpg"')
        print('  python scripts/test.py "data/reports/使用的报告单/1-23/10.jpg" --model gpt-4o')
        print('  python scripts/test.py "data/medicine/药品-按药名分原图/美林 布洛芬混悬液/不带背景/美林 布洛芬混悬液.jpg" --model glm-4v-plus')
        print("\nRequired API Keys:")
        print("  export OPENROUTER_API_KEY='...'  # For InternVL baseline")
        print("  export DEEPSEEK_API_KEY='...'    # For DeepSeek judge")
        print("  export GLM_API_KEY='...'         # Only if using glm-4v-plus")
        print("  export OPENAI_API_KEY='...'      # Only if using gpt-4o")
        sys.exit(0 if sys.argv[1] in ["-h", "--help"] else 1)

    image_path = sys.argv[1]

    # Get DUT model
    dut_name = "qwen2.5vl"  # default
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            dut_name = sys.argv[idx + 1]

    if dut_name not in DUTS:
        print(f"❌ Unknown model: {dut_name}")
        print(f"   Supported: {', '.join(DUTS.keys())}")
        sys.exit(1)

    dut_config = DUTS[dut_name]

    # Check API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY required (for InternVL baseline)")
        sys.exit(1)
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ DEEPSEEK_API_KEY required (for DeepSeek judge)")
        sys.exit(1)
    if dut_config["provider"] == "glm" and not os.getenv("GLM_API_KEY"):
        print("❌ GLM_API_KEY required for glm-4v-plus")
        sys.exit(1)
    if dut_config["provider"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required for gpt-4o")
        sys.exit(1)

    # Auto-detect type
    doc_type = "medicine" if ("药品" in image_path or "medicine" in image_path) else "report"

    print(f"Image: {image_path}")
    print(f"DUT: {dut_name}")
    print(f"Judge: InternVL3-78b + DeepSeek")
    print("="*60)

    # Extract with DUT
    print(f"\n[1/3] Extracting with {dut_name}...")
    dut = OCRTester(provider=dut_config["provider"], model=dut_config["model"])
    result = dut.test_single_image(image_path, doc_type)

    if result['extraction_error']:
        print(f"❌ {dut_name} failed: {result['extraction_error']}")
        sys.exit(1)
    print(f"✓ Completed in {result['extraction_time']:.2f}s")

    # Extract with InternVL (baseline)
    print(f"\n[2/3] Extracting with InternVL3-78b (baseline)...")
    baseline = OCRTester(provider="openrouter", model="internvl3-78b")
    ref_result = baseline.test_single_image(image_path, doc_type)

    if ref_result['extraction_error']:
        print(f"❌ InternVL failed: {ref_result['extraction_error']}")
        sys.exit(1)
    print(f"✓ Completed in {ref_result['extraction_time']:.2f}s")

    # Compare with DeepSeek
    print(f"\n[3/3] Comparing with DeepSeek judge...")
    judge = LLMJudge(judge_provider="deepseek", judge_model="deepseek-chat")
    evaluation = judge.compare_extractions(
        image_path=image_path,
        extraction_a=result['extracted_data'],
        extraction_b=ref_result['extracted_data'],
        model_a_name=dut_name,
        model_b_name="InternVL3-78b"
    )

    if evaluation.get('comparison_error'):
        print(f"❌ Judge failed: {evaluation['comparison_error']}")
        sys.exit(1)

    # Display results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({dut_name})")
    print("="*60)

    if 'comparison' in evaluation and evaluation['comparison']:
        comp = evaluation['comparison']

        if 'model_a' in comp:
            score = comp['model_a'].get('total_score', 'N/A')
            print(f"\n📊 {dut_name} Score: {score}/10")

            if comp['model_a'].get('strengths'):
                strengths = comp['model_a']['strengths'][:3] if isinstance(comp['model_a']['strengths'], list) else []
                if strengths:
                    print(f"\n✅ Strengths:")
                    for s in strengths:
                        print(f"   • {s}")

            if comp['model_a'].get('weaknesses'):
                weaknesses = comp['model_a']['weaknesses'][:3] if isinstance(comp['model_a']['weaknesses'], list) else []
                if weaknesses:
                    print(f"\n❌ Weaknesses:")
                    for w in weaknesses:
                        print(f"   • {w}")

            if 'key_differences' in comp:
                diffs = comp['key_differences'][:3] if isinstance(comp['key_differences'], list) else []
                if diffs:
                    print(f"\n🔍 vs Baseline:")
                    for d in diffs:
                        print(f"   • {d}")

            if 'conclusion' in comp:
                print(f"\n💬 {comp['conclusion']}")
        else:
            print(json.dumps(comp, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
