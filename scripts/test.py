#!/usr/bin/env python
"""
OCR evaluation with LLM-as-Judge

Usage:
  python scripts/test.py <image> [options]
  python scripts/test.py --random <dir> [options]

Options:
  --model <name>      DUT model (qwen2.5vl, minicpm-v4.5, glm-4v-plus, gpt-4o)
  --verbose, -v       Show detailed extraction results
  --random <dir>      Randomly select images from directory
  --count <n>         Number of random samples (default: 3, max: 10)
  --save <file>       Save results to JSON file

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
import random
from pathlib import Path
from datetime import datetime
from src.testers import OCRTester, LLMJudge

# DUT configurations
DUTS = {
    "qwen2.5vl": {"provider": "ollama", "model": "qwen2.5vl"},
    "minicpm-v4.5": {"provider": "ollama", "model": "minicpm-v4.5"},
    "glm-4v-plus": {"provider": "glm", "model": "glm-4v-plus"},
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"}
}

def get_random_images(directory, count=3):
    """Get random images from directory."""
    directory = Path(directory)

    # Find all images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        images.extend(directory.rglob(ext))

    if not images:
        print(f"❌ No images found in {directory}")
        sys.exit(1)

    # Sample random images
    count = min(count, len(images), 10)  # Max 10
    selected = random.sample(images, count)

    return [str(img) for img in selected]


def evaluate_single(image_path, dut_name, dut_config, verbose=False):
    """Evaluate a single image."""
    # Auto-detect type
    doc_type = "medicine" if ("药品" in image_path or "medicine" in image_path) else "report"

    print(f"\n{'='*60}")
    print(f"Image: {image_path}")
    print(f"Type: {doc_type}")
    print(f"DUT: {dut_name}")
    print("="*60)

    # Extract with DUT
    print(f"\n[1/3] Extracting with {dut_name}...")
    dut = OCRTester(provider=dut_config["provider"], model=dut_config["model"])
    result = dut.test_single_image(image_path, doc_type)

    if result['extraction_error']:
        print(f"❌ {dut_name} failed: {result['extraction_error']}")
        return None

    print(f"✓ Completed in {result['extraction_time']:.2f}s")

    if verbose:
        print(f"\n📝 {dut_name} Extraction Preview:")
        preview = json.dumps(result['extracted_data'], ensure_ascii=False, indent=2)
        print(preview[:500] + "..." if len(preview) > 500 else preview)

    # Extract with InternVL (baseline)
    print(f"\n[2/3] Extracting with InternVL3-78b (baseline)...")
    baseline = OCRTester(provider="openrouter", model="internvl3-78b")
    ref_result = baseline.test_single_image(image_path, doc_type)

    if ref_result['extraction_error']:
        print(f"❌ InternVL failed: {ref_result['extraction_error']}")
        return None

    print(f"✓ Completed in {ref_result['extraction_time']:.2f}s")

    if verbose:
        print(f"\n📝 InternVL Extraction Preview:")
        preview = json.dumps(ref_result['extracted_data'], ensure_ascii=False, indent=2)
        print(preview[:500] + "..." if len(preview) > 500 else preview)

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
        return None

    print(f"✓ Completed in {evaluation['comparison_time']:.2f}s")

    # Display results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS")
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

            if verbose and 'detailed_analysis' in comp:
                print(f"\n📋 Detailed Analysis:")
                print(json.dumps(comp['detailed_analysis'], ensure_ascii=False, indent=2))
        else:
            print(json.dumps(comp, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))

    print("\n" + "="*60)

    return {
        "image": image_path,
        "dut_result": result,
        "baseline_result": ref_result,
        "evaluation": evaluation,
        "score": comp['model_a'].get('total_score', 0) if 'model_a' in comp else 0
    }


def main():
    # Parse arguments
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        print("\nExamples:")
        print('  # Single image')
        print('  python scripts/test.py "data/reports/使用的报告单/1-23/10.jpg"')
        print('  python scripts/test.py "data/reports/使用的报告单/1-23/10.jpg" --model gpt-4o --verbose')
        print()
        print('  # Random sampling')
        print('  python scripts/test.py --random "data/reports/使用的报告单/1-23"')
        print('  python scripts/test.py --random "data/reports/使用的报告单" --count 5 --model minicpm-v4.5')
        print('  python scripts/test.py --random "data/reports" --count 10 --save results.json')
        print("\nRequired API Keys:")
        print("  export OPENROUTER_API_KEY='...'  # For InternVL baseline")
        print("  export DEEPSEEK_API_KEY='...'    # For DeepSeek judge")
        print("  export GLM_API_KEY='...'         # Only if using glm-4v-plus")
        print("  export OPENAI_API_KEY='...'      # Only if using gpt-4o")
        sys.exit(0 if sys.argv[1] in ["-h", "--help"] else 1)

    # Parse options
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    random_mode = "--random" in sys.argv

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

    # Get random count
    count = 3
    if "--count" in sys.argv:
        idx = sys.argv.index("--count")
        if idx + 1 < len(sys.argv):
            count = int(sys.argv[idx + 1])
            count = max(1, min(count, 10))  # Between 1-10

    # Get save path
    save_path = None
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        if idx + 1 < len(sys.argv):
            save_path = sys.argv[idx + 1]

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

    # Get images
    if random_mode:
        idx = sys.argv.index("--random")
        if idx + 1 >= len(sys.argv):
            print("❌ --random requires a directory path")
            sys.exit(1)
        directory = sys.argv[idx + 1]
        images = get_random_images(directory, count)
        print(f"\n🎲 Randomly selected {len(images)} images from {directory}")
    else:
        images = [sys.argv[1]]

    # Evaluate all images
    results = []
    for i, image_path in enumerate(images, 1):
        if len(images) > 1:
            print(f"\n{'#'*60}")
            print(f"# Processing {i}/{len(images)}")
            print(f"{'#'*60}")

        result = evaluate_single(image_path, dut_name, dut_config, verbose)
        if result:
            results.append(result)

    # Summary for multiple images
    if len(results) > 1:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        avg_score = sum(r['score'] for r in results) / len(results)
        print(f"\n📊 Average Score: {avg_score:.2f}/10")
        print(f"✓ Successful: {len(results)}/{len(images)}")

        print(f"\n📝 Individual Scores:")
        for r in results:
            fname = Path(r['image']).name
            print(f"   {r['score']:.1f}/10  {fname}")

    # Save results
    if save_path and results:
        output = {
            "timestamp": datetime.now().isoformat(),
            "dut_model": dut_name,
            "total_images": len(results),
            "average_score": sum(r['score'] for r in results) / len(results) if results else 0,
            "results": results
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Results saved to: {save_path}")

if __name__ == "__main__":
    main()
