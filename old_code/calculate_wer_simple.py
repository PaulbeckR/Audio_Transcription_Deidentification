"""
Simple WER calculation without Unicode emojis for Windows compatibility.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.metrics import calculate_wer_from_files
import pandas as pd

# File paths
GROUND_TRUTH = "Audio_Local_tests/transcription_test/hamlet_test_truth.xlsx"
BASE_OUTPUT = "Audio_Local_tests/baseline_output/hamlet_test_whisper_base.csv"
LARGE_OUTPUT = "Audio_Local_tests/baseline_output/hamlet_test_whisper_large.csv"

print("="*80)
print("WER CALCULATION - Baseline Test Results")
print("="*80)

# First, check what columns exist
print("\nChecking ground truth file columns...")
try:
    df_truth = pd.read_excel(GROUND_TRUTH)
    print(f"Available columns: {list(df_truth.columns)}")
    print(f"Number of rows: {len(df_truth)}")

    # Try to find text column
    text_col = None
    for col in df_truth.columns:
        if 'transcr' in col.lower() or 'text' in col.lower():
            text_col = col
            print(f"Using column: '{text_col}'")
            break

    if not text_col and len(df_truth.columns) > 0:
        # Use first column that looks like text
        for col in df_truth.columns:
            if df_truth[col].dtype == 'object':
                text_col = col
                print(f"Using first text column: '{text_col}'")
                break

except Exception as e:
    print(f"Error reading ground truth: {e}")
    sys.exit(1)

if not text_col:
    print("ERROR: Could not find text column in ground truth file")
    sys.exit(1)

# Calculate WER for Base model
print("\n" + "="*80)
print("WHISPER BASE MODEL")
print("="*80)

try:
    wer_base = calculate_wer_from_files(
        GROUND_TRUTH,
        BASE_OUTPUT,
        text_column=text_col
    )

    print(f"\nWord Error Rate (WER): {wer_base['wer']:.2%}")
    print(f"Match Error Rate (MER): {wer_base['mer']:.2%}")
    print(f"Word Info Lost (WIL): {wer_base['wil']:.2%}")
    print(f"\nError Breakdown:")
    print(f"  Substitutions: {wer_base['substitutions']}")
    print(f"  Deletions: {wer_base['deletions']}")
    print(f"  Insertions: {wer_base['insertions']}")
    print(f"  Correct (Hits): {wer_base['hits']}")

    total_words = wer_base['hits'] + wer_base['substitutions'] + wer_base['deletions']
    print(f"\nTotal Reference Words: {total_words}")

except Exception as e:
    print(f"\nError calculating WER for base model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Calculate WER for Large model
print("\n" + "="*80)
print("WHISPER LARGE MODEL")
print("="*80)

try:
    wer_large = calculate_wer_from_files(
        GROUND_TRUTH,
        LARGE_OUTPUT,
        text_column=text_col
    )

    print(f"\nWord Error Rate (WER): {wer_large['wer']:.2%}")
    print(f"Match Error Rate (MER): {wer_large['mer']:.2%}")
    print(f"Word Info Lost (WIL): {wer_large['wil']:.2%}")
    print(f"\nError Breakdown:")
    print(f"  Substitutions: {wer_large['substitutions']}")
    print(f"  Deletions: {wer_large['deletions']}")
    print(f"  Insertions: {wer_large['insertions']}")
    print(f"  Correct (Hits): {wer_large['hits']}")

    total_words = wer_large['hits'] + wer_large['substitutions'] + wer_large['deletions']
    print(f"\nTotal Reference Words: {total_words}")

except Exception as e:
    print(f"\nError calculating WER for large model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Comparison
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

improvement = wer_base['wer'] - wer_large['wer']
pct_improvement = (improvement / wer_base['wer']) * 100 if wer_base['wer'] > 0 else 0

print(f"\nBase Model WER:  {wer_base['wer']:.2%}")
print(f"Large Model WER: {wer_large['wer']:.2%}")
print(f"\nAbsolute Improvement: {improvement:.2%}")
print(f"Relative Improvement: {pct_improvement:.1f}%")

if improvement > 0:
    print("\nLarge model is MORE accurate")
elif improvement < 0:
    print("\nBase model is MORE accurate (unexpected)")
else:
    print("\nModels performed equally")

# Cost-benefit analysis
print("\n" + "-"*80)
print("COST-BENEFIT ANALYSIS")
print("-"*80)

base_time = 105.21  # seconds from baseline
large_time = 1742.74  # seconds from baseline
time_ratio = large_time / base_time

print(f"\nProcessing Time:")
print(f"  Base:  {base_time/60:.2f} minutes")
print(f"  Large: {large_time/60:.2f} minutes")
print(f"  Ratio: {time_ratio:.1f}x slower")

print(f"\nAccuracy Gain per Unit Time:")
if improvement > 0:
    efficiency = (improvement * 100) / (large_time - base_time) * 60
    print(f"  {efficiency:.4f}% WER reduction per minute of extra processing")

print(f"\nRECOMMENDATION:")
if improvement > 0.05:
    print("  Use Large model - significant accuracy gain ({:.1f}% improvement)".format(pct_improvement))
elif improvement > 0.02:
    print("  Use Large for important content, Base for drafts")
else:
    print("  Base model may be sufficient - minimal difference")

print("\n" + "="*80)
print("WER CALCULATION COMPLETE")
print("="*80)

# Save results
results = {
    'base_model': wer_base,
    'large_model': wer_large,
    'improvement': {
        'absolute': improvement,
        'relative_pct': pct_improvement
    },
    'processing_time': {
        'base_seconds': base_time,
        'large_seconds': large_time,
        'ratio': time_ratio
    }
}

import json
with open('Audio_Local_tests/baseline_output/wer_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: Audio_Local_tests/baseline_output/wer_results.json")
