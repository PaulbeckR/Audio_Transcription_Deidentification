"""
Calculate WER (Word Error Rate) for baseline test results.

Compares Whisper Base and Large outputs against ground truth.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.metrics import calculate_wer_from_files

# File paths
GROUND_TRUTH = "Audio_Local_tests/transcription_test/hamlet_test_truth.xlsx"
BASE_OUTPUT = "Audio_Local_tests/baseline_output/hamlet_test_whisper_base.csv"
LARGE_OUTPUT = "Audio_Local_tests/baseline_output/hamlet_test_whisper_large.csv"

print("="*80)
print("WER CALCULATION - Baseline Test Results")
print("="*80)

print(f"\nGround Truth: {GROUND_TRUTH}")
print(f"Base Model Output: {BASE_OUTPUT}")
print(f"Large Model Output: {LARGE_OUTPUT}")

# Try different possible column names
possible_columns = ['Transcription', 'Text', 'Transcript', 'transcription', 'text']

print("\n" + "="*80)
print("WHISPER BASE MODEL")
print("="*80)

success = False
for col in possible_columns:
    try:
        print(f"\nTrying column: '{col}'...")
        wer_base = calculate_wer_from_files(
            GROUND_TRUTH,
            BASE_OUTPUT,
            text_column=col
        )

        print(f"\n✅ Success with column '{col}'!")
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

        success = True
        text_col_name = col
        break

    except Exception as e:
        print(f"  Failed: {e}")
        continue

if not success:
    print("\n❌ Could not calculate WER for base model")
    print("Available columns in ground truth file:")
    import pandas as pd
    try:
        df = pd.read_excel(GROUND_TRUTH)
        print(f"  {list(df.columns)}")
    except:
        pass
    sys.exit(1)

print("\n" + "="*80)
print("WHISPER LARGE MODEL")
print("="*80)

try:
    wer_large = calculate_wer_from_files(
        GROUND_TRUTH,
        LARGE_OUTPUT,
        text_column=text_col_name
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
    print(f"\n❌ Error calculating WER for large model: {e}")
    wer_large = None

# Comparison
if wer_large:
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    improvement = wer_base['wer'] - wer_large['wer']
    pct_improvement = (improvement / wer_base['wer']) * 100 if wer_base['wer'] > 0 else 0

    print(f"\nBase Model WER:  {wer_base['wer']:.2%}")
    print(f"Large Model WER: {wer_large['wer']:.2%}")
    print(f"\nImprovement: {improvement:.2%} ({pct_improvement:.1f}% relative reduction)")

    if improvement > 0:
        print("✅ Large model is more accurate")
    elif improvement < 0:
        print("⚠️ Base model performed better (unexpected)")
    else:
        print("= Models performed equally")

    # Cost-benefit analysis
    print("\n" + "-"*80)
    print("COST-BENEFIT ANALYSIS")
    print("-"*80)

    # From baseline metrics
    base_time = 105.21  # seconds
    large_time = 1742.74  # seconds
    time_ratio = large_time / base_time

    print(f"\nProcessing Time:")
    print(f"  Base:  {base_time/60:.2f} minutes")
    print(f"  Large: {large_time/60:.2f} minutes")
    print(f"  Ratio: {time_ratio:.1f}x slower")

    print(f"\nAccuracy Improvement per Minute of Processing:")
    if improvement > 0:
        efficiency = (improvement * 100) / (large_time / 60)
        print(f"  {efficiency:.3f}% WER reduction per minute of extra processing")

    print(f"\nRecommendation:")
    if improvement > 0.05:  # More than 5% absolute improvement
        print("  ✅ Use Large model - significant accuracy gain worth the time")
    elif improvement > 0.02:  # 2-5% improvement
        print("  ⚖️ Use Large for important content, Base for drafts")
    else:  # Less than 2% improvement
        print("  💡 Base model may be sufficient - minimal accuracy difference")

print("\n" + "="*80)
print("WER CALCULATION COMPLETE")
print("="*80)
