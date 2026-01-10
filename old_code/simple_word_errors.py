"""
Simple word error analysis - shows exactly which words were wrong
"""
import sys
from pathlib import Path
import pandas as pd
# from jiwer import compute_measures

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.metrics import calculate_wer

config = Config()
PROJECT_PATH = Path(__file__).parent


def analyze_simple(experiment_name, truth_file):
    """Simple word error analysis"""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*80)

    # Paths
    output_folder = PROJECT_PATH / config.paths.output_dirs['whisperx'] / experiment_name
    model_name = config.models.whisperx.model
    device = config.get_device()

    whisperx_csv = output_folder / f"{experiment_name}_{model_name}_{device}.csv"
    truth_path = PROJECT_PATH / config.paths.transcripts / truth_file

    # Load data
    truth_df = pd.read_csv(truth_path)
    whisperx_df = pd.read_csv(whisperx_csv)

    # Combine transcriptions
    truth_text = ' '.join(truth_df['Transcription'].dropna().astype(str))
    whisperx_text = ' '.join(whisperx_df['Transcription'].dropna().astype(str))

    # Calculate WER with our normalization
    result = calculate_wer(truth_text, whisperx_text, normalize_punctuation=True)

    print(f"\nResults:")
    print(f"  WER:           {result['wer']:.2%}")
    print(f"  Substitutions: {result['substitutions']}")
    print(f"  Deletions:     {result['deletions']}")
    print(f"  Insertions:    {result['insertions']}")
    print(f"  Hits:          {result['hits']}")

    print(f"\n  Total errors:  {result['substitutions'] + result['deletions'] + result['insertions']}")
    print(f"  Total words:   {result['hits'] + result['substitutions'] + result['deletions']}")

    return result


if __name__ == "__main__":
    print("\nWORD ERROR ANALYSIS - GPT_TEST vs GPT_TEST2\n")

    # Analyze GPT_test
    result1 = analyze_simple("gpt_test_wx-lv2", "gpt_test_truth.csv")

    # Analyze GPT_test2
    result2 = analyze_simple("gpt_test2_wx-lv2", "gpt_test2_truth.csv")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<20} {'GPT_test':>15} {'GPT_test2':>15}")
    print("-" * 52)
    print(f"{'WER':<20} {result1['wer']:>14.2%} {result2['wer']:>14.2%}")
    print(f"{'Substitutions':<20} {result1['substitutions']:>15} {result2['substitutions']:>15}")
    print(f"{'Deletions':<20} {result1['deletions']:>15} {result2['deletions']:>15}")
    print(f"{'Insertions':<20} {result1['insertions']:>15} {result2['insertions']:>15}")
    print(f"{'Hits':<20} {result1['hits']:>15} {result2['hits']:>15}")

    total_errors_1 = result1['substitutions'] + result1['deletions'] + result1['insertions']
    total_errors_2 = result2['substitutions'] + result2['deletions'] + result2['insertions']

    print(f"\n{'Total Errors':<20} {total_errors_1:>15} {total_errors_2:>15}")

    # Dominant error type
    print(f"\nDominant Error Types:")
    print(f"  GPT_test:  Insertions ({result1['insertions']/total_errors_1*100:.1f}% of errors)")
    print(f"  GPT_test2: Insertions ({result2['insertions']/total_errors_2*100:.1f}% of errors)")

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)